"""
New version - updates read_reprsentation to
1. calculate & return h_train_means
2. Completely change how we calculate the directionality, from a poll to ideally ground-truth.
"""

import dataclasses
import os
import typing
import warnings
from typing import Optional, Tuple

import gguf
import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .control import ControlModel, model_layer_list
from .saes import Sae


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str
    flip: Optional[bool] = None
    # If true, flip the data. Used in training to avoid overfitting to pairs
    # TODO: This is necessary for the rep reader - is it necessary for control?


@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]
    h_train_means: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """
        with torch.inference_mode():
            dirs, means = read_representations(
                model,
                tokenizer,
                dataset,
                **kwargs,
            )
        return cls(
            model_type=model.config.model_type, directions=dirs, h_train_means=means
        )

    @classmethod
    def train_with_sae(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        sae: Sae,
        dataset: list[DatasetEntry],
        *,
        decode: bool = True,
        method: typing.Literal["pca_diff", "pca_center", "umap"] = "pca_center",
        **kwargs,
    ) -> "ControlVector":
        """
        Like ControlVector.train, but using an SAE. It's better! WIP.


        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            sae (saes.Sae): See the `saes` module for how to load this.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                decode (bool, optional): Whether to decode the vector to make it immediately usable.
                    If not, keeps it as monosemantic SAE features for introspection, but you will need to decode it manually
                    to use it. Defaults to True.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_center"! This is different
                    than ControlVector.train, which defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """

        def transform_hiddens(hiddens: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            sae_hiddens = {}
            for k, v in tqdm.tqdm(hiddens.items(), desc="sae encoding"):
                sae_hiddens[k] = sae.layers[k].encode(v)
            return sae_hiddens

        with torch.inference_mode():
            dirs, _ = read_representations(
                model,
                tokenizer,
                dataset,
                transform_hiddens=transform_hiddens,
                method=method,
                **kwargs,
            )

            final_dirs = {}
            if decode:
                for k, v in tqdm.tqdm(dirs.items(), desc="sae decoding"):
                    final_dirs[k] = sae.layers[k].decode(v)
            else:
                final_dirs = dirs

        return cls(model_type=model.config.model_type, directions=final_dirs)

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained ControlVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

        ```python
        vector = ControlVector.train(...)
        vector.export_gguf("path/to/write/vector.gguf")
        ```
        ```
        """

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "ControlVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data

        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.number) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(1 / other)


def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal["pca_diff", "pca_center", "umap"] = "pca_diff",
    transform_hiddens: (
        typing.Callable[[dict[int, np.ndarray]], dict[int, np.ndarray]] | None
    ) = None,
) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Extract concept directions for contrasting pairs.
    For every layer, return a 1‑D direction *unit* vector that scores
    the *positive* text higher than the *negative* text.

    • pca_diff   – train rows are (positive − negative); sign fixed implicitly
    • pca_center – train rows are centered points; sign fixed by mean_delta test
    • umap       – raw/centered rows; sign fixed by mean_delta test
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # Flip is necessary to prevent PCA from learning order
    # Record the flip into pos_mask - we will use this at the end to make sure that directions point towards positive
    flattened_strs = []
    pos_mask = []  # True -> positive
    for input in inputs:
        if input.flip:
            flattened_strs += [
                input.negative,
                input.positive,
            ]
            pos_mask += [False, True]
        else:
            flattened_strs += [input.positive, input.negative]
            pos_mask += [True, False]

    pos_mask = np.asarray(pos_mask, dtype=bool)

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, flattened_strs, hidden_layers, batch_size
    )
    if transform_hiddens is not None:
        layer_hiddens = transform_hiddens(layer_hiddens)

    directions: dict[int, np.ndarray] = {}
    h_train_means: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers, desc="fitting directions"):
        h = layer_hiddens[layer]  # (2*N, d)
        pos = h[pos_mask]  # (N, d)
        neg = h[~pos_mask]  # (N, d)
        assert h.shape[0] == len(inputs) * 2

        # Use this later to recenter direction (for reader)
        h_train_means[layer] = h.mean(axis=0).astype(np.float32)

        if method == "pca_diff":
            train = pos - neg  # Sign fixed (pos-neg)
        elif method == "pca_center":
            center = (pos + neg) / 2
            train = np.vstack((pos - center, neg - center))
        elif method == "umap":
            train = np.vstack((pos, neg))
        else:
            raise ValueError("unknown method " + method)

        if method != "umap":
            pca = PCA(n_components=1, whiten=False).fit(train)
            v = pca.components_[0].astype(np.float32)
        else:
            import umap  # type: ignore

            emb = umap.UMAP(n_components=1).fit_transform(train).astype(np.float32)
            v = (train * emb).sum(axis=0) / emb.sum()
            v = v.astype(np.float32)

        # Check if direction is pointing wrong way & correct it (only needed for non pca-diff but doesn't hurt)
        # This replaces the not perfect check of project_onto_direction
        mean_delta = (pos - neg).mean(axis=0)
        if mean_delta @ v < 0:
            v = -v

        # Unit vector
        directions[layer] = v / np.linalg.norm(v)

    return directions, h_train_means


def batched_get_hiddens(
    model: PreTrainedModel,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
    rep_token: Optional[int] = None,
    hide_progress: Optional[bool] = None,
) -> dict[int, np.ndarray]:
    """
    Changed this to add a rep_token. This is necessary when reading w/ the reader (getting H_test).
    If no rep token is passed it defaults to False, and uses the last non padding index

    Also added hide_progress flag

    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}

    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs, disable=hide_progress):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]

            for i in range(len(batch)):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                token_index = last_non_padding_index if rep_token is None else rep_token

                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][token_index]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    hidden_states[layer].append(hidden_state)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}
