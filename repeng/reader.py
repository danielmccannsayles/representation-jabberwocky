"""
DANIEL
New strategy - implement the rep reading from the old code.
"""

from itertools import islice
from typing import Optional

import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from repeng.extract import DatasetEntry, read_representations


### Helpers
# Taken from repeng, slightly modified
def batched_get_hiddens2(
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


# Taken from repeng.
def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,).

    Returns np"""
    # added this to avoid an error - don't know if these are supposed to be on CPU but whatever
    # TODO fix this upstream.
    if isinstance(H, torch.Tensor):
        H = H.cpu().numpy()
    if isinstance(direction, torch.Tensor):
        direction = direction.cpu().numpy()
    mag = np.linalg.norm(direction)
    print(f"multiplying {H.shape} by {direction.shape}, dividing by {mag.shape}")
    assert not np.isinf(mag)
    return (H @ direction) / mag


def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x, axis=0, keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x - mean


### Actual class!
class PCARepReader:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_layers: list[int],
    ):
        super().__init__()
        # The model & tokenizer should stay the same, since they're used to intialize the reader
        self.model = model
        self.tokenizer = tokenizer

        # The means - used to recenter
        self.h_train_means = {}

        # Directions are used in get_signs and in transform
        self.directions = {}

        self.hidden_layers = hidden_layers
        self.signs = {}

    def read(
        self,
        input: str,
        mean_layers: range,
        batch_size,
    ):
        """
        Consolidated version of the old `transform` + `read`.

        For each token position:
        1. Grab hidden state at that token (via `rep_token`).
        2. (Optionally) recentre with the same means used in training.
        3. Project onto the stored direction vectors.
        4. Apply the stored sign.
        5. Collect per‑layer scores and the mean across `mean_layers`.
        """
        input_ids = self.tokenizer.tokenize(input)

        scores = []  # per‑token, per‑layer
        score_means = []  # per‑token

        for pos in range(len(input_ids)):
            rep_token = -len(input_ids) + pos

            hidden_states = batched_get_hiddens2(
                self.model,
                self.tokenizer,
                inputs=[input],
                hidden_layers=self.hidden_layers,
                batch_size=batch_size,
                rep_token=rep_token,
                hide_progress=True,
            )

            per_layer_scores = []
            per_layer_mean_pool = []

            for layer in self.hidden_layers:
                layer_hidden = hidden_states[layer]

                # Transform
                if hasattr(self, "H_train_means"):
                    layer_hidden = recenter(
                        layer_hidden, mean=self.h_train_means[layer]
                    )

                projected = project_onto_direction(layer_hidden, self.directions[layer])

                signed_score = projected[0] * self.signs[layer]
                per_layer_scores.append(signed_score)

                if layer in mean_layers:
                    per_layer_mean_pool.append(signed_score)

            scores.append(per_layer_scores)
            score_means.append(np.mean(per_layer_mean_pool))

        return input_ids, scores, score_means

    def initialize(
        self,
        train_inputs: list[DatasetEntry],
        batch_size: int = 8,
    ):
        """Initializes the rep reader! Run this first"""
        # Only pca diff for now
        directions, signs, h_train_means = read_representations(
            self.model,
            self.tokenizer,
            train_inputs,
            self.hidden_layers,
            batch_size,
            "pca_diff",
        )
        # TODO: switch to using the same layer convention (also stop hard-coding this value)
        transformed_directions = {-(32 - k): v for k, v in directions.items()}
        self.directions = transformed_directions

        transformed_means = {-(32 - k): v for k, v in h_train_means.items()}
        self.h_train_means = transformed_means

        transformed_signs = {-(32 - k): -v for k, v in signs.items()}
        self.signs = transformed_signs

    def reset_reader():
        """TODO: make this"""
        pass
