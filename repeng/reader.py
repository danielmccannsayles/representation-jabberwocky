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

from repeng.extract import DatasetEntry


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
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # added this to avoid an error - don't know if these are supposed to be on CPU but whatever
    # TODO fix this upstream.
    if isinstance(H, torch.Tensor):
        H = H.cpu()
    if isinstance(direction, torch.Tensor):
        direction = direction.cpu()
    mag = np.linalg.norm(direction)
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
        n_components=1,
    ):
        super().__init__()
        # The model & tokenizer should stay the same, since they're used to intialize the reader
        self.model = model
        self.tokenizer = tokenizer

        self.n_components = n_components
        # These h_train_means are used in transform. I think these are like the
        self.H_train_means = {}

        # Directions are used in get_signs and in transform
        self._directions = {}

        # Direction suigns are used
        self._direction_signs = {}

        # This is used a lot but is always the same
        self._hidden_layers = hidden_layers

    def get_rep_directions(self, hidden_states):
        """Get PCA components for each layer"""
        directions = {}

        for layer in self._hidden_layers:
            H_train = hidden_states[layer]
            H_train_mean = H_train.mean(axis=0, keepdims=True)
            self.H_train_means[layer] = H_train_mean
            H_train = recenter(H_train, mean=H_train_mean).cpu()
            H_train = np.vstack(H_train)
            pca_model = PCA(n_components=self.n_components, whiten=False).fit(H_train)

            directions[layer] = (
                pca_model.components_
            )  # shape (n_components, n_features)
            self.n_components = pca_model.n_components_

        # Brought this over from the pipeline. Should consolidate it into the above code??
        for layer in directions:
            if isinstance(directions[layer], np.ndarray):
                directions[layer] = directions[layer].astype(np.float32)

        self._directions = directions

    def get_signs(self, hidden_states, train_labels):
        """TODO: consolidate this. Why do we need to get signs?"""
        signs = {}

        for layer in self._hidden_layers:
            assert (
                hidden_states[layer].shape[0] == len(np.concatenate(train_labels))
            ), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            layer_hidden_states = hidden_states[layer]

            # TODO: Can I get rid of this?
            # NOTE: since scoring is ultimately comparative, the effect of this is moot
            layer_hidden_states = recenter(
                layer_hidden_states, mean=self.H_train_means[layer]
            )

            # get the signs for each component
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):
                transformed_hidden_states = project_onto_direction(
                    layer_hidden_states, self._directions[layer][component_index]
                ).cpu()

                pca_outputs_comp = [
                    list(
                        islice(
                            transformed_hidden_states,
                            sum(len(c) for c in train_labels[:i]),
                            sum(len(c) for c in train_labels[: i + 1]),
                        )
                    )
                    for i in range(len(train_labels))
                ]

                # We do elements instead of argmin/max because sometimes we pad random choices in training
                pca_outputs_min = np.mean(
                    [
                        o[train_labels[i].index(1)] == min(o)
                        for i, o in enumerate(pca_outputs_comp)
                    ]
                )
                pca_outputs_max = np.mean(
                    [
                        o[train_labels[i].index(1)] == max(o)
                        for i, o in enumerate(pca_outputs_comp)
                    ]
                )

                layer_signs[component_index] = np.sign(
                    np.mean(pca_outputs_max) - np.mean(pca_outputs_min)
                )
                if layer_signs[component_index] == 0:
                    layer_signs[component_index] = (
                        1  # default to positive in case of tie
                    )

            signs[layer] = layer_signs
        self._direction_signs = signs

    def transform(self, hidden_states, component_index):
        """Project the hidden states onto the concept directions in self.directions

        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            component_index: index of the component to use from self.directions

        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """

        assert component_index < self.n_components
        transformed_hidden_states = {}
        for layer in self._hidden_layers:
            layer_hidden_states = hidden_states[layer]

            if hasattr(self, "H_train_means"):
                layer_hidden_states = recenter(
                    layer_hidden_states, mean=self.H_train_means[layer]
                )

            # project hidden states onto found concept directions (e.g. onto PCA comp 0)
            H_transformed = project_onto_direction(
                layer_hidden_states, self._directions[layer][component_index]
            )
            transformed_hidden_states[layer] = H_transformed.cpu().numpy()
        return transformed_hidden_states

    def read(
        self,
        input: str,
        mean_layers: range,
        batch_size,
        component_index=0,
    ):
        """The actual load bearing call!

        Reads a string.
        Goes through it and gets an h_test for each token ->. Then calculates scores according to rep_reader.
        Takes in some input, runs it through the model, uses the rep_reader on it."""
        input_ids = self.tokenizer.tokenize(input)
        results = []

        # For each position (token index) we need to make H_tests
        for pos in range(len(input_ids)):
            rep_token = -len(input_ids) + pos

            hidden_states = batched_get_hiddens2(
                self.model,
                self.tokenizer,
                inputs=[input],
                hidden_layers=self._hidden_layers,
                batch_size=batch_size,
                rep_token=rep_token,
                hide_progress=True,
            )

            H_tests = self.transform(hidden_states, component_index)

            results.append(H_tests)

        # Turn results into the scores & the mean scores
        scores = []
        score_means = []
        for result in results:
            mean_scores = []
            normal_scores = []
            for layer in self._hidden_layers:
                normal_scores.append(result[layer][0] * self._direction_signs[layer][0])

                if layer in mean_layers:
                    mean_scores.append(
                        result[layer][0] * self._direction_signs[layer][0]
                    )

            scores.append(normal_scores)
            score_means.append(np.mean(mean_scores))

        return (input_ids, scores, score_means)

    def initialize(
        self,
        train_inputs: list[DatasetEntry],
        n_difference: int = 1,
        batch_size: int = 8,
    ):
        """Initializes the rep reader! Run this first"""
        # Turn inputs into train & labels
        train_strs = []
        train_labels = []
        for input in train_inputs:
            if input.flip:
                train_strs += [input.negative, input.positive]
                train_labels.append([False, True])
            else:
                train_strs += [input.positive, input.negative]
                train_labels.append([True, False])

        # PCA needs hidden state data!
        # get hidden states for the train inputs
        hidden_states = batched_get_hiddens2(
            self.model,
            self.tokenizer,
            train_strs,
            self._hidden_layers,
            batch_size,
        )

        # get differences between pairs
        relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
        for layer in self._hidden_layers:
            for _ in range(n_difference):
                relative_hidden_states[layer] = (
                    relative_hidden_states[layer][::2]
                    - relative_hidden_states[layer][1::2]
                )

        # I'train' rep reader
        self.get_rep_directions(relative_hidden_states)
        self.get_signs(hidden_states, train_labels)

    def reset_reader():
        """TODO: make this"""
        pass
