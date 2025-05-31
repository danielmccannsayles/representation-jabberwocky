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

from repeng.extract import DatasetEntry, batched_get_hiddens, read_representations


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

            hidden_states = batched_get_hiddens(
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

                # Recenter layer_hidden if we have the 'means' (pun intended)
                if self.h_train_means:
                    mean = torch.Tensor(self.h_train_means[layer]).to(self.model.device)
                    layer_hidden = torch.Tensor(layer_hidden).to(self.model.device)

                    # If no mean for layer, we just re-calculate it? Idk just preserving functionality
                    if mean is None:
                        mean = torch.mean(layer_hidden, axis=0, keepdims=True).to(
                            self.model.device
                        )

                    layer_hidden = layer_hidden - mean

                # Transform
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
