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
        For each token in the input:
        - Gets hidden states at that position
        - Projects them onto signed concept directions
        - Directly collects per-token, per-layer scores and their mean
        """
        input_ids = self.tokenizer.tokenize(input)
        scores = []
        score_means = []

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

            normal_scores = []
            mean_scores = []

            for layer in self.hidden_layers:
                h = hidden_states[layer]

                # TODO: explore recentering h here - doesn't seem to be needed but may provide a clearer picture?
                # Original uses h_train_means to do this..

                direction = self.directions[layer]
                score = project_onto_direction(h, direction)[0]

                normal_scores.append(score)

                if layer in mean_layers:
                    mean_scores.append(score)

            scores.append(normal_scores)
            score_means.append(np.mean(mean_scores))

        return input_ids, scores, score_means

    def initialize(
        self,
        train_inputs: list[DatasetEntry],
        batch_size: int = 8,
    ):
        """Initializes the rep reader! Run this first"""
        # Only pca diff for now
        directions = read_representations(
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

    def reset_reader():
        """TODO: make this"""
        pass
