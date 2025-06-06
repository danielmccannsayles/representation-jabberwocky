"""
Old version as of june 2


"""

from typing import Optional

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from repeng.extract import ControlVector, batched_get_hiddens

# # Taken from repeng.
# def project_onto_direction(H, direction):
#     """Project matrix H (n, d_1) onto direction vector (d_2,).

#     Returns np"""
#     # added this to avoid an error - don't know if these are supposed to be on CPU but whatever
#     # TODO fix this upstream.
#     if isinstance(H, torch.Tensor):
#         H = H.cpu().numpy()
#     if isinstance(direction, torch.Tensor):
#         direction = direction.cpu().numpy()
#     mag = np.linalg.norm(direction)
#     print(f"multiplying {H.shape} by {direction.shape}, dividing by {mag.shape}")
#     assert not np.isinf(mag)
#     return (H @ direction) / mag

# ### Cleaned up version of read_representation for PCA. Left for posterity as of June 2
# def read_representations(
#     model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizerBase,
#     dataset: list[DatasetEntry],
#     batch_size: int = 32,
# ):
#     """Get concept vectors using the provided contrasting dataset."""

#     # Pos, neg
#     train_strs = [s for ex in dataset for s in (ex.positive, ex.negative)]
#     hiddens_by_layer = batched_get_hiddens(model, tokenizer, train_strs, batch_size)

#     concept_directions = {}
#     means = {}
#     for layer_idx, hiddens in tqdm(
#         hiddens_by_layer.items(), desc="Computing concept directions"
#     ):
#         # Means are used to recenter in read()
#         mean = np.mean(hiddens, axis=0, keepdims=True)
#         means[layer_idx] = mean.flatten()  # 2d -> 1d to match concept_directions

#         positive_hiddens = hiddens[::2] - mean  # Even
#         negative_hiddens = hiddens[1::2] - mean  # Odd

#         diff = positive_hiddens - negative_hiddens

#         # Get first principal component
#         pca = PCA(n_components=1).fit(diff)
#         direction = pca.components_[0].astype(np.float32)

#         # PCA doesn't always point in the right direction.
#         positive_projections = positive_hiddens @ direction
#         negative_projections = negative_hiddens @ direction
#         if np.mean(positive_projections) < np.mean(negative_projections):
#             direction = -direction

#         concept_directions[layer_idx] = direction

#     return concept_directions, means


class RepReader:
    """Rep Reading class! The model & tokenizer will be used during reading (they're what the vectors will be pulled out of)"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_layers: list[int],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_layers = hidden_layers

        # Directions & h_train_means come from control vector
        self.directions = {}
        self.h_train_means = {}

        # Multiplies the scores in read()
        self.multiplier: Optional[float] = None

    def read(
        self,
        input: str,
        mean_layers: range,
        batch_size: int,
        multiplier: Optional[float] = None,
    ):
        """
        For each token in the input:
        - Gets hidden states at that position
        - Projects them onto signed concept directions
        - Directly collects per-token, per-layer scores and their mean

        Mean layers: a set of layers to take the mean across. Usually used in sentence view
        Accepts an optional multiplier. Not saved. Pass -1 to invert the concept direction
        """
        assert (
            self.directions and self.h_train_means
        ), "No direction/means. Run set_vector first!"

        scores = []
        score_means = []

        # local > self > 1
        multiplier = (
            multiplier if multiplier else self.multiplier if self.multiplier else 1
        )

        input_ids = self.tokenizer.tokenize(input)
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
                direction = self.directions[layer]
                h = hidden_states[layer]
                mean = self.h_train_means[layer]

                # Re-center w/ h_train_means
                h_centered = h - mean

                score = float(h_centered @ direction) * multiplier

                normal_scores.append(score)

                if layer in mean_layers:
                    mean_scores.append(score)

            scores.append(normal_scores)
            score_means.append(np.mean(mean_scores))

        return input_ids, scores, score_means

    def set_vector(
        self,
        vector: "ControlVector",
        multiplier: Optional[float] = None,
    ):
        """Initializes the rep reader with a control vector - this vector is used to isolate representations.

        Optionally pass in a multiplier, to be used in read().
        Set this to -1 to flip the concept direction.
        Setting it here will set it for the RepReader instance"""
        self.multiplier = multiplier
        directions = vector.directions
        h_train_means = vector.h_train_means

        # TODO: switch to using the same layer convention (also stop hard-coding this value)
        transformed_directions = {-(32 - k): v for k, v in directions.items()}
        self.directions = transformed_directions
        transformed_means = {-(32 - k): v for k, v in h_train_means.items()}
        self.h_train_means = transformed_means
