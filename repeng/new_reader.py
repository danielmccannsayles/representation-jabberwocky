"""
From scratch reader - most up to date as of June 2

Some assumptions:

Default control:
1. To match w/ control I'm assuming `normalize` is false (so the layer's activation is not rescaled after)
2. Assuming that the operator for adding the direction vector is +.

"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from repeng.extract import ControlVector


### Simplified batched_get_hiddens
# Not actually used here - eventually will move this to  extract
def batched_get_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[str],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Passes `inputs` through `model`, returning hidden activations for each transformer layer
    Returns a dict mapping layer index (0 = first transformer layer)
    to a (n_inputs, hidden_dim) array.
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]

    hiddens_by_layer = None  # Reassign once in loop
    with torch.no_grad():
        for batch in tqdm(batched_inputs, desc="Getting hidden states"):
            toks = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
            mask = toks["attention_mask"]

            out = model(**toks, output_hidden_states=True)
            hs = out.hidden_states
            del out

            if hiddens_by_layer is None:
                hiddens_by_layer = {i: [] for i in range(len(hs) - 1)}

            for i, _ in enumerate(batch):
                idx = mask[i].nonzero(as_tuple=True)[0][-1].item()
                for layer in range(1, len(hs)):
                    vec = hs[layer][i][idx].cpu().float().numpy()
                    hiddens_by_layer[layer - 1].append(vec)

    return {k: np.vstack(v) for k, v in hiddens_by_layer.items()}


## Get the hiddens of one string @ a token
def get_hiddens_at_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input: str,
    rep_token: int,
) -> dict[int, np.ndarray]:
    """
    Passes a string input through model, returning hidden activations for each transformer layer
    at the specified token position.
    Returns a dict mapping layer index (1 = first transformer layer)
    to a (hidden_dim,) array.
    """

    with torch.no_grad():
        toks = tokenizer(input, return_tensors="pt").to(model.device)

        out = model(**toks, output_hidden_states=True)
        hs = out.hidden_states
        del out

        hiddens_by_layer = {}
        for layer in range(1, len(hs) - 1):
            vec = hs[layer][0][rep_token].cpu().float().numpy()
            hiddens_by_layer[layer] = vec

    return hiddens_by_layer


### Reader class!
class NewReader:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Concept directions
        self.concept_directions = {}

        # Baseline (mean) (used to get accurate reading)
        self.baseline = {}

    def read(
        self,
        input: str,
        mean_layers: range,
    ):
        """
        Reads a string. Returns scores.

        mean_layers are certain layers we want to take the mean of..

        For each token in the input:
        - Gets hidden states at that position
        - Projects them onto normalized concept directions
        - Directly collects per-token, per-layer scores and their mean
        """
        input_ids = self.tokenizer.tokenize(input)
        scores = []
        score_means = []

        for pos in range(len(input_ids)):
            rep_token = -len(input_ids) + pos

            hiddens_by_layer = get_hiddens_at_token(
                self.model, self.tokenizer, input, rep_token
            )

            normal_scores = []
            mean_scores = []
            for layer_idx, hiddens in hiddens_by_layer.items():
                direction = self.concept_directions[layer_idx]
                baseline = self.baseline[layer_idx]

                # Center hiddens
                centered_hiddens = hiddens - baseline

                # Get unit direction out of concept direction
                unit_direction = direction / np.linalg.norm(direction)

                # Project centered hiddens onto the unit direction
                score = float(centered_hiddens @ unit_direction)
                normal_scores.append(score)

                if layer_idx in mean_layers:
                    mean_scores.append(score)

            # Add scores, calculate mean for mean scores
            scores.append(normal_scores)
            score_means.append(np.mean(mean_scores))

        return input_ids, scores, score_means

    def set_vector(
        self,
        vector: "ControlVector",
        coeff: Optional[float] = 1,
    ):
        """Sets a direction vector. Multiplies it by coeff to match the control.

        You should set this to an empiricially validated coeff"""
        directions = vector.directions
        baseline = vector.baseline

        self.concept_directions = {k: v * coeff for k, v in directions.items()}
        self.baseline = baseline
