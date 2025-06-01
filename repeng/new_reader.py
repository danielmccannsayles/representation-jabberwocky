"""
I've been trying to replicate the original paper, but I'm not sure its reader is correct or super useful.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


### Simplified batched_get_hiddens
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
        for batch in tqdm(batched_inputs):
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
    Returns a dict mapping layer index (0 = first transformer layer)
    to a (hidden_dim,) array.
    """
    with torch.no_grad():
        toks = tokenizer(input, return_tensors="pt").to(model.device)

        out = model(**toks, output_hidden_states=True)
        hs = out.hidden_states
        del out

        hiddens_by_layer = {}
        for layer in range(1, len(hs)):
            vec = hs[layer][0][rep_token].cpu().float().numpy()
            hiddens_by_layer[layer - 1] = vec

    return hiddens_by_layer


### Concept vector stuff
@dataclass
class DatasetEntry:
    positive: str
    negative: str


def read_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[DatasetEntry],
    batch_size: int = 32,
):
    """Get a concept vector using the provided contrasting data set"""
    # TODO: make train_strs out of dataset
    # Dataset has a positive & a negative string
    hiddens_by_layer = batched_get_hiddens(model, tokenizer, train_strs, batch_size)

    #
    for layer_idx, hiddens in tqdm(hiddens_by_layer.items()):
        pass


### Reader class!
class NewReader:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        # Concept directions
        self.concept_directions = {}

    def read(
        self,
        input: str,
        mean_layers: range,
    ):
        """
        Reads a string. Returns scores

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

            hidden_states = get_hiddens_at_token(
                self.model, self.tokenizer, input, rep_token
            )

            normal_scores = []
            mean_scores = []

            for layer in self.hidden_layers:
                h = hidden_states[layer]
                direction = self.directions[layer]

                score = float(h @ direction)
                normal_scores.append(score)

                if layer in mean_layers:
                    mean_scores.append(score)

            scores.append(normal_scores)
            score_means.append(np.mean(mean_scores))

        return input_ids, scores, score_means

    def set_vector(self, directions: dict[int, np.ndarray]):
        """Sets a direciton vector."""

        # TODO: switch to using the same layer convention (also stop hard-coding this value)
        transformed_directions = {-(32 - k): v for k, v in directions.items()}
        self.directions = transformed_directions

    def reset_reader():
        """TODO: make this"""
        pass
