"""
Continuing the same idea - recreate the old code. This will contain what used to be the rep_reading_pipeline
"""

from typing import Optional

import numpy as np
import torch
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from repeng.extract import DatasetEntry
from repeng.reader import PCARepReader


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


# TODO: can we do this in an actual batch? / Just better?? !
def rep_read(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input: str,
    hidden_layers,
    mean_layers: range,
    batch_size,
    rep_reader: PCARepReader,
    component_index=0,
):
    """
    Reads a string.
    Goes through it and gets an h_test for each token ->. Then calculates scores according to rep_reader.
    Incorporates 'forward' on the pipeline. Takes in some input, runs it through the model, uses the rep_reader on it."""

    input_ids = tokenizer.tokenize(input)
    results = []

    # For each position (token index) we need to make H_tests
    for pos in range(len(input_ids)):
        rep_token = -len(input_ids) + pos

        hidden_states = batched_get_hiddens2(
            model,
            tokenizer,
            inputs=[input],
            hidden_layers=hidden_layers,
            batch_size=batch_size,
            rep_token=rep_token,
            hide_progress=True,
        )

        H_tests = rep_reader.transform(hidden_states, hidden_layers, component_index)

        results.append(H_tests)

    # Turn results into the scores & the mean scores
    scores = []
    score_means = []
    for result in results:
        mean_scores = []
        normal_scores = []
        for layer in hidden_layers:
            normal_scores.append(
                result[layer][0] * rep_reader.direction_signs[layer][0]
            )

            if layer in mean_layers:
                mean_scores.append(
                    result[layer][0] * rep_reader.direction_signs[layer][0]
                )

        scores.append(normal_scores)
        score_means.append(np.mean(mean_scores))

    return (input_ids, scores, score_means)


def create_rep_reader(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    hidden_layers: list[int],
    train_inputs: list[DatasetEntry],
    n_difference: int = 1,
    batch_size: int = 8,
):
    """Creates a rep_reader and initializes it sort of"""
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

    direction_finder = PCARepReader()

    # PCA needs hidden state data for training set
    hidden_states = None
    relative_hidden_states = None

    # get raw hidden states for the train inputs
    hidden_states = batched_get_hiddens2(
        model,
        tokenizer,
        train_strs,
        hidden_layers,
        batch_size,
    )

    # get differences between pairs
    relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
    for layer in hidden_layers:
        for _ in range(n_difference):
            relative_hidden_states[layer] = (
                relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]
            )

    # get the directions
    direction_finder.directions = direction_finder.get_rep_directions(
        relative_hidden_states,
        hidden_layers,
    )
    for layer in direction_finder.directions:
        if type(direction_finder.directions[layer]) == np.ndarray:
            direction_finder.directions[layer] = direction_finder.directions[
                layer
            ].astype(np.float32)

    direction_finder.direction_signs = direction_finder.get_signs(
        hidden_states, train_labels, hidden_layers
    )

    return direction_finder
