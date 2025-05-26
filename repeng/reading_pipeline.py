"""
Continuing the same idea - recreate the old code. This will contain what used to be the rep_reading_pipeline
"""

from typing import List, Optional, Union

import numpy as np
import torch
import tqdm

from repeng.extract import DatasetEntry
from repeng.reading import PCARepReader


class RepReading:
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(**kwargs)

        # I added model & tokenizer - in the past they were implicitly passed in through the HF pipeline class. We use them in batched_get_hiddens, since we need to run the model & tokenizer
        self.model = model
        self.tokenizer = tokenizer

    def _get_hidden_states(
        self,
        outputs,
        rep_token=-1,
        hidden_layers: Union[List[int], int] = -1,
    ):
        """I think this just gets hidden states from the outputs of a model -> i.e. its just a small data transformation"""
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs["hidden_states"][layer]
            hidden_states = hidden_states[:, rep_token, :].detach()
            if hidden_states.dtype == torch.bfloat16:
                hidden_states = hidden_states.float()
            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers

    # Removed this - kept for posterity
    # def _sanitize_parameters()

    # Removed this - looks like it only had to do w/ image models
    # def preprocess()

    # This is the main call (used to be _forward)
    # This takes in the rep_reader & runs the model, then transforms the outputs w/ the rep_reader
    # TODO: this should be similar if not the same to batched_get_hiddens
    def forward(
        self,
        test_inputs,
        hidden_layers,
        batch_size,
        rep_reader=None,
        component_index=0,
        rep_token: Optional[int] = None,
    ):
        """ """
        # Removed encoder/decoder handling. TODO: see if this causes problems (it shouldn't)
        # get model hidden states and optionally transform them with a RepReader
        # We need to get input_ids manually -> this was done implicitly by the transformer
        # TODO: It would be nice to not constantly have to switch from DatasetEntry to str everywhere.

        # TODO: I think that what they were doing earlier is analogous to batched_get_hiddens :D.
        if isinstance(test_inputs, list) and all(
            isinstance(x, DatasetEntry) for x in test_inputs
        ):
            test_strs = [s for ex in test_inputs for s in (ex.positive, ex.negative)]

        else:  # assume list of strings
            test_strs = test_inputs

        tokenized_inputs = self.tokenizer(
            test_strs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**tokenized_inputs, output_hidden_states=True)

        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers)

        # TODO: can we use the batched_get_hiddens?
        # hidden_states = batched_get_hiddens(
        #     self.model, self.tokenizer, test_strs, hidden_layers, batch_size, rep_token
        # )

        if rep_reader is None:
            return hidden_states

        return rep_reader.transform(hidden_states, hidden_layers, component_index)

    # Uses PCA
    def get_directions(
        self,
        train_inputs: list[DatasetEntry],
        hidden_layers: Union[str, int] = -1,
        n_difference: int = 1,
        batch_size: int = 8,
        train_labels: List[int] = None,
    ):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list):
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        # initialize a DirectionFinder
        direction_finder = PCARepReader()

        # if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            # the order is [positive, negative, positive, negative, ...]
            train_strs = [s for ex in train_inputs for s in (ex.positive, ex.negative)]
            hidden_states = batched_get_hiddens(
                self.model,
                self.tokenizer,
                train_strs,
                hidden_layers,
                batch_size,
            )

            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = (
                        relative_hidden_states[layer][::2]
                        - relative_hidden_states[layer][1::2]
                    )

        # get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            relative_hidden_states,
            hidden_layers,
        )
        for layer in direction_finder.directions:
            if isinstance(direction_finder.directions[layer], np.ndarray):
                direction_finder.directions[layer] = direction_finder.directions[
                    layer
                ].astype(np.float32)

        # TODO: fix train_labels. For now we just mock it
        mock_train_labels = [[1, 0] for _ in train_inputs]
        signs = direction_finder.get_signs(
            hidden_states, mock_train_labels, hidden_layers
        )
        direction_finder.direction_signs = signs

        return direction_finder


# Taken from repeng
def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
    rep_token: Optional[int] = None,
) -> dict[int, np.ndarray]:
    """
    Changed this to add a rep_token. This is necessary for the reader to work.
    If no rep token is passed it defaults to False, and uses the last non padding index

    OLD desc:
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}

    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]

            for i in range(len(batch)):
                # rep token calculation (-1 defaults to )
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


#### NEW stuff


### I'm going to flatten this pipeline into a few functions and get it working correctly
def get_hidden_states(
    outputs,
    rep_token=-1,
    hidden_layers: Union[List[int], int] = -1,
):
    hidden_states_layers = {}
    for layer in hidden_layers:
        hidden_states = outputs["hidden_states"][layer]
        hidden_states = hidden_states[:, rep_token, :].detach()
        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        hidden_states_layers[layer] = hidden_states.detach()

    return hidden_states_layers


def string_to_hiddens(
    model,
    tokenizer,
    train_strs,
    rep_token,
    hidden_layers,
    batch_size,
):
    """Have to manually preprocess & batch & call"""
    hidden_states = {layer: [] for layer in hidden_layers}

    for start in range(0, len(train_strs), batch_size):
        batch_inputs = train_strs[start : start + batch_size]

        toks = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**toks, output_hidden_states=True)

        batch_hiddens = get_hidden_states(outputs, rep_token, hidden_layers)

        for layer in hidden_layers:
            # .cpu() so we don’t pin GPU memory; .numpy() for consistency
            hidden_states[layer].append(batch_hiddens[layer].cpu().numpy())

    # concatenate the pieces exactly the same way the pipeline helper did
    return {layer: np.vstack(chunks) for layer, chunks in hidden_states.items()}


def create_rep_reader(
    model,
    tokenizer,
    train_inputs: list[DatasetEntry],
    rep_token: int = -1,
    hidden_layers: Union[str, int] = -1,
    n_difference: int = 1,
    batch_size: int = 8,
):
    if not isinstance(hidden_layers, list):
        assert isinstance(hidden_layers, int)
        hidden_layers = [hidden_layers]

    direction_finder = PCARepReader()

    # PCA needs hidden state data for training set
    hidden_states = None
    relative_hidden_states = None

    train_strs = [s for ex in train_inputs for s in (ex.positive, ex.negative)]
    # get raw hidden states for the train inputs
    hidden_states = string_to_hiddens(
        model,
        tokenizer,
        train_strs,
        rep_token,
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

        # TODO: fix train_labels. For now we just mock it
        mock_train_labels = [[1, 0] for _ in train_inputs]
    direction_finder.direction_signs = direction_finder.get_signs(
        hidden_states, mock_train_labels, hidden_layers
    )

    return direction_finder
