"""
DANIEL
New strategy - implement the rep reading from the old code.
"""

from abc import ABC, abstractmethod
from itertools import islice

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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


# TODO: Do I neded this?
def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x, axis=0, keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x - mean


class RepReader(ABC):
    """Class to identify and store concept directions.

    Subclasses implement the abstract methods to identify concept directions
    for each hidden layer via (only) PCA

    RepReader instances are used by RepReaderPipeline to get concept scores.

    Directions can be used for downstream interventions."""

    @abstractmethod
    def __init__(self) -> None:
        self.directions = (
            None  # directions accessible via directions[layer][component_index]
        )
        self.direction_signs = (
            None  # direction of high concept scores (mapping min/max to high/low)
        )

    @abstractmethod
    def get_rep_directions(
        self, model, tokenizer, hidden_states, hidden_layers, **kwargs
    ):
        """Get concept directions for each hidden layer of the model

        Args:
            model: Model to get directions for
            tokenizer: Tokenizer to use
            hidden_states: Hidden states of the model on the training data (per layer)
            hidden_layers: Layers to consider

        Returns:
            directions: A dict mapping layers to direction arrays (n_components, hidden_size)
        """
        pass

    def get_signs(self, hidden_states, train_choices, hidden_layers):
        """Given labels for the training data hidden_states, determine whether the
        negative or positive direction corresponds to low/high concept
        (and return corresponding signs -1 or 1 for each layer and component index)

        NOTE: This method assumes that there are 2 entries in hidden_states per label,
        aka len(hidden_states[layer]) == 2 * len(train_choices). For example, if
        n_difference=1, then hidden_states here should be the raw hidden states
        rather than the relative (i.e. the differences between pairs of examples).

        Args:
            hidden_states: Hidden states of the model on the training data (per layer)
            train_choices: Labels for the training data
            hidden_layers: Layers to consider

        Returns:
            signs: A dict mapping layers to sign arrays (n_components,)
        """
        signs = {}

        if self.needs_hiddens and hidden_states is not None and len(hidden_states) > 0:
            for layer in hidden_layers:
                assert hidden_states[layer].shape[0] == 2 * len(train_choices), (
                    f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(train_choices)})"
                )

                signs[layer] = []
                for component_index in range(self.n_components):
                    transformed_hidden_states = project_onto_direction(
                        hidden_states[layer], self.directions[layer][component_index]
                    )
                    projected_scores = [
                        transformed_hidden_states[i : i + 2]
                        for i in range(0, len(transformed_hidden_states), 2)
                    ]

                    outputs_min = [
                        1 if min(o) == o[label] else 0
                        for o, label in zip(projected_scores, train_choices)
                    ]
                    outputs_max = [
                        1 if max(o) == o[label] else 0
                        for o, label in zip(projected_scores, train_choices)
                    ]

                    signs[layer].append(
                        -1 if np.mean(outputs_min) > np.mean(outputs_max) else 1
                    )
        else:
            for layer in hidden_layers:
                signs[layer] = [1 for _ in range(self.n_components)]

        return signs

    def transform(self, hidden_states, hidden_layers, component_index):
        """Project the hidden states onto the concept directions in self.directions

        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            hidden_layers: list of layers to consider
            component_index: index of the component to use from self.directions

        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """

        assert component_index < self.n_components
        transformed_hidden_states = {}
        for layer in hidden_layers:
            layer_hidden_states = hidden_states[layer]

            if hasattr(self, "H_train_means"):
                layer_hidden_states = recenter(
                    layer_hidden_states, mean=self.H_train_means[layer]
                )

            # project hidden states onto found concept directions (e.g. onto PCA comp 0)
            H_transformed = project_onto_direction(
                layer_hidden_states, self.directions[layer][component_index]
            )
            transformed_hidden_states[layer] = H_transformed.cpu().numpy()
        return transformed_hidden_states


class PCARepReader(RepReader):
    """Extract directions via PCA"""

    needs_hiddens = True

    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components
        self.H_train_means = {}

    def get_rep_directions(self, hidden_states, hidden_layers):
        """Get PCA components for each layer"""
        directions = {}

        for layer in hidden_layers:
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

        return directions

    def get_signs(self, hidden_states, train_labels, hidden_layers):
        """this used to accept train labels, as
        [[1,0], [1,0]..] I believe

        """
        signs = {}

        for layer in hidden_layers:
            # assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), (
            #     f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            # )
            layer_hidden_states = hidden_states[layer]

            # NOTE: since scoring is ultimately comparative, the effect of this is moot
            layer_hidden_states = recenter(
                layer_hidden_states, mean=self.H_train_means[layer]
            )

            # get the signs for each component
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):
                transformed_hidden_states = project_onto_direction(
                    layer_hidden_states, self.directions[layer][component_index]
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

        return signs
