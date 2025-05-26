"""
DANIEL
New strategy - implement the rep reading from the old code.
"""

from itertools import islice

import numpy as np
import torch
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


def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x, axis=0, keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x - mean


class PCARepReader:
    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components
        # TODO: what are h_train_means??
        self.H_train_means = {}

        # TODO: add hidden_states, directions, direction signs

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
        signs = {}

        for layer in hidden_layers:
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

    def train_reader():
        """TODO: make this"""
        pass

    def reset_reader():
        """TODO: make this"""
        pass
