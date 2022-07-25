import torch
import torch.nn as nn

from src.models import config
from src.preprocessing.sm_label_transformer import *


class OrdinalClassifierFNN(nn.Module):
    """ Ordinal classification neural network used as surrogate model for the evaluation of path animations. """

    # Ordinal Regression
    def __init__(self, num_classes, layer_sizes=[36, 28]):
        """
        Args:
            num_classes (int): Number of classes in classification task.
            layer_sizes (list): Number of neurons in input and hidden layer.

        """
        super().__init__()

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))

        self.coral_weights = nn.Linear(layer_sizes[-1], 1, bias=False)
        self.coral_bias = torch.nn.Parameter(
            torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))

    # Forward Pass
    def forward(self, X):
        for layer in self.hidden:
            X = torch.relu(layer(X))
        logits = self.coral_weights(X) + self.coral_bias
        return logits


def predict(path_animation_vectors):
    """ Use pre-trained surrogate model (src.models.config.sm_fnn_path) to evaluate given path animations.

    Args:
        path_animation_vectors (torch.Tensor): Concatenated path and animation vectors to evaluate.

    Returns:
        np.ndarray: Decoded evaluation score (between 0 and 4).

    """
    sm = OrdinalClassifierFNN(num_classes=5, layer_sizes=[38, 28])
    sm.load_state_dict(torch.load(config.sm_fnn_path))
    sm_output = sm(path_animation_vectors)
    return decode_classes(sm_output)


def predict_backpropagation(path_animation_vectors):
    sm = OrdinalClassifierFNN(num_classes=5, layer_sizes=[38, 28])
    sm.load_state_dict(torch.load(config.sm_fnn_path))
    sm_output = sm(np.array(path_animation_vectors))
    decoded_sm_output = -(decode_classes(sm_output))
    return decoded_sm_output
