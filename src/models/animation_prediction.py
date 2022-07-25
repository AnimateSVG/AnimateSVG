import torch
import torch.nn as nn
import time

from src.models import config


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class AnimationPredictor(nn.Module):
    """ Animation prediction model to be trained using genetic algorithm. """

    def __init__(self, input_size=config.a_input_size, hidden_sizes=config.a_hidden_sizes,
                 out_sizes=config.a_out_sizes):
        """
        Args:
            input_size (int): Number of neurons in input layer (input dimension).
            hidden_sizes (list): Number of neurons in each hidden layer. Must be of length=2.
            out_sizes (list): Number of neurons in each output layer. Must be of length=2.

        """

        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.out_sizes = out_sizes

        # Hidden Layers
        self.hidden_1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.hidden_2 = nn.Linear(self.input_size + self.out_sizes[0], self.hidden_sizes[1])

        # Output Layers
        self.out_1 = nn.Linear(self.hidden_sizes[0], self.out_sizes[0])
        self.out_2 = nn.Linear(self.hidden_sizes[1], self.out_sizes[1])

        self.apply(weights_init)

    # Forward Pass
    def forward(self, X):
        """ Forward pass to generate animations.

        Args:
            X (np.ndarray): Path vectors for which animations are to be generated.

        Returns:
            torch.Tensor: Generated animation vectors.

        """
        # forward pass of model two: predict type of animation (choice out of 6)
        h_1 = torch.relu(self.hidden_1(X))
        y_1 = nn.functional.softmax(self.out_1(h_1), dim=1)
        # max_indices = y_1.argmax(1)
        # y_1 = torch.tensor([[1 if j == max_indices[i] else 0 for j in range(self.out_sizes[0])]
        #                     for i in range(X.shape[0])])

        # forward pass of model three: predict animation parameters
        h_2 = torch.relu(self.hidden_2(torch.cat((X, y_1), 1)))
        y_2 = torch.sigmoid(self.out_2(h_2))

        output = torch.cat((y_1, y_2, X), 1)

        return output
