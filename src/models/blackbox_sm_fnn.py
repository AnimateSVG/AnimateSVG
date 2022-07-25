from entmoot.benchmarks import BenchmarkFunction
from entmoot.space.space import Integer
from src.models.ordinal_classifier_fnn import *
import torch


class SurrogateModelFNN(BenchmarkFunction):
    """ Black-box function that is used for evaluation in ENTMOOT optimization. """

    def __init__(self):
        # load best surrogate model (FNN)
        self.sm = OrdinalClassifierFNN(num_classes=5, layer_sizes=[38, 28])
        self.sm.load_state_dict(torch.load(config.sm_fnn_path))
        self.sm.eval()

        # define value restrictions for animation vector
        self.an_statement_dims = [Integer(low=0, high=1) for _ in range(6)]
        self.an_parameters_dims = [(0.0, 1.0) for _ in range(6)]

    def get_bounds(self, n_dim=26):
        """ Get bounds of input variables.

        Args:
            n_dim (int): Number of dimensions in path vector.

        Returns:
            list: Types and bounds of input variables.

        """
        return self.an_statement_dims + self.an_parameters_dims + [(-10.0, 10.0) for _ in range(n_dim)]

    def get_X_opt(self):
        """ Necessary for class to work properly, but not directly used.

        """
        pass

    def _eval_point(self, x):
        """ Evaluates input vector using surrogate model.

        Args:
            x (list): Input vector.

        Returns:
            int: Negative output of surrogate model (as we minimize).

        """
        x = torch.tensor(x)
        logits = self.sm(x)
        output = decode_classes(torch.sigmoid(logits).reshape(1,-1))[0][0]
        return -output
