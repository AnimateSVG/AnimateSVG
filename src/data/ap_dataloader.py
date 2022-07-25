import torch
import pandas as pd
import numpy as np

from src.models import config


class DatasetAP(torch.utils.data.Dataset):
    """ Surrogate model dataset. """

    # Characterizes a dataset for PyTorch
    def __init__(self, path, train=True):
        """
        Args:
            path (str): Path of folder that contains the data for the surrogate model.
            train (bool): True if training data should be loaded, else False.

        """
        # Read csv file and load data into variables
        dataset = 'train' if train else 'test'
        file_path = f'{path}/selected_paths_{dataset}.csv'

        file_out = pd.read_csv(file_path)
        X = file_out[config.sm_features].values
        y = np.zeros((X.shape[0], config.dim_animation_vector + config.dim_path_vectors))

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def scale(self, fitted_scaler):
        """ Scale the numeric data in the dataset based on the given fitted scaler object.

        Args:
            fitted_scaler (object): Fitted scaler.

        """
        sc = fitted_scaler
        self.X[:, :] = torch.from_numpy(sc.transform(self.X[:, :]).astype(np.float32))

    def __len__(self):
        """ Denotes the total number of samples.

        Returns:
            int: Total number of samples.

        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data.

        Args:
            index (int): Row index of sample to generate.

        Returns:
            torch.tensor, torch.tensor: Data that is generated (data, target).

        """
        return self.X[index], self.y[index]
