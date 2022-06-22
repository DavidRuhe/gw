import os
import numpy as np
import torch

from data.utils import get_k_folds, train_test_split


def load_data(path):
    X = np.load(path)
    return torch.from_numpy(X).float()


class SyntheticDataset(torch.utils.data.TensorDataset):
    def __init__(self, path, split, fold=0, test_size=0.1):

        conditional_path = os.path.join(path, "posterior.npy")
        conditional_data = load_data(conditional_path)
        self.conditional_data = conditional_data

        self.test_data, self.train_data = train_test_split(
            self.conditional_data, test_fraction=test_size
        )
        folds = get_k_folds(self.train_data[0], 5)
        fold_indices = folds[fold]
        valid_indices, train_indices = fold_indices

        train_conditional_data, = self.train_data
        if split == "train":
            super().__init__(
                train_conditional_data[train_indices],
            )
        elif split == "valid":
            super().__init__(
                train_conditional_data[valid_indices],
            )
        elif split == "test":
            super().__init__(*self.test_data)
