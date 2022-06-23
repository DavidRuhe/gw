import os
import torch
import numpy as np

from data.utils import get_k_folds, train_test_split


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


def load_data(path):
    data = np.load(path)
    X = data["m1"]
    X = torch.from_numpy(X)

    # X = softplus_inv(X)
    X = torch.log(X)

    X = (X - X.mean()) / X.std()

    return X.float()


class M1Dataset(torch.utils.data.TensorDataset):
    dimensionality = 1
    hierarchical=True

    def __init__(self, path, split, fold=0, test_size=0.1, limit_samples=0):

        conditional_path = os.path.join(path)
        conditional_data = load_data(conditional_path)
        if limit_samples > 0:
            conditional_data = conditional_data[:limit_samples]
        self.conditional_data = conditional_data

        self.test_data, self.train_data = train_test_split(
            self.conditional_data, test_fraction=test_size
        )
        folds = get_k_folds(self.train_data[0], 5)
        fold_indices = folds[fold]
        valid_indices, train_indices = fold_indices

        (train_conditional_data,) = self.train_data
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
