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
    return X.float()


class M1Dataset(torch.utils.data.TensorDataset):
    dimensionality = 1

    def __init__(
        self, path, split, fold=0, test_size=0.1, limit_samples=0, hierarchical=True
    ):

        data = load_data(path)[..., None]
        # data = load_data(path)
        if not hierarchical:
            data = data[:, :1]

        self.loc, self.scale = None, None
        data = self.normalize_forward(data)

        (self.test_data,), (self.train_data,) = train_test_split(
            data, test_fraction=test_size
        )
        folds = get_k_folds(self.train_data, 5)
        fold_indices = folds[fold]
        valid_indices, train_indices = fold_indices


        if split == "train":
            super().__init__(
                self.train_data[train_indices],
            )
        elif split == "valid":
            super().__init__(
                self.train_data[valid_indices],
            )
        elif split == "test":
            super().__init__(self.test_data)

    def normalize_forward(self, x):
        x_log = x
        if self.loc is None and self.scale is None:
            self.loc, self.scale = x_log.mean(dim=(0, 1), keepdim=False), x_log.std(
                dim=(0, 1), keepdim=False
            )
            return self.normalize_forward(x)
        else:
            return (x_log - self.loc) / self.scale

    def normalize_inverse(self, y):
        y = y * self.scale + self.loc
        # y = torch.nn.functional.softplus(y)
        # y = torch.exp(y)
        return y
