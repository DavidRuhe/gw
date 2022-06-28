import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from data.utils import get_k_folds, train_test_split


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


def load_data(path):
    X = np.load(path)
    return torch.from_numpy(X).float()


class PowerPlusPeakDataset(torch.utils.data.TensorDataset):
    dimensionality = 1
    def __init__(self, path, split, fold=0, test_size=0.1, hierarchical=False):

        if hierarchical:
            path = os.path.join(path, "posterior.npy")
        else:
            path = os.path.join(path, "marginal.npy")

        data = load_data(path)[:, None]

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
        # x_log = x.log()
        x_log = softplus_inv(x)
        if self.loc is None and self.scale is None:
            self.loc, self.scale = x_log.mean(dim=0, keepdim=True), x_log.std(
                dim=0, keepdim=True
            )
            return self.normalize_forward(x)
        else:
            return (x_log - self.loc) / self.scale

    def normalize_inverse(self, y):
        # y = torch.nn.functional.softplus(y)
        # return torch.exp(y * self.scale + self.loc)
        return torch.nn.functional.softplus(y * self.scale + self.loc)
