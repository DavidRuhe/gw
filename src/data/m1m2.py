import os
import torch
import numpy as np

from data.utils import get_k_folds, train_test_split


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


def load_data(path):
    data = np.load(path)
    M1 = data["m1"]
    M2 = data["m2"]
    return torch.from_numpy(M1), torch.from_numpy(M2)


class M1M2Dataset(torch.utils.data.TensorDataset):
    dimensionality = 2

    def __init__(
        self, path, split, fold=0, test_size=0.1, limit_samples=0, hierarchical=True
    ):

        self.hierarchical = hierarchical
        M1, M2 = load_data(path)

        if not hierarchical:
            M1 = M1.mean(dim=-1)
            M2 = M2.mean(dim=-1)

        self.normalization_parameters = {}
        M1 = self.normalize_forward(M1, "M1").float()
        M2 = self.normalize_forward(M2, "M2").float()

        data = torch.stack([M1, M2], dim=-1)

        if limit_samples > 0:
            data = data[:limit_samples]

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

    def normalize_forward(self, x, key):
        x_log = x.log()
        if key in self.normalization_parameters:
            loc, scale = self.normalization_parameters[key]
            return (x_log - loc) / scale
        else:
            self.normalization_parameters[key] = (x_log.mean(), x_log.std())
            return self.normalize_forward(x, key)

    def normalize_inverse(self, y, key):
        assert (
            key in self.normalization_parameters
        ), f"Normalization for {key} unknown. First run forward normalization."
        loc, scale = self.normalization_parameters[key]
        return torch.exp(y * scale + loc)
