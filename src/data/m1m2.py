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
    return torch.from_numpy(M1).float(), torch.from_numpy(M2).float()


class M1M2Dataset(torch.utils.data.TensorDataset):
    dimensionality = 2

    def __init__(
        self, path, split, fold=0, test_size=0.1, limit_samples=0, hierarchical=True
    ):

        self.hierarchical = hierarchical
        M1, M2 = load_data(path)

        if not hierarchical:
            # M1 = M1.mean(dim=-1)
            # M2 = M2.mean(dim=-1)
            M1 = M1[:, 0]
            M2 = M2[:, 0]

        data = torch.stack([M1, M2], dim=-1)
        self.loc, self.scale = None, None
        data = self.normalize_forward(data)

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

    def normalize_forward(self, x):
        # x_log = x.log()
        x_log = softplus_inv(x)
        if self.loc is None and self.scale is None:
            self.loc, self.scale = x_log.mean(dim=0, keepdim=True), x_log.std(dim=0, keepdim=True)
            return self.normalize_forward(x)
        else:
            return (x_log - self.loc) / self.scale

    def normalize_inverse(self, y):
        # y = torch.nn.functional.softplus(y)
        # return torch.exp(y * self.scale + self.loc)
        return torch.nn.functional.softplus(y * self.scale + self.loc)
