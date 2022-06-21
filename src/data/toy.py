import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import unittest


DATA_SEED = 0


def train_test_split(data, test_fraction):
    """
    Split data into train and test sets.
    """
    n = len(data)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(DATA_SEED))
    test_size = int(n * test_fraction)
    return data[indices[:test_size]], data[indices[test_size:]]


def get_k_folds(data, k):
    """
    Split data into k folds.
    """
    n = len(data)
    valid_size = int(n / k)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(DATA_SEED))
    folds = []
    for i in range(k):
        folds.append(
            (
                indices[i * valid_size : (i + 1) * valid_size],
                torch.cat([indices[: i * valid_size], indices[(i + 1) * valid_size :]]),
            )
        )
    return folds


def load_data(name):
    if name == "circles":
        X, y = datasets.make_circles(n_samples=30000, factor=0.5, noise=0.05)
        X = StandardScaler().fit_transform(X)
    elif name == "moons":
        X, y = datasets.make_moons(n_samples=30000, noise=0.05)
        X = StandardScaler().fit_transform(X)
    else:
        raise ValueError(f"Unknown dataset {name}.")

    return torch.from_numpy(X).float()


class ToyDataset(torch.utils.data.TensorDataset):
    def __init__(self, name, split, fold=0, test_size=0.1):

        self.data = load_data(name)
        self.test_data, self.train_data = train_test_split(self.data, test_size)
        folds = get_k_folds(self.train_data, 5)
        fold_indices = folds[fold]
        valid_indices, train_indices = fold_indices
        if split == "train":
            super().__init__(self.train_data[train_indices])
        elif split == "valid":
            super().__init__(self.train_data[valid_indices])
        elif split == "test":
            super().__init__(self.test_data)

        print(self.train_data[train_indices].sum(), self.train_data[valid_indices].sum(), self.test_data.sum())


class TestToyDataset(unittest.TestCase):
    def test_toy_dataset(self):
        dataset = ToyDataset("circles", 0)
        self.assertTrue(dataset[0][0].shape == (2,))
        dataset = ToyDataset("moons", 0)
        self.assertTrue(dataset[0][0].shape == (2,))
