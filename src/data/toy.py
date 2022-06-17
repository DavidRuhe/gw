import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import unittest

DATA_SEED = 0


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


def load_data(dataset_name):
    if dataset_name == "circles":
        X, y = datasets.make_circles(n_samples=30000, factor=0.5, noise=0.05)
        X = StandardScaler().fit_transform(X)
    elif dataset_name == "moons":
        X, y = datasets.make_moons(n_samples=30000, noise=0.05)
        X = StandardScaler().fit_transform(X)
    else:
        raise ValueError("Unknown dataset.")

    return torch.from_numpy(X).float()


class ToyDataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        dataset_name,
        fold,
        train=True,
    ):
        self.data = load_data(dataset_name)
        folds = get_k_folds(self.data, 5)
        fold_indices = folds[fold]
        test_indices, train_indices = fold_indices
        if train:
            super().__init__(self.data[train_indices])
        else:
            super().__init__(self.data[test_indices])


class TestToyDataset(unittest.TestCase):
    def test_toy_dataset(self):
        dataset = ToyDataset("circles", 0)
        self.assertTrue(dataset[0][0].shape == (2,))
        dataset = ToyDataset("moons", 0)
        self.assertTrue(dataset[0][0].shape == (2,))
