import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from data.utils import train_test_split, get_k_folds


def load_data(name):
    if name == "circles":
        X, y = datasets.make_circles(n_samples=1422, noise=0.05)
        # X = StandardScaler().fit_transform(X)
    elif name == "moons":
        X, y = datasets.make_moons(n_samples=1422, noise=0.05)
        # X = StandardScaler().fit_transform(X)
    else:
        raise ValueError(f"Unknown dataset {name}.")

    return torch.from_numpy(X).float()


class ToyDataset(torch.utils.data.TensorDataset):
    dimensionality = 2
    hierarchical = False

    def __init__(self, name, split, fold=0, test_size=0.1):

        self.data = load_data(name)
        self.test_data, self.train_data = train_test_split(
            self.data, test_fraction=test_size
        )
        self.test_data = self.test_data[0]
        self.train_data = self.train_data[0]
        folds = get_k_folds(self.train_data, 5)
        fold_indices = folds[fold]
        valid_indices, train_indices = fold_indices
        if split == "train":
            super().__init__(self.train_data[train_indices])
        elif split == "valid":
            super().__init__(self.train_data[valid_indices])
        elif split == "test":
            super().__init__(self.test_data)


# class ToyDataset(torch.utils.data.Dataset):
#     def __init__(self, n_samples=1024):
#         super().__init__()
#         self.X = datasets.make_moons(n_samples=n_samples, noise=0.05)[0].astype(
#             np.float32
#         )

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx]
