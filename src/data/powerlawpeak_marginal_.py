import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from data.utils import get_k_folds, train_test_split


def load_data(path):
    X = np.load(path)[:, None]
    X = StandardScaler().fit_transform(X)
    return torch.from_numpy(X).float()


class PowerPlusPeakMarginalDataset(torch.utils.data.TensorDataset):
    def __init__(self, path, split, fold=0, test_size=0.1):

        self.data = load_data(path)
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