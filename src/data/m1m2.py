import torch
import numpy as np
import sklearn

from data.utils import get_k_folds, train_test_split

M_RNG = (0.2, 100)


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


def load_data(path):
    data = np.load(path)
    m1 = data["m1"]
    m1 = m1.clip(*M_RNG)

    m2 = data["m2"]
    m2 = m2.clip(*M_RNG)

    return m1, m2


class M1M2Dataset:
    dimensionality = 2
    has_normalization = True

    n_grid = 1024
    grid_m1 = np.linspace(*M_RNG, n_grid)
    grid_m2 = np.linspace(*M_RNG, n_grid)
    grid = {
        "m1": grid_m1,
        "m2": grid_m2,
    }

    m2_normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    m1_normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def __init__(
        self, path, hierarchical=True, train_val_test_split=(0.8, 0.1, 0.1), fold=0
    ):

        self.hierarchical = hierarchical
        m1, m2 = load_data(path)

        self.m1minmax = None
        self.m2minmax = None

        if not hierarchical:
            raise NotImplementedError
        m1, m2 = self.normalize_forward(m1, m2)
        data = torch.from_numpy(np.stack([m1, m2], axis=-1)).float().permute(1, 0, 2)

        train_fraction, val_fraction, test_fraction = train_val_test_split

        (self.test_data,), (self.train_data,) = train_test_split(
            data, test_fraction=test_fraction
        )
        (self.valid_data,), (self.train_data,) = train_test_split(
            self.train_data, test_fraction=val_fraction
        )
        self.train_dataset = torch.utils.data.TensorDataset(self.train_data)
        self.valid_dataset = torch.utils.data.TensorDataset(self.valid_data)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_data)

    def normalize_forward(self, m1, m2):
        # m2 = np.log(m2)
        # m1 = np.log(m1)
        # m2 = softplus_inv(torch.from_numpy(m2)).numpy()
        # m1 = softplus_inv(torch.from_numpy(m1)).numpy()
        # m1 = self.m1_normalizer.fit_transform(m1.reshape(-1, 1)).reshape(m1.shape)
        # m2 = self.m2_norma0lizer.fit_transfmrm(m2.reshape(-1, 1)).reshape(m2.shape)
        if self.m1minmax is None:
            self.m1minmax = (m1.min(), m1.max())
        if self.m2minmax is None:
            self.m2minmax = (m2.min(), m2.max())

        # m1 = (m1 - self.m1minmax[0]) / (self.m1minmax[1] - self.m1minmax[0]) * 2 - 1
        # m2 = (m2 - self.m2minmax[0]) / (self.m2minmax[1] - self.m2minmax[0]) * 2 - 1
        return m1, m2

    def normalize_inverse(self, m1, m2):
        raise NotImplementedError

        m2 = self.m2_normalizer.inverse_transform(m2.reshape(-1, 1)).reshape(m2.shape)
        m1 = self.m1_normalizer.inverse_transform(m1.reshape(-1, 1)).reshape(m1.shape)
        # m2 = np.exp(m2)
        # m1 = np.exp(m1)
        return m1, m2
