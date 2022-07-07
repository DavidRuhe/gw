import torch
import numpy as np
import sklearn

from data.utils import get_k_folds, train_test_split


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


def sigmoid_inv(y):
    return torch.logit(y.clamp(1e-2, 1 - 1e-2))


M_RNG = (0.2, 100)
Q_RNG = (0.1, 0.99)


def load_data(path):
    data = np.load(path)
    m1 = data["m1"]
    m1 = m1.clip(*M_RNG)

    m2 = data["m2"]
    m2 = m2.clip(*M_RNG)

    q = m2 / m1
    q = q.clip(*Q_RNG)
    return m1, q


class M1QDataset:
    dimensionality = 2
    has_normalization = True

    n_grid = 1024
    grid_m1 = np.linspace(*M_RNG, n_grid)
    grid_q = np.linspace(*Q_RNG, n_grid)
    grid = {
        "m1": grid_m1,
        "q": grid_q,
    }

    q_normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    m1_normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def __init__(
        self, path, hierarchical=True, train_val_test_split=(0.8, 0.1, 0.1), fold=0
    ):

        self.hierarchical = hierarchical
        m1, q = load_data(path)
        if not hierarchical:
            raise NotImplementedError
        m1, q = self.normalize_forward(m1, q)
        data = torch.from_numpy(np.stack([m1, q], axis=-1)).float().permute(1, 0, 2)

        train_fraction, val_fraction, test_fraction = train_val_test_split

        (self.test_data,), (self.train_data,) = train_test_split(
            data, test_fraction=test_fraction
        )
        (self.valid_data,), (self.train_data,) = train_test_split(
            self.train_data, test_fraction=val_fraction
        )

        # folds = get_k_folds(self.train_data, 5)
        # fold_indices = folds[fold]
        # valid_indices, train_indices = fold_indices
        # self.train_data = self.train_data[train_indices]
        # self.valid_data = self.train_data[valid_indices]

        self.train_dataset = torch.utils.data.TensorDataset(self.train_data)
        self.valid_dataset = torch.utils.data.TensorDataset(self.valid_data)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_data)

    def normalize_forward(self, m1, q):
        q = 1 / q - 1
        m1 = np.log(m1)
        q = np.log(q)
        m1 = self.m1_normalizer.fit_transform(m1.reshape(-1, 1)).reshape(m1.shape)
        q = self.q_normalizer.fit_transform(q.reshape(-1, 1)).reshape(q.shape)
        return m1, q

    def normalize_inverse(self, m1, q):
        q = self.q_normalizer.inverse_transform(q.reshape(-1, 1)).reshape(q.shape)
        m1 = self.m1_normalizer.inverse_transform(m1.reshape(-1, 1)).reshape(m1.shape)
        q = np.exp(q)
        m1 = np.exp(m1)
        q = 1 / (q + 1)

        breakpoint()

        return m1, q
