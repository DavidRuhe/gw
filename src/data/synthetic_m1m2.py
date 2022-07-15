import os
import torch
import numpy as np
import math


from data.utils import get_k_folds, train_test_split
from simulators.synthetic_m1m2 import Mixture, PowerLaw, rand_between


def softplus_inv(y):
    return y + y.neg().expm1().neg().log()


lpeak = 0.1
mmax = 86.22
mmin = 4.59
qmin=0.1
qmax=1
alpha = 2.63
sigmam = 5.69
mum = 33.07


def load_data(path, num_events=1024, num_posterior_samples=32768, posterior_std=1e-4):
    prior = Mixture(
        (
            torch.distributions.Normal(mum, sigmam),
            PowerLaw(-alpha, mmin),
        ),
        (lpeak, 1 - lpeak),
    )
    m1 = prior.sample(num_events).squeeze()
    m1 = m1[:, None] + posterior_std * torch.randn(len(m1), num_posterior_samples)

    m1 = m1[torch.all(m1 > mmin, dim=1)]
    m1 = m1[torch.all(m1 < mmax, dim=1)]

    m2 = m1.clone()
    m2 = m1 - rand_between(m2.shape, 0, 4)
    # m2 = np.clip(m2, mmin, None)
    return m1.float(), m2.float()


class SynetheticM1M2Dataset(torch.utils.data.TensorDataset):
    dimensionality = 2
    has_normalization = True
    n_grid = 1024
    grid = (
        ("m1", torch.linspace(1, 128, n_grid)),
        ("m2", torch.linspace(1, 128, n_grid)),
    )

    def __init__(
        self, path, split, fold=0, test_size=0.1, limit_samples=0, hierarchical=False
    ):

        self.hierarchical = hierarchical
        M1, M2 = load_data(path)

        if not hierarchical:
            ix = 0
            M1 = M1[:, ix]
            M2 = M2[:, ix]

        data = torch.stack([M1, M2], dim=-1)
        self.loc, self.scale = None, None
        data = self.normalize_forward(data.view(-1, self.dimensionality)).view(
            data.shape
        )

        print(f"Normalized location and scale: {data.mean()}, {data.std()}")

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
        # x_log = softplus_inv(x)
        # x_log = x.log()
        x_log = x

        if self.loc is None and self.scale is None:

            self.loc, self.scale = x_log.mean(dim=0, keepdim=True), x_log.std(
                dim=0, keepdim=True
            )
            return self.normalize_forward(x)
        else:
            return (x_log - self.loc) / self.scale

    def normalize_inverse(self, y):
        y = y * self.scale + self.loc
        # y = torch.nn.functional.softplus(y)
        # y = torch.softplus(y)
        return y
