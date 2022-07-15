import torch
import numpy as np

from data.utils import get_k_folds, train_test_split

M_RNG = (0.2, 100)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.len = max(len(d) for d in self.datasets)
        super().__init__()

    def __getitem__(self, index):
        assert index < self.len
        return (torch.cat([torch.stack(d[index % len(d)]) for d in self.datasets]),)

    def __len__(self):
        return self.len


def process_data(path, num_events=30000):
    events = np.load("../datasets/sampleDict_FAR_1_in_1_yr.pickle", allow_pickle=True)
    data = []
    for i, (n, event) in enumerate(events.items()):
        m1 = event["m1"]
        m2 = event["m2"]
        m1 = np.clip(m1, *M_RNG)
        m2 = np.clip(m2, *M_RNG)
        z_prior = event["z_prior"]
        x = np.stack([m1, m2, z_prior], axis=0)
        state = np.random.RandomState(i)  # Use same seed every time.
        if len(x) > num_events:
            data.append(state.choice(x, num_events, replace=False))
        else: 
            data.append(x[:, state.randint(0, x.shape[1], num_events)])  # Bootstrap


    data = np.stack(data, axis=1)
    return data


class M1M2SBDataset:
    dimensionality = 2
    has_normalization = False

    n_grid = 1024
    grid_m1 = np.linspace(*M_RNG, n_grid)
    grid_m2 = np.linspace(*M_RNG, n_grid)
    grid = {
        "m1": grid_m1,
        "m2": grid_m2,
    }

    def __init__(
        self,
        injection_path,
        sample_path,
        hierarchical=True,
        train_val_test_split=(0.8, 0.1, 0.1),
        fold=0,
    ):

        self.hierarchical = hierarchical
        self.m1minmax = None
        self.m2minmax = None

        if not hierarchical:
            raise NotImplementedError

        m1, m2, z, z_prior = process_data(sample_path)

        train_fraction, val_fraction, test_fraction = train_val_test_split

        self.m1minmax = None
        self.m2minmax = None

        if not hierarchical:
            raise NotImplementedError
        m1, m2, z = self.normalize_forward(m1, m2)
        data = torch.from_numpy(np.stack([m1, m2], axis=-1)).float().permute(1, 0, 2)

        train_fraction, val_fraction, test_fraction = train_val_test_split

        (self.test_data,), (self.train_data,) = train_test_split(
            data, test_fraction=test_fraction
        )
        (self.val_dataset,), (self.train_data,) = train_test_split(
            self.train_data, test_fraction=val_fraction
        )
        self.train_dataset = torch.utils.data.TensorDataset(self.train_data)
        self.val_dataset = torch.utils.data.TensorDataset(self.val_dataset)
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
