import torch
import numpy as np

from data.utils import get_k_folds, train_test_split

M_RNG = (0.2, 100)
Z_RNG = (1e-2, 6)


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


def process_gw_data(path):
    events = np.load(path, allow_pickle=True)
    datasets = []
    m1min, m1max = float("inf"), -float("inf")
    m2min, m2max = float("inf"), -float("inf")
    zmin, zmax = float("inf"), -float("inf")
    for n, event in events.items():
        m1 = torch.from_numpy(event["m1"]).float()
        m1 = m1.clamp(*M_RNG)
        m1min = min(m1min, m1.min())
        m1max = max(m1max, m1.max())

        m2 = torch.from_numpy(event["m2"]).float()
        m2 = m2.clamp(*M_RNG)
        m2min = min(m2min, m2.min())
        m2max = max(m2max, m2.max())

        z = torch.from_numpy(event["z_prior"]).float()
        z = z.clamp(*Z_RNG)
        zmin = min(zmin, z.min())
        zmax = max(zmax, z.max())

        z_prior = torch.from_numpy(event["z_prior"]).float()

        x = torch.stack([m1, m2, z, z_prior], dim=-1)

        datasets.append(torch.utils.data.TensorDataset(x))

    return ConcatDataset(*datasets), (m1min, m1max), (m2min, m2max), (zmin, zmax)


def process_selection_data(path):
    selection_data = np.load(path, allow_pickle=True)
    m1 = torch.from_numpy(selection_data["m1"]).float()
    m1 = m1.clamp(*M_RNG)

    m2 = torch.from_numpy(selection_data["m2"]).float()
    m2 = m2.clamp(*M_RNG)

    z = torch.from_numpy(selection_data["z"]).float()
    z = z.clamp(*Z_RNG)

    p_draw_m1m2z = torch.from_numpy(selection_data["p_draw_m1m2z"]).float()

    ntrials = selection_data["nTrials"]
    ntrials = torch.tensor(ntrials).repeat(len(m1))

    m1m2zpt = torch.stack([m1, m2, z, p_draw_m1m2z], dim=-1)

    return torch.utils.data.TensorDataset(m1m2zpt)


class M1M2SBDataset:
    dimensionality = 2
    has_normalization = False

    n_grid = 64
    grid_m1 = np.linspace(*M_RNG, n_grid)
    grid_m2 = np.linspace(*M_RNG, n_grid)
    grid_z = np.linspace(*Z_RNG, n_grid)
    grid = {
        "m1": grid_m1,
        "m2": grid_m2,
        "z": grid_z,
    }

    def __init__(
        self,
        injection_path,
        sample_path,
        gw_train_batch_size,
        sel_train_batch_size,
        hierarchical=True,
        train_val_test_split=(0.8, 0.1, 0.1),
        fold=0,
        loader_kwargs={},
    ):

        self.hierarchical = hierarchical
        self.m1minmax = None
        self.m2minmax = None

        if not hierarchical:
            raise NotImplementedError

        # train_fraction, val_fraction, test_fraction = train_val_test_split
        self.gw_data, self.m1minmax, self.m2minmax, self.zminmax = process_gw_data(
            sample_path
        )
        self.selection_data = process_selection_data(injection_path)
        self.gw_train_batch_size = gw_train_batch_size
        self.sel_train_batch_size = sel_train_batch_size
        self.loader_kwargs = loader_kwargs

        # train_length = int(train_fraction * len(dataset))
        # val_length = int(val_fraction * len(dataset))
        # test_length = len(dataset) - train_length - val_length

        # (
        #     self.train_dataset,
        #     self.val_dataset,
        #     self.test_dataset,
        # ) = torch.utils.data.random_split(
        #     dataset, (train_length, val_length, test_length)
        # )

        # self.loader_kwargs = loader_kwargs

    def normalize_forward(self, m1, m2):
        raise NotImplementedError
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

    def train_dataloader(self):
        gw_data = self.gw_data
        sel_data = self.selection_data

        gw_loader = torch.utils.data.DataLoader(
            gw_data,
            batch_size=self.gw_train_batch_size,
            shuffle=True,
            **self.loader_kwargs
        )
        sel_loader = torch.utils.data.DataLoader(
            sel_data,
            batch_size=self.sel_train_batch_size,
            shuffle=True,
            **self.loader_kwargs
        )
        return [gw_loader, sel_loader]

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
