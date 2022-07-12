import torch
import numpy as np
import os

# import sklearn

# from data.utils import get_k_folds, train_test_split

M_RNG = (0.2, 90)
Z_RNG = (0.1, 3)
CHI_RNG = (-1, 1)


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

    chimin, chimax = float("inf"), -float("inf")
    for n, event in events.items():
        m1 = torch.from_numpy(event["m1"]).float()
        m1 = m1.clamp(*M_RNG)
        m1min = min(m1min, m1.min())
        m1max = max(m1max, m1.max())

        m2 = torch.from_numpy(event["m2"]).float()
        m2 = m2.clamp(*M_RNG)
        m2min = min(m2min, m2.min())
        m2max = max(m2max, m2.max())

        z = torch.from_numpy(event["z"]).float()
        z = z.clamp(*Z_RNG)
        zmin = min(zmin, z.min())
        zmax = max(zmax, z.max())

        chi = torch.from_numpy(event["Xeff"]).float()
        chi = chi.clamp(*CHI_RNG)
        chimin = min(chimin, chi.min())
        chimax = max(chimax, chi.max())

        x = torch.stack([m1, m2, z, chi], dim=-1)

        datasets.append(torch.utils.data.TensorDataset(x))

    return ConcatDataset(*datasets), (m1min, m1max), (m2min, m2max), (zmin, zmax)


class M1M2ZChiDataset:
    dimensionality = 4
    has_normalization = True

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
        path,
        hierarchical=True,
        train_val_test_split=(0.8, 0.1, 0.1),
        loader_kwargs={},
        train_batch_size=32,
    ):
        path = os.path.join(os.environ['DATAROOT'], path)

        self.hierarchical = hierarchical
        self.train_batch_size = train_batch_size

        if not hierarchical:
            raise NotImplementedError

        self.gw_data, self.m1minmax, self.m2minmax, self.zminmax = process_gw_data(path)
        self.loader_kwargs = loader_kwargs

    def train_dataloader(self):
        gw_data = self.gw_data
        gw_loader = torch.utils.data.DataLoader(
            gw_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            **self.loader_kwargs
        )
        return gw_loader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
