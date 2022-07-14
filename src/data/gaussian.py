import torch
import numpy as np
import os


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


RNG = (-3, 12)


def process_data():
    m1 = torch.randn(64, 1024) + 3
    m2 = torch.randn(64, 1024) - 3 
    z = torch.randn(64, 1024) + 0
    chi = torch.randn(64, 1024) - 0 
    x = torch.stack([m1, m2, z, chi], dim=-1).permute(1, 0, 2)
    return torch.utils.data.TensorDataset(x)


class SyntheticDataset:
    dimensionality = 4
    has_normalization = True

    # n_grid = 32
    grid_m1 = np.linspace(*RNG, 32)
    grid_q = np.linspace(*RNG, 31)
    grid_z = np.linspace(*RNG, 30)
    grid_chi = np.linspace(*RNG, 29)
    grid = {
        "m1": grid_m1,
        "q": grid_q,
        "z": grid_z,
        "chi": grid_chi,
    }

    def __init__(
        self,
        hierarchical=True,
        train_val_test_split=(0.8, 0.1, 0.1),
        loader_kwargs={},
        train_batch_size=32,
    ):
        self.hierarchical = hierarchical
        self.train_batch_size = train_batch_size

        if not hierarchical:
            raise NotImplementedError

        self.gw_data = process_data()
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
