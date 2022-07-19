import torch
import numpy as np
import os
import random
from itertools import cycle
import math


M_RNG = (0.2, 100)
Z_RNG = (0.1, 3)
CHI_RNG = (-1, 1)


class DataLoader:
    batches = None

    def __init__(self, data, batch_size, shuffle=True):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_length = len(self.data)

    def __iter__(self):

        indices = list(reversed(range(0, self.data_length)))
        if self.shuffle:
            random.shuffle(indices)

        self.batches = [
            indices[i : i + self.batch_size]
            for i in range(0, self.data_length, self.batch_size)
        ]
        return self

    def __next__(self):
        if len(self.batches) > 0:
            return self.data[self.batches.pop()]
        raise StopIteration

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


class TensorDataset:
    def __init__(self, tensor):
        self.tensor = tensor
        self.len = len(tensor)

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.len


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.len = max(len(d) for d in self.datasets)
        super().__init__()

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self))))
        index = torch.tensor(index)
        return torch.stack([d[index % len(d)] for d in self.datasets], dim=1)

    def __len__(self):
        return self.len

    def __len__(self):
        return self.len


class ConcatDataLoader:
    iterable = None

    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

        i, max_size = None, 0
        for j, loader in enumerate(self.dataloaders):
            length = len(loader)
            if len(loader) > max_size:
                i, max_size = j, length
        self.i = i
        self.max_size = max_size

    def __iter__(self):

        loaders = [
            loader if j == self.i else cycle(loader)
            for j, loader in enumerate(self.dataloaders)
        ]
        self.iterable = iter(zip(*loaders))
        return self

    def __next__(self):
        return next(self.iterable)

    def __len__(self):
        return self.max_size


def process_gw_data(path):
    events = np.load(path, allow_pickle=True)
    datasets = []
    m1min, m1max = float("inf"), -float("inf")
    m2min, m2max = float("inf"), -float("inf")
    zmin, zmax = float("inf"), -float("inf")

    chimin, chimax = float("inf"), -float("inf")
    for n, event in events.items():
        m1 = torch.from_numpy(event["m1"])
        m1 = m1.clamp(*M_RNG)
        m1min = min(m1min, m1.min())
        m1max = max(m1max, m1.max())

        m2 = torch.from_numpy(event["m2"])
        m2 = m2.clamp(*M_RNG)
        m2min = min(m2min, m2.min())
        m2max = max(m2max, m2.max())

        z = torch.from_numpy(event["z"])
        z = z.clamp(*Z_RNG)
        zmin = min(zmin, z.min())
        zmax = max(zmax, z.max())

        chi = torch.from_numpy(event["Xeff"])
        chi = chi.clamp(*CHI_RNG)
        chimin = min(chimin, chi.min())
        chimax = max(chimax, chi.max())

        z_prior = torch.from_numpy(event["z_prior"])
        chi_prior = torch.from_numpy(event["Xeff_priors"])

        gw_data = torch.stack([m1, m2, z, chi, z_prior, chi_prior], dim=-1).float()

        datasets.append(TensorDataset(gw_data))

    dataset = ConcatDataset(*datasets)

    return dataset


class M1M2ZChiDataset:
    dimensionality = 4
    has_normalization = True

    n_grid = 32
    grid_m1 = np.linspace(*M_RNG, n_grid)
    grid_m2 = np.linspace(*M_RNG, n_grid)
    grid_z = np.linspace(*Z_RNG, n_grid)
    grid_chi = np.linspace(*CHI_RNG, n_grid)
    grid = {
        "m1": grid_m1,
        "m2": grid_m2,
        "z": grid_z,
        "chi": grid_chi,
    }

    def __init__(
        self,
        gw_path,
        hierarchical=True,
        train_val_test_split=(0.8, 0.1, 0.1),
        loader_kwargs={},
        train_batch_size=32,
        val_batch_size=32,
    ):
        gw_path = os.path.join(os.environ["DATAROOT"], gw_path)

        self.hierarchical = hierarchical
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        if not hierarchical:
            raise NotImplementedError

        self.gw_data = process_gw_data(gw_path)
        self.loader_kwargs = loader_kwargs

    def train_dataloader(self):
        gw_data = self.gw_data
        gw_loader = DataLoader(
            gw_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            **self.loader_kwargs
        )
        return gw_loader

    def val_dataloader(self):
        gw_data = self.gw_data
        gw_loader = DataLoader(
            gw_data, batch_size=self.val_batch_size, shuffle=False, **self.loader_kwargs
        )
        return gw_loader

    def test_dataloader(self):
        return None


def process_selection_data(path):
    selection_data = np.load(path, allow_pickle=True)
    m1 = torch.from_numpy(selection_data["m1"])
    m1 = m1.clamp(*M_RNG)

    m2 = torch.from_numpy(selection_data["m2"])
    m2 = m2.clamp(*M_RNG)

    z = torch.from_numpy(selection_data["z"])
    z = z.clamp(*Z_RNG)

    chi = torch.from_numpy(selection_data["Xeff"])
    chi = chi.clamp(*CHI_RNG)

    p_draw_m1m2z = torch.from_numpy(selection_data["p_draw_m1m2z"])
    p_draw_chi = torch.from_numpy(selection_data["p_draw_chiEff"])

    ntrials = selection_data["nTrials"]
    ntrials = torch.tensor(ntrials).repeat(len(m1))

    naccepted = len(m1)
    naccepted = torch.tensor(naccepted).repeat(len(m1))

    selection_data = torch.stack(
        [m1, m2, z, chi, p_draw_m1m2z, p_draw_chi, ntrials, naccepted], dim=-1
    ).float()

    return TensorDataset(selection_data)


class M1M2ZChiSBDataset(M1M2ZChiDataset):
    def __init__(self, sb_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sb_path = os.path.join(os.environ["DATAROOT"], sb_path)
        self.sb_data = process_selection_data(self.sb_path)

    def train_dataloader(self):
        gw_data = self.gw_data
        gw_loader = DataLoader(
            gw_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            **self.loader_kwargs
        )

        sb_data = self.sb_data
        sb_loader = DataLoader(
            sb_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            **self.loader_kwargs
        )
        return ConcatDataLoader(gw_loader, sb_loader)

    def val_dataloader(self):
        gw_data = self.gw_data
        gw_loader = DataLoader(
            gw_data, batch_size=self.val_batch_size, shuffle=False, **self.loader_kwargs
        )

        sb_data = self.sb_data
        sb_loader = DataLoader(
            sb_data, batch_size=self.val_batch_size, shuffle=False, **self.loader_kwargs
        )
        return ConcatDataLoader(gw_loader, sb_loader)
