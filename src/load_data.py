import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def load_data(cfg):
    if cfg.dataset == "circles":
        X, y = datasets.make_circles(n_samples=30000, factor=0.5, noise=0.05)
        X = StandardScaler().fit_transform(X)
    elif cfg.dataset == "moons":
        X, y = datasets.make_moons(n_samples=30000, noise=0.05)
        X = StandardScaler().fit_transform(X)
    elif cfg.dataset == "gw":
        X = np.load(cfg.dataroot)["m1"].T
        X = StandardScaler().fit_transform(X)
    else:
        raise ValueError("Unknown dataset.")

    return torch.from_numpy(X).float()
