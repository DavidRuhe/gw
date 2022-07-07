import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np


@torch.no_grad()
def flow_heatmap_2d(dir, model, dataset, boundaries=(-3, 3)):
    # grid = torch.linspace(*boundaries, resolution)
    # meshgrid = torch.meshgrid(grid, grid, indexing="xy")
    # x = torch.stack(meshgrid, dim=-1).reshape(-1, 2)
    axes_names = []
    axes = []
    if dataset.has_normalization:
        for n, ax in dataset.grid.items():
            axes_names.append(n)
            axes.append(ax)

        m1, q = np.stack(np.meshgrid(*axes, indexing="xy")).reshape(2, -1)
        m1, q = dataset.normalize_forward(m1, q)

    else:
        raise NotImplementedError

    resolutions = [len(ax) for ax in axes]

    x_log = torch.stack([torch.from_numpy(m1), torch.from_numpy(q)], dim=-1).float()
    prob = model.log_prob(x_log).exp().view(*resolutions)

    fig = plt.figure(figsize=(16, 16), facecolor="white")
    plt.imshow(
        prob,
        cmap="jet",
        origin="lower",
        extent=(
            axes[0][0],
            axes[0][-1],
            axes[1][0],
            axes[1][-1],
        ),  # origin='lower' changes the order
        aspect="auto",
    )

    plt.xlabel(axes_names[0])  # origin='lower' changes the order
    plt.ylabel(axes_names[1])
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "flow_heatmap_2d.png"), bbox_inches="tight")
    plt.close()

    for d in range(2):
        plt.plot(axes[d], prob.sum(d), label=axes_names[d])
        plt.savefig(
            os.path.join(dir, "marginal_%d.png" % d), bbox_inches="tight"
        )
        plt.close()

