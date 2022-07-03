import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def flow_marginals(
    dir, model, dataset, boundaries=(-3, 3), resolution=256, normalize=True, keys=None
):
    grid = torch.linspace(*boundaries, resolution)
    d_grid = [grid] * dataset.dimensionality
    meshgrid = torch.meshgrid(*d_grid, indexing="xy")
    x = torch.stack(meshgrid).reshape(dataset.dimensionality, -1).permute(1, 0)
    shape = [resolution] * dataset.dimensionality
    prob = model.log_prob(x).exp().view(*shape)

    d_grid = torch.stack(d_grid, dim=-1)

    if normalize:
        d_grid = dataset.normalize_inverse(d_grid)

    for d in range(dataset.dimensionality):

        if keys is not None:
            key = keys[d]
        else:
            key = d

        dimensions_to_sum = list(range(dataset.dimensionality))
        dimensions_to_sum.pop(d)

        if len(dimensions_to_sum) == 0:
            marginal = prob
        else:
            marginal = prob.sum(dim=tuple(dimensions_to_sum))

        plt.plot(d_grid[:, d].numpy(), marginal.numpy())
        plt.title(key)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"flow_marginals_{key}.png"), bbox_inches="tight")
        plt.close()
