import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def flow_marginals(trainer, model, dataset, mode):
    if trainer.logger is None:
        return
    axes_names = []
    axes = []

    for n, ax in dataset.grid.items():
        axes_names.append(n)
        axes.append(ax)

    d = dataset.dimensionality
    components = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(d, -1).T

    resolutions = [len(ax) for ax in axes]
    input = torch.from_numpy(components).float()
    prob = model.log_prob(input).exp().view(*resolutions)

    numbered_axes = tuple(range(d))

    for i in range(d):
        axes_to_sum = tuple(numbered_axes[:i] + numbered_axes[i + 1 :])
        fig = plt.figure()
        plt.plot(axes[i], prob.sum(axes_to_sum), label=axes_names[i])

        trainer.logger.log_image(
            {f"{mode}_marginal_%d" % i: fig}, step=trainer.global_step
        )
        plt.close()
        # plt.savefig(os.path.join(dir, "marginal_%d.png" % d), bbox_inches="tight")
        # plt.close()

    # Hacky to put it here, but let's keep it for now
    axes_to_sum = numbered_axes[2:]
    pm1m2 = prob.sum(dim=axes_to_sum)
    fig = plt.figure(figsize=(16, 16), facecolor="white")
    plt.imshow(
        pm1m2,
        cmap="jet",
        origin="lower",
        extent=(
            axes[1][0],
            axes[1][-1],
            axes[0][0],
            axes[0][-1],
        ),
        aspect="auto",
    )

    plt.xlabel(axes_names[1])
    plt.ylabel(axes_names[0])
    plt.tight_layout()

    trainer.logger.log_image(
        {f"{mode}_flow_heatmap_m1m2": fig}, step=trainer.global_step
    )
    plt.close()
