import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def flow_marginals(trainer, model, dataset, mode):
    axes_names = []
    axes = []

    for n, ax in dataset.grid.items():
        axes_names.append(n)
        axes.append(ax)

    d = dataset.dimensionality
    components = np.stack(np.meshgrid(*axes, indexing="xy")).reshape(d, -1)

    resolutions = [len(ax) for ax in axes]
    input = np.stack(components, axis=-1)
    input = torch.from_numpy(input).float()
    prob = model.log_prob(input).exp().view(*resolutions)

    numbered_axes = tuple(range(d))

    for i in range(d):
        axes_to_sum = tuple(numbered_axes[:i] + numbered_axes[i + 1 :])
        fig = plt.figure()
        plt.plot(axes[i], prob.sum(axes_to_sum), label=axes_names[i])

        model.logger.log_image(
            key=f"{mode}_marginal_%d" % i, images=[fig], step=trainer.global_step
        )
        plt.close()
        # plt.savefig(os.path.join(dir, "marginal_%d.png" % d), bbox_inches="tight")
        # plt.close()

