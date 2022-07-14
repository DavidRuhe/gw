import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np


@torch.no_grad()
def flow_heatmap_2d(trainer, model, dataset, mode):
    axes_names = []
    axes = []

    for n, ax in dataset.grid.items():
        axes_names.append(n)
        axes.append(ax)

    x, y = np.stack(np.meshgrid(*axes, indexing="xy")).reshape(2, -1)

    resolutions = [len(ax) for ax in axes]

    x_log = torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=-1).float()
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
    # plt.savefig(os.path.join(dir, "flow_heatmap_2d.png"), bbox_inches="tight")
    model.logger.log_image(
        key=f"{mode}_flow_heatmap_2d", images=[fig], step=trainer.global_step
    )
    plt.close()

    for d in range(2):
        fig = plt.figure()
        plt.plot(axes[d], prob.sum(d), label=axes_names[d])
        # plt.savefig(os.path.join(dir, "marginal_%d.png" % d), bbox_inches="tight")
        model.logger.log_image(
            key=f"{mode}_marginal_%d" % d, images=[fig], step=trainer.global_step
        )
        plt.close()


@torch.no_grad()
def flow_heatmap_m1m2z(trainer, model, dataset, mode):
    axes_names = []
    axes = []

    for n, ax in dataset.grid.items():
        axes_names.append(n)
        axes.append(ax)

    m1, m2, z = np.stack(np.meshgrid(*axes, indexing="xy")).reshape(3, -1)

    if dataset.has_normalization:
        raise NotImplementedError
        x, y = dataset.normalize_forward(x, y)

    resolutions = [len(ax) for ax in axes]

    input = np.stack([m1, m2, z], axis=-1)
    input = torch.from_numpy(input).float()
    prob = model.log_prob(input).exp().view(*resolutions)

    pm1m2 = prob.sum(-1)

    fig = plt.figure(figsize=(16, 16), facecolor="white")
    plt.imshow(
        pm1m2,
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

    model.logger.log_image(
        key=f"{mode}_flow_heatmap_2d", images=[fig], step=trainer.global_step
    )
    plt.close()
    # plt.savefig(os.path.join(dir, "flow_heatmap_2d.png"), bbox_inches="tight")
    # plt.close()

    numbered_axes = tuple(range(3))

    for d in range(3):
        axes_to_sum = tuple(numbered_axes[:d] + numbered_axes[d + 1 :])
        fig = plt.figure()
        plt.plot(axes[d], prob.sum(axes_to_sum), label=axes_names[d])

        model.logger.log_image(
            key=f"{mode}_marginal_%d" % d, images=[fig], step=trainer.global_step
        )
        plt.close()
        # plt.savefig(os.path.join(dir, "marginal_%d.png" % d), bbox_inches="tight")
        # plt.close()


@torch.no_grad()
def flow_heatmap_m1m2(trainer, model, dataset, mode):
    axes_names = []
    axes = []

    for n, ax in dataset.grid.items():
        axes_names.append(n)
        axes.append(ax)

    d = dataset.dimensionality
    x = np.stack(np.meshgrid(*axes, indexing="xy")).reshape(d, -1)

    resolutions = [len(ax) for ax in axes]

    input = np.stack(x, axis=-1)
    input = torch.from_numpy(input).float()
    prob = model.log_prob(input).exp().view(*resolutions)

    numbered_axes = tuple(range(d))
    axes_to_sum = numbered_axes[2:]
    pm1m2 = prob.sum(dim=axes_to_sum)
    fig = plt.figure(figsize=(16, 16), facecolor="white")
    plt.imshow(
        pm1m2,
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

    model.logger.log_image(
        key=f"{mode}_flow_heatmap_m1m2", images=[fig], step=trainer.global_step
    )
    plt.close()