import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def kde_2d(dir, model, dataset, keys=None, num_flow_samples=1024):
    x_dataset = torch.stack([dataset[i][0] for i in range(len(dataset))])

    if len(x_dataset.shape) == 3:
        x_dataset = x_dataset.mean(1)

    x_dataset = dataset.normalize_inverse(x_dataset)

    sns.kdeplot(x=x_dataset.numpy()[:, 0], y=x_dataset.numpy()[:, 1], label="dataset")
    plt.legend()
    plt.title("2d_kde")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"kdeplots_2d.png"), bbox_inches="tight")
    plt.close()

    for d in range(x_dataset.shape[1]):
        sns.kdeplot(x=x_dataset.numpy()[:, d])
        plt.title(f"2d_kde_{d}")
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"kdeplots_2d_{d}.png"), bbox_inches="tight")
        plt.close()
