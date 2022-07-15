import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def kde_2d(dir, model, dataset, keys=None, num_flow_samples=1024):
    x_dataset = torch.stack([dataset[i][0] for i in range(len(dataset))])
    x_flow = model.flow_dist.sample((num_flow_samples,))

    x_dataset = dataset.normalize_inverse(x_dataset)
    x_flow = dataset.normalize_inverse(x_flow)

    if len(x_dataset.shape) == 3:
        x_dataset = x_dataset.mean(1)

    sns.kdeplot(x=x_dataset.numpy()[:, 0], y=x_dataset.numpy()[:, 1], label="dataset")
    sns.kdeplot(x=x_flow.numpy()[:, 0], y=x_flow.numpy()[:, 1], label="flow")
    plt.legend()
    plt.title("2d_kde")
    plt.tight_layout()
    plt.savefig(
        os.path.join(dir, f"flow_marginal_kdeplots_2d_kde.png"), bbox_inches="tight"
    )
    plt.close()
