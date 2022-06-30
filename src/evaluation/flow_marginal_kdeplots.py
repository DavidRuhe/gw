import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def marginal_kdeplots(dir, model, dataset, keys=None, num_flow_samples=1024):
    x_dataset = torch.stack([dataset[i][0] for i in range(len(dataset))])
    x_flow = model.flow_dist.sample((num_flow_samples,))

    # x_dataset = dataset.normalize_inverse(x_dataset)
    x_flow = dataset.normalize_inverse(x_flow)

    for d in range(x_flow.shape[1]):
        if keys is not None:
            key = keys[d]
        else:
            key = d
        # x_dataset_d = x_dataset[:, d]
        x_flow_d = x_flow[:, d]
        # sns.kdeplot(x_dataset_d.numpy(), label="dataset")
        sns.kdeplot(x_flow_d.numpy(), label="flow")
        plt.legend()
        plt.title(key)
        plt.tight_layout()
        plt.savefig(
            os.path.join(dir, f"flow_marginal_kdeplots_{key}.png"), bbox_inches="tight"
        )
        plt.close()
