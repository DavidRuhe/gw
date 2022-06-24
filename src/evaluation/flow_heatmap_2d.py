import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def flow_heatmap_2d(dir, model, dataloader, boundaries=(-3, 3), resolution=256):
    grid = torch.linspace(*boundaries, resolution)
    meshgrid = torch.meshgrid(grid, grid, indexing="xy")
    x = torch.stack(meshgrid, dim=-1).reshape(-1, 2)
    prob = model.log_prob(x).exp()

    df = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "z": prob})
    data = df.pivot(index="x", columns="y", values="z")
    sns.heatmap(data)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "flow_heatmap_2d.png"), bbox_inches="tight")
