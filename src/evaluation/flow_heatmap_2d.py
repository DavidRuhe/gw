import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


@torch.no_grad()
def flow_heatmap_2d(dir, model, dataset, boundaries=(-3, 3), resolution=256):
    grid = torch.linspace(*boundaries, resolution)
    meshgrid = torch.meshgrid(grid, grid, indexing="xy")
    x = torch.stack(meshgrid, dim=-1).reshape(-1, 2)
    prob = model.log_prob(x).exp().view(resolution, resolution)
    plt.imshow(prob, cmap='jet')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "flow_heatmap_2d.png"), bbox_inches="tight")
    plt.close()
