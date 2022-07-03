import matplotlib.pyplot as plt
import torch
import numpy as np
from data.m1m2 import M1M2Dataset
import pyro.distributions as dist
import pyro.distributions.transforms as T
import math
import seaborn as sns


train_dataset = M1M2Dataset("/Users/druhe/Projects/gw/datasets/Combined_GWTC_m1m2chieffz.npz", "train", 0)
valid_dataset = M1M2Dataset("/Users/druhe/Projects/gw/datasets/Combined_GWTC_m1m2chieffz.npz", "valid", 0)
test_dataset = M1M2Dataset("/Users/druhe/Projects/gw/datasets/Combined_GWTC_m1m2chieffz.npz", "test", 0)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=128,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
)


dim = 2
steps = 5000

base_dist = dist.Normal(torch.zeros(dim), torch.ones(dim))
spline_transform = T.ComposeTransformModule(
    [T.spline_autoregressive(dim) for _ in range(4)]
)
flow_dist = dist.TransformedDistribution(base_dist, spline_transform)


best_val_loss = float("inf")
batch_size = 30000
optimizer = torch.optim.Adam(spline_transform.parameters(), lr=5e-3)
for step in range(steps + 1):
    for (X,) in train_loader:
        optimizer.zero_grad()
        randindex = torch.randint(0, 30000, (batch_size,))
        # batch = X[:, randindex, :]
        batch = X

        log_prob = flow_dist.log_prob(batch.view(-1, dim)).view(batch.shape[:-1])

        log_prob = torch.logsumexp(log_prob, dim=-1) - math.log(log_prob.shape[-1])
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()
        flow_dist.clear_cache()

    valid_loss = 0
    for (X,) in valid_loader:
        randindex = torch.randint(0, 30000, (batch_size,))
        # batch = X[:, randindex, :]
        batch = X
        log_prob = flow_dist.log_prob(batch.view(-1, dim)).view(batch.shape[:-1])
        log_prob = torch.logsumexp(log_prob, dim=-1) - math.log(log_prob.shape[-1])
        loss = -log_prob.mean()

        valid_loss += loss.item()

    if loss < best_val_loss:
        best_val_loss = loss
        print(f"Best val loss: {best_val_loss}")

        with torch.no_grad():
            if dim == 1:
                X = torch.linspace(-5, 5, 128)
                p = flow_dist.log_prob(X[:, None]).exp()
                plt.plot(X.numpy(), p.numpy())
                plt.close()
            else:
                linspace = torch.linspace(-5, 5, 128)
                grid = torch.meshgrid(linspace, linspace, indexing="xy")
                grid = torch.stack(grid).view(2, -1).permute(1, 0)
                p = flow_dist.log_prob(grid).exp().view(128, 128)
                plt.plot(linspace.numpy(), p.sum(0).numpy())
                plt.savefig(f"/Users/druhe/Projects/gw/p0_{step}.png")
                plt.close()

                plt.plot(linspace.numpy(), p.sum(1).numpy())
                plt.savefig(f"/Users/druhe/Projects/gw/p1_{step}.png")
                plt.close()

                samples = flow_dist.sample((1024,))
                sns.kdeplot(samples[:, 0].numpy())
                sns.kdeplot(X[:, :, 0].mean(1).numpy())
                plt.close()

                plt.plot(linspace.numpy(), p.sum(1).numpy())
                plt.savefig(f"/Users/druhe/Projects/gw/kde0_{step}.png")
                plt.close()

                sns.kdeplot(samples[:, 1].numpy())
                sns.kdeplot(X[:, :, 1].mean(1).numpy())
                plt.savefig(f"/Users/druhe/Projects/gw/kde1_{step}.png")
                plt.close()

    else:
        break


        # log_prob_val = flow_dist.log_prob(X_val.view(-1, dim)).view(
        #     X_val.shape[:-1]
        # )

        # log_prob_val = torch.logsumexp(log_prob_val, dim=-1) - math.log(
        #     log_prob_val.shape[-1]
        # )
        # val_loss = -log_prob_val.mean()
        # if val_loss > best_val_loss:
        #     break
        # else:
        #     best_val_loss = val_loss
        # print("val loss", loss.item())
