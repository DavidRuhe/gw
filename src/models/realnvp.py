"""
Torch/Pyro Implementation of (Unconditional) Normalizing Flow. In particular, as seen in RealNVP (z = x * exp(s) + t), where half of the 
dimensions in x are linearly scaled/transfromed as a function of the other half.
"""
import torch
from torch import nn
from pyro.nn import DenseNN
import pyro.distributions as dist
from pyro.distributions.transforms import AffineCoupling, Permute
import itertools
import pytorch_lightning as pl


class RealNVP(pl.LightningModule):
    def __init__(
        self,
        input_dim=2,
        split_dim=1,
        hidden_dim=32,
        num_layers=1,
        flow_length=10,
        lr=5e-4,
    ):
        super().__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        self.param_dims = [input_dim - split_dim, input_dim - split_dim]
        # Define series of bijective transformations
        permutations = list(itertools.permutations(range(input_dim)))
        j = 0

        def pinv(p):
            """Reverses a permutation."""
            return sorted(range(len(p)), key=p.__getitem__)

        self.transforms = []
        for i in range(flow_length):
            j = i % len(permutations)
            p = list(permutations[j])
            self.transforms.append(Permute(torch.tensor(p)))
            self.transforms.append(
                AffineCoupling(
                    split_dim,
                    DenseNN(split_dim, [hidden_dim] * num_layers, self.param_dims),
                )
            )
            self.transforms.append(Permute(torch.tensor(pinv(p))))
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.transforms)

        self.trainable_modules = nn.ModuleList(
            [m for m in self.transforms if isinstance(m, nn.Module)]
        )
        self.lr = lr

    def step(self, batch, batch_idx):
        (x,) = batch
        loss = -self.log_prob(x).mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def parameters(self):
        return self.trainable_modules.parameters()

    def log_prob(self, x):
        return self.flow_dist.log_prob(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
