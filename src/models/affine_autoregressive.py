import math

import pyro.distributions as dist
import pyro.distributions.transforms as T
import pytorch_lightning as pl
import torch
from nn.dense_nn import DenseNN
from torch import nn

from utils import count_parameters


class AffineAutoregressiveFlow(pl.LightningModule):
    def __init__(self, d=1, objective="mle", precision=1, num_layers=1):
        super().__init__()

        self.objective = objective
        self.precision = precision
        assert self.objective in ["mle", "map"]

        self.base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
        self.transform = nn.ModuleList(
            [T.affine_autoregressive(d, hidden_dims=[d*8, d*8]) for _ in range(num_layers)]
        )
        self.flow = dist.TransformedDistribution(self.base_dist, list(self.transform))

        print(f"Number of parameters: {count_parameters(self)}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def step(self, batch, batch_idx):
        (x_posterior,) = batch
        log_prob = torch.logsumexp(
            self.flow.log_prob(x_posterior.reshape(-1, 1)).view(x_posterior.shape),
            dim=-1,
        )

        log_prob = log_prob - math.log(x_posterior.shape[-1])

        loss = -log_prob.mean()

        self.log("nll", loss, prog_bar=True)

        if self.objective == "map":
            weights = torch.cat(
                [p.flatten() for p in self.parameters() if p.requires_grad]
            )
            prior = self.precision * weights.dot(weights)
            self.log("prior", prior, prog_bar=True)
            loss += prior

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss
