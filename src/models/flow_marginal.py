import torch
from torch import nn
import pyro.distributions as dist
import pytorch_lightning as pl


class NormalizingFlow(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        transform,
        lr=1e-2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transform = transform

        self.base_dist = dist.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        self.flow_dist = dist.TransformedDistribution(self.base_dist, [self.transform])

        # self.trainable_modules = nn.ModuleList(
        #     [m for m in self.transforms if isinstance(m, nn.Module)]
        # )
        self.trainable_modules = nn.ModuleList([self.transform])
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