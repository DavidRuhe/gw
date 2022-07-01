import math
import pyro.distributions as dist
import torch
import pyro.distributions.transforms as T
from torch import nn
import pytorch_lightning as pl
from collections.abc import Iterable


class NormalizingFlow(pl.LightningModule):
    def __init__(self, flows, lr=1.0e-3, d=1):
        super().__init__()

        self.flows = flows
        self.base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
        self.flow_dist = dist.TransformedDistribution(self.base_dist, flows)
        self.lr = lr
        if isinstance(flows, Iterable):
            self.trainable_flows =  nn.ModuleList([flow for flow in self.flows if isinstance(flow, nn.Module)])
        else:
            self.trainable_flows = flows

    def parameters(self):
        return self.trainable_flows.parameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_prob(self, x):
        return self.flow_dist.log_prob(x)

    def step(self, batch, batch_idx):
        (x,) = batch
        log_prob = self.flow_dist.log_prob(x)
        loss = -log_prob.mean()
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        return loss
# class NormalizingFlow(pl.LightningModule):
#     def __init__(self, flows,lr,  d=1):
#         super().__init__()

#         self.base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
#         self.spline_transform = T.spline_coupling(d)
#         self.flow_dist = dist.TransformedDistribution(
#             self.base_dist, [self.spline_transform]
#         )

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.spline_transform.parameters(), lr=5e-3)

#     def step(self, batch, batch_idx):
#         (x_posterior,) = batch

#         log_prob = torch.logsumexp(
#             self.flow_dist.log_prob(x_posterior.reshape(-1, 1)).view(x_posterior.shape),
#             dim=-1,
#         )

#         log_prob = log_prob - math.log(x_posterior.shape[-1])

#         return -log_prob.mean()

#     def training_step(self, batch, batch_idx):
#         return self.step(batch, batch_idx)

#     def validation_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         return loss




class HierarchicalNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx):
        (x,) = batch

        log_prob = torch.logsumexp(
            self.flow_dist.log_prob(x.reshape(-1, 1)).view(x.shape),
            dim=-1,
        )

        log_prob = log_prob - math.log(x.shape[-1])

        return -log_prob.mean()
