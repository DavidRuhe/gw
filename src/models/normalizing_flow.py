import math
import pyro.distributions as dist
import torch
from torch import nn
from collections.abc import Iterable
import pyro.distributions.transforms as T

import numpy as np
from models.partial_transform import PartialTransform

from astropy.cosmology import Planck18 as cosmo

from utils.interp1d import Interp1d


z_axis = np.linspace(0, 10, 100000)
dVdz = (
    cosmo.differential_comoving_volume(z_axis).value / 1e9 * 4 * np.pi
)  # Astropy dVcdz is per stradian
z_axis = torch.from_numpy(z_axis)
dVdz = torch.from_numpy(dVdz)
dVdz_interp = lambda z: Interp1d()(z_axis, dVdz, z)


def z_distribution_unnormalized(z):
    return dVdz_interp(z) * (1 + z) ** 1.7


z_normalization = torch.trapz(z_distribution_unnormalized(z_axis), z_axis)


def z_distribution(z):
    return dVdz_interp(z).squeeze(0) * (1 + z) ** 1.7 / z_normalization


def log_prob(x, transforms, base_dist, base_loc, base_scale):
    x = x.clone()
    J = 0
    for t in transforms:
        y = t(x)
        J_ = t.log_abs_det_jacobian(x, y)
        if len(J_.shape) == 2:
            J_ = J_.sum(1)
        J = J + J_
        x = y

    log_prob = base_dist(base_loc, base_scale).log_prob(x).sum(-1) + J
    return log_prob, y


def log_selection_bias(model, selection_data):
    x = selection_data[:, : model.d]
    logp, _ = log_prob(
        x, model.flows, model.base_dist, model.base_loc, model.base_logscale.exp()
    )
    p_drawm1m2z = selection_data[:, 4]
    p_drawchi = selection_data[:, 5]
    ntrials = selection_data[:, 6]
    naccepted = selection_data[:, 7]
    fraction = naccepted / ntrials
    ntrials_eff = len(x) / fraction
    return torch.logsumexp(logp - p_drawm1m2z - p_drawchi, dim=0) - torch.log(
        ntrials_eff
    )


def log_prob_sb(
    gw_x,
    sel_x,
    model,
    sb_weight,
    prior_weight,
):

    if len(gw_x.shape) == 2:
        gw_x = gw_x[None]

    z_prior = gw_x[:, :, 4].clone() / 1e9
    chi_prior = gw_x[:, :, 5].clone()

    if not model.train_z:
        z = gw_x[:, :, 2].clone()
        gw_x = torch.cat([gw_x[:, :, :2], gw_x[:, :, 3:]], dim=-1).clone()
        assert model.d == 3

    gw_x_ = gw_x[:, :, : model.d].clone().view(-1, model.d)

    logp, y = log_prob(
        gw_x_,
        model.flows,
        model.base_dist,
        model.base_loc,
        model.base_logscale.exp(),
    )
    logp = logp.view(gw_x.shape[:-1])

    q = torch.tensor(1.0)
    # if model.d >= 3 and gw_data.shape[-1] > 4:
    q = q * z_prior  # z_prior, keep in mind.
    # if model.d >= 4 and gw_data.shape[-1] > 5:
    q = q * chi_prior
    logq = q.log()

    if not model.train_z:
        pz = z_distribution(z.view(-1))
        logpz = pz.log()
        logp = logp + logpz.view(logp.shape)

    ll = torch.logsumexp(logp - prior_weight * logq, dim=0) - math.log(len(logp))

    if sb_weight > 0:
        if sel_x is not None:
            sb = log_selection_bias(model, sel_x)
            sb = sb.mean(0)
            # model.log("log_selection_bias", sb.mean(), on_epoch=True, on_step=False)
            ll = ll - sb * sb_weight
        else:
            print("Warning: no selection bias data provided.")

    return ll, y


class Model(nn.Module):
    def __init__(self, device="auto"):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)


class NormalizingFlow(Model):
    def __init__(
        self, dataset, flows, dims_to_constrain_positive=None, positive_transform=None
    ):
        super().__init__()

        self.dims_to_constrain_positive = dims_to_constrain_positive
        if positive_transform is not None:
            if positive_transform == "softplus":
                flows.insert(
                    0,
                    PartialTransform(
                        T.SoftplusTransform().inv, dims_to_constrain_positive
                    ),
                )
            if positive_transform == "log":
                flows.insert(
                    0,
                    PartialTransform(T.ExpTransform().inv, dims_to_constrain_positive),
                )
            else:
                raise ValueError(f"Unknown positive transform {positive_transform}")
        self.flows = flows
        self.d = self.flows[-1].input_dim
        self.register_buffer("base_loc", torch.zeros(self.d))
        self.register_buffer("base_logscale", torch.zeros(self.d))

        self.base_dist = dist.Normal
        if isinstance(flows, Iterable):
            self.trainable_flows = nn.ModuleList(
                [flow for flow in self.flows if isinstance(flow, nn.Module)]
            )
        else:
            self.trainable_flows = flows

    def log_prob(self, x):

        if type(x) in (list, tuple):
            (x,) = x
        lp, _ = log_prob(
            x, self.flows, self.base_dist, self.base_loc, self.base_logscale.exp()
        )
        return lp

    def forward(self, x):
        return self.log_prob(x)

    def parameters(self):
        return self.trainable_flows.parameters()

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        return loss, metrics

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        return loss, metrics

    def test_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        return loss, metrics


class HierarchicalNormalizingFlow(NormalizingFlow):
    def __init__(self, train_z=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_z = train_z

    def step(self, batch, batch_idx):

        lp = self(batch)
        loss = -lp.mean(0)
        metrics = {"loss": loss}
        return loss, metrics

    def log_prob(self, batch):

        if type(batch) in [list, tuple]:
            (x,) = batch
        else:
            x = batch

        if len(x.shape) == 2:
            x = x[None]

        if not self.train_z:
            z = x[:, :, 2].clone()
            x = torch.cat([x[:, :, :2], x[:, :, 3:]], dim=-1).clone()

        x = x[:, :, : self.d].clone()

        x_ = x.view(-1, self.d)
        lp, y = log_prob(
            x_,
            self.flows,
            self.base_dist,
            self.base_loc,
            self.base_logscale.exp(),
        )

        lp = lp.view(x.shape[:-1])

        if not self.train_z:
            pz = z_distribution(z.view(-1))
            logpz = pz.log()
            lp = lp + logpz.view(lp.shape)

        lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])

        return lp

    def forward(self, batch):
        return self.log_prob(batch)


class HierarchicalNormalizingFlowSB(HierarchicalNormalizingFlow):
    def __init__(self, sb_weight, prior_weight, train_z=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sb_weight = sb_weight
        self.prior_weight = prior_weight
        self.train_z = train_z

    def step(self, batch, batch_idx):

        lp = self.log_prob_sb(batch)
        loss = -lp.mean(0)
        metrics = {"loss": loss}
        return loss, metrics

    def log_prob_sb(self, batch):

        gw_x, sel_x = batch

        lp, y = log_prob_sb(
            gw_x,
            sel_x,
            self,
            self.sb_weight,
            self.prior_weight,
        )

        return lp
