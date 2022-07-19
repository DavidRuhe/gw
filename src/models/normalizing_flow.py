import math
import pyro.distributions as dist
import torch
import pyro.distributions.transforms as T
from torch import nn
from collections.abc import Iterable

import numpy as np


from astropy.cosmology import Planck18 as cosmo

from models.planar import BayesianPlanar
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
    return dVdz_interp(z) * (1 + z) ** 1.7 / z_normalization


def log_prob(x, transforms, base_dist, base_loc, base_scale):
    x = x.clone()
    J = 0
    for t in transforms:
        y = t(x)
        J += t.log_abs_det_jacobian(x, y)
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
    gw_data,
    sel_data,
    model,
    sb_weight,
    prior_weight,
):

    x = gw_data[:, :, : model.d]
    logp, _ = log_prob(
        x.view(-1, model.d),
        model.flows,
        model.base_dist,
        model.base_loc,
        model.base_logscale.exp(),
    )
    logp = logp.view(x.shape[:-1])
    q = torch.tensor(1.0)
    if model.d >= 3 and gw_data.shape[-1] > 4:
        q = q * gw_data[:, :, 4] / 1e9  # z_prior, keep in mind.
    if model.d >= 4 and gw_data.shape[-1] > 5:
        q = q * gw_data[:, :, 5]
    logq = q.log()
    ll = torch.logsumexp(logp - prior_weight * logq, dim=0) - math.log(len(logp))

    # model.log("log_prob", ll.mean(), on_epoch=True, on_step=False)

    if sb_weight > 0:
        if sel_data is not None:
            sb = log_selection_bias(model, sel_data)
            sb = sb.mean(0)
            # model.log("log_selection_bias", sb.mean(), on_epoch=True, on_step=False)
            ll = ll - sb * sb_weight
        else:
            print("Warning: no selection bias data provided.")

    return ll


class Model(nn.Module):
    def __init__(self, device="auto"):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)


class NormalizingFlow(Model):
    def __init__(self, dataset, flows, dims_to_constrain_positive=None, positive_transform=None):
        super().__init__()

        self.dims_to_constrain_positive = dims_to_constrain_positive
        if positive_transform is not None:
            if positive_transform == 'softplus':
                breakpoint()
        self.flows = flows
        self.d = dataset.dimensionality
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx):

        if type(batch) in [list, tuple]:
            (x,) = batch
        else:
            x = batch

        x = x[:, :, : self.d].clone()
        lp, y = log_prob(
            x.view(-1, self.d),
            self.flows,
            self.base_dist,
            self.base_loc,
            self.base_logscale.exp(),
        )
        lp = lp.view(x.shape[:-1])
        lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        loss = -lp.mean(0)
        metrics = {"loss": loss}
        return loss, metrics

    def forward(self, batch):

        if type(batch) in [list, tuple]:
            (x,) = batch
        else:
            x = batch

        if len(x.shape) == 3:
            x = x[:, :, : self.d].clone()
            lp, y = log_prob(
                x.view(-1, self.d),
                self.flows,
                self.base_dist,
                self.base_loc,
                self.base_logscale.exp(),
            )
            lp = lp.view(x.shape[:-1])
            lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        else:
            lp, y = log_prob(
                x, self.flows, self.base_dist, self.base_loc, self.base_logscale.exp()
            )
        return lp


class HierarchicalNormalizingFlowSB(NormalizingFlow):
    def __init__(self, sb_weight, prior_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sb_weight = sb_weight
        self.prior_weight = prior_weight

    def step(self, batch, batch_idx):
        gw_data, sel_data = batch

        lp = log_prob_sb(
            gw_data,
            sel_data,
            self,
            self.sb_weight,
            self.prior_weight,
        )

        loss = -lp.mean(0)
        metrics = {"loss": loss}
        return loss, metrics

    def log_prob(self, gw_data, sel_data=None, verbose=False):

        if type(gw_data) in (tuple, list):
            (gw_data,) = gw_data
        if type(sel_data) in (tuple, list):
            (sel_data,) = sel_data

        if len(gw_data.shape) == 2:
            if verbose:
                print("Got 2D data, reshaping to 3D")
            gw_data = gw_data[None]

        return log_prob_sb(
            gw_data,
            sel_data,
            self,
            self.sb_weight,
            self.prior_weight,
        )

    def forward(self, *args, **kwargs):
        return self.log_prob(*args, **kwargs)