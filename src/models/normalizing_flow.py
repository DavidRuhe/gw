import math
import pyro.distributions as dist
import torch
import pyro.distributions.transforms as T
from torch import nn
import pytorch_lightning as pl
from collections.abc import Iterable

import numpy as np

from scipy.interpolate import interp1d

# from cmath import log
from astropy.cosmology import Planck18 as cosmo

from models.planar import BayesianPlanar
from utils.interp1d import Interp1d

# import emcee

# catalog_data = np.load(
#     "/mnt/home/wwong/ceph/GWProject/SymbolicPopulation/GWTC3-preprocessed-data/sampleDict_FAR_1_in_1_yr.pickle",
#     allow_pickle=True,
# )
# selection_data = np.load(
#     "/mnt/home/wwong/ceph/GWProject/SymbolicPopulation/GWTC3-preprocessed-data/injectionDict_FAR_1_in_1.pickle",
#     allow_pickle=True,
# )

# def log_population_likelihood(params):
#     selection_bias_ = selection_bias(likelihood, params)
#     log_likelihood = 0
#     for i in range(len(events_posterior)):
#         log_likelihood += np.log(
#             np.mean(
#                 likelihood(events_posterior[i], params)
#                 / events_prior[i]
#                 / selection_bias_
#             )
#         )

#     return log_likelihood

M_RNG = (0.2, 100)


z_axis = np.linspace(0, 10, 100000)
dVdz = (
    cosmo.differential_comoving_volume(z_axis).value / 1e9 * 4 * np.pi
)  # Astropy dVcdz is per stradian
# dVdz_interp = interp1d(z_axis, dVdz)
z_axis = torch.from_numpy(z_axis)
dVdz = torch.from_numpy(dVdz)

dVdz_interp = lambda z: Interp1d()(z_axis, dVdz, z)


def z_distribution_unnormalized(z):
    # return torch.from_numpy(dVdz_interp(z)) * (1 + z) ** 1.7
    return dVdz_interp(z) * (1 + z) ** 1.7


z_normalization = torch.trapz(z_distribution_unnormalized(z_axis), z_axis)


def z_distribution(z):
    return dVdz_interp(z) * (1 + z) ** 1.7 / z_normalization


def log_prob(x, transforms, base_dist):
    x = x.clone()
    J = 0
    for t in transforms:
        y = t(x)
        J += t.log_abs_det_jacobian(x, y)
        x = y

    log_prob = base_dist.log_prob(x).sum(-1) + J
    return log_prob, y


def likelihood(m1m2z, model, z_distribution):
    # p_m1 = truncated_powerlaw(x[:, 0], params[0], params[1], params[2])
    # p_m2 = M2_distribution(x[:, 1], x[:, 0], mmin=0)

    m1m2 = m1m2z[:, :2]
    p_m1m2 = model.log_prob(m1m2).exp()
    p_z = z_distribution(m1m2z[:, 2])
    return p_m1m2 * p_z


def log_selection_bias(model, selection_data):
    x = selection_data[:, : model.d]
    logp = model.log_prob(x)
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
):

    m1m2zchi = gw_data[:, :, :4]
    logp = model.log_prob(m1m2zchi.view(-1, 4)).view(m1m2zchi.shape[:-1])
    q_z = gw_data[:, :, 4] / 1e9  # z_prior, keep in mind.
    q_chieff = gw_data[:, :, 5]
    # logq = q_z.log()
    # ll = torch.logsumexp(logpm1m2z - logq, dim=0)
    ll = torch.logsumexp(logp, dim=0)

    if sb_weight > 0:

        sb = log_selection_bias(model, sel_data)
        sb = sb.mean(0)
        ll = ll + sb * sb_weight

    return ll


class NormalizingFlow(pl.LightningModule):
    def __init__(self, dataset, flows, lr=1.0e-3):
        super().__init__()

        self.flows = flows
        self.d = dataset.dimensionality
        self.base_dist = dist.Normal(torch.zeros(self.d), torch.ones(self.d))
        self.lr = lr
        if isinstance(flows, Iterable):
            self.trainable_flows = nn.ModuleList(
                [flow for flow in self.flows if isinstance(flow, nn.Module)]
            )
        else:
            self.trainable_flows = flows

    def log_prob(self, x):
        lp, _ = log_prob(x, self.flows, self.base_dist)
        return lp

    def parameters(self):
        return self.trainable_flows.parameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss


class HierarchicalNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx):
        (x,) = batch
        lp, y = log_prob(x.view(-1, self.d), self.flows, self.base_dist)
        lp = lp.view(x.shape[:-1])
        lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        loss = -lp.mean(0)
        return loss

    def forward(self, batch):

        if type(batch) in [list, tuple]:
            (x,) = batch
        else:
            x = batch

        if len(x.shape) == 3:
            x = x[:, :, : self.d].clone()
            lp, y = log_prob(x.view(-1, self.d), self.flows, self.base_dist)
            lp = lp.view(x.shape[:-1])
            lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        else:
            lp, y = log_prob(x, self.flows, self.base_dist)
        return lp


class HierarchicalMarginalNormalizingFlow(HierarchicalNormalizingFlow):
    def __init__(self, flows, *args, **kwargs):
        super().__init__(*args, flows=None, **kwargs)
        self.flows = flows

        self.trainable_flows = nn.ModuleList([f for flow in flows for f in flow])

    def log_prob(self, input):
        total_log_prob = 0
        for d in range(self.d):
            flows = self.flows[d]
            x = input[:, d][:, None]

            x = x.clone()
            J = 0
            for t in flows:
                y = t(x)
                J += t.log_abs_det_jacobian(x, y)
                x = y

            log_prob = self.base_dist.log_prob(x).sum(-1) + J
            total_log_prob += log_prob

        return total_log_prob


class HierarchicalNormalizingFlowSB(NormalizingFlow):
    def __init__(self, sb_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sb_weight = sb_weight

    def step(self, batch, batch_idx):
        (gw_data,), (sel_data,) = batch

        lp = log_prob_sb(
            gw_data,
            sel_data,
            self,
            self.sb_weight,
        )

        loss = -lp.mean(0)
        return loss

    # def log_prob(self, x):
    #     logpx, _ = log_prob(x, self.flows, self.base_dist)
    #     return logpx
    # if type(gw_data) in (tuple, list):
    #     (gw_data,) = gw_data
    # if len(gw_data.shape) == 2:

    # elif len(gw_data.shape) == 3:
    #     return log_prob_sb(
    #         gw_data,
    #         sel_input,
    #         self,
    #         self.sb_weight,
    #     )
    # else:
    #     raise ValueError("input must be 2 or 3 dimensional")

    def forward(self, *args, **kwargs):
        return self.log_prob(*args, **kwargs)

    # def step(self, batch, batch_idx):
    #     (x,) = batch
    #     lp, y = log_prob(x.view(-1, 2), self.flows, self.base_dist)
    #     lp = lp.view(x.shape[:-1])
    #     lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
    #     loss = -lp.mean(0)
    #     return loss


# def truncated_powerlaw(m1, index, mmin=5, mmax=50):
#     try:
#         output = m1.copy() ** index
#         index_in = np.where((m1 >= mmin) * (m1 <= mmax))[0]
#         index_out = np.where((m1 < mmin) + (m1 > mmax))[0]
#         normalization = ((mmax) ** (1 + index) - mmin ** (1 + index)) / (1 + index)
#         output[index_out] = 1e-30
#         output[index_in] = m1[index_in] ** index / normalization
#     except ZeroDivisionError:
#         output = m1.copy() ** index
#         index_in = np.where((m1 >= mmin) * (m1 <= mmax))[0]
#         index_out = np.where((m1 < mmin) + (m1 > mmax))[0]
#         normalization = np.log(mmax) - np.log(mmin)
#         output[index_out] = 1e-30
#         output[index_in] = m1[index_in] ** index / normalization
#     return output


# m1_axis = np.linspace(0.01, 100, 10000)


# def M2_distribution(m2, m1, mmin):
#     return 3 * m2**2 / (m1**3 - mmin**3)


# events_posterior = []
# events_prior = []
# for event in catalog_data:
#     event_data = catalog_data[event]
#     if np.median(event_data["m2"]) > 3:
#         posterior = np.stack(
#             [event_data["m1"], event_data["m2"], event_data["z"]], axis=1
#         )
#         prior = event_data["z_prior"] / 1e9
#         events_posterior.append(posterior)
#         events_prior.append(prior)


class HierarchicalNormalizingBayesianFlowSB(HierarchicalNormalizingFlowSB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx):
        (gw_data,), (sel_data,) = batch

        lp = log_prob_sb(
            gw_data,
            sel_data,
            self,
            self.sb_weight,
        )

        loss = -lp.mean(0)

        h = self.entropy()
        h = h / len(lp)

        self.log("nll", loss, on_epoch=True, on_step=False)
        self.log("h", h, on_epoch=True, on_step=False)
        loss = loss + 1 * h
        return loss

    def entropy(self):
        h = 0
        for m in self.modules():
            if isinstance(m, BayesianPlanar):
                h += m.entropy()

        return h
