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
dVdz_interp = interp1d(z_axis, dVdz)


def z_distribution_unnormalized(z):
    return dVdz_interp(z) * (1 + z) ** 1.7


z_normalization = np.trapz(z_distribution_unnormalized(z_axis), z_axis)


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


def log_selection_bias(
    flows, base_dist, model, selection_m1m2z, p_draw_m1m2z, selection_trials
):
    # p_m1m2z = likelihood(selection_m1m2z, model)
    # likelihood_samples = p_m1m2z / selection_x["p_draw_m1m2z"]
    # return np.sum(likelihood_samples / selection_data["nTrials"])

    # m1m2, z = (
    #     selection_m1m2z[:, :2],
    #     selection_m1m2z[:, 2],
    # )
    # logpm1m2, _ = log_prob(m1m2, flows, base_dist)
    # logpz = torch.from_numpy(np.log(z_distribution(z.numpy())))
    # logpm1m2z = logpm1m2 + logpz
    logpm1m2z = model.log_prob(selection_m1m2z)
    log_pdraw = p_draw_m1m2z.log()
    return torch.logsumexp(logpm1m2z - log_pdraw, dim=0) - math.log(selection_trials)


def log_prob_sb(
    x,
    flows,
    base_dist,
    model,
    selection_m1m2z,
    p_draw_m1m2z,
    selection_trials,
    sb_weight,
):


    m1m2z = x[:, :, :3]
    logpm1m2z = model.log_prob(m1m2z.view(-1, 3)).view(m1m2z.shape[:-1])
    q_z = x[:, :, 3] / 1e9  # z_prior, keep in mind.
    logq = q_z.log()
    ll = torch.logsumexp(logpm1m2z - logq, dim=0)

    if sb_weight > 0:

        sb = log_selection_bias(
            flows, base_dist, model, selection_m1m2z, p_draw_m1m2z, selection_trials
        )
        ll = ll + sb * sb_weight

    return ll


class NormalizingFlow(pl.LightningModule):
    def __init__(self, flows, lr=1.0e-3, d=1):
        super().__init__()

        self.flows = flows
        self.base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
        self.lr = lr
        if isinstance(flows, Iterable):
            self.trainable_flows = nn.ModuleList(
                [flow for flow in self.flows if isinstance(flow, nn.Module)]
            )
        else:
            self.trainable_flows = flows

        self.d = d

    def log_prob(self, x):
        lp, _ = log_prob(x, self.flows, self.base_dist)
        return lp

    def parameters(self):
        return self.trainable_flows.parameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        return loss


class HierarchicalNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx):
        (x,) = batch
        lp, y = log_prob(x.view(-1, 2), self.flows, self.base_dist)
        lp = lp.view(x.shape[:-1])
        lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        loss = -lp.mean(0)
        return loss


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
    def __init__(self, selection_data, sb_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_data = selection_data
        self.selection_m1m2z = torch.from_numpy(
            np.stack(
                [selection_data["m1"], selection_data["m2"], selection_data["z"]],
                axis=-1,
            )
        ).float()
        self.p_draw_m1m2z = torch.from_numpy(selection_data["p_draw_m1m2z"]).float()
        self.selection_trials = torch.tensor(selection_data["nTrials"])
        self.sb_weight = sb_weight

    def step(self, batch, batch_idx):
        (x,) = batch
        lp = log_prob_sb(
            x,
            self.flows,
            self.base_dist,
            self,
            self.selection_m1m2z,
            self.p_draw_m1m2z,
            self.selection_trials,
            self.sb_weight,
        )
        loss = -lp.mean(0)
        return loss

    def log_prob(self, input):
        assert len(input.shape) == 2
        m1m2 = input[:, :2]
        logpm1m2, _ = log_prob(m1m2, self.flows, self.base_dist)
        z = input[:, 2]
        logpz = torch.from_numpy(np.log(z_distribution(z.view(-1).numpy()))).view(
            z.shape
        )
        logpm1m2z = logpm1m2 + logpz
        return logpm1m2z

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
