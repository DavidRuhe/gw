import math
import torch
from torch import nn
from collections.abc import Iterable

import numpy as np
from astropy.cosmology import Planck18 as cosmo
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

# M_RNG = (0.2, 100)


# z_axis = np.linspace(0, 10, 100000)
# dVdz = (
#     cosmo.differential_comoving_volume(z_axis).value / 1e9 * 4 * np.pi
# )  # Astropy dVcdz is per stradian
# # dVdz_interp = interp1d(z_axis, dVdz)
# z_axis = torch.from_numpy(z_axis)
# dVdz = torch.from_numpy(dVdz)

# dVdz_interp = lambda z: Interp1d()(z_axis, dVdz, z)


# def z_distribution_unnormalized(z):
#     # return torch.from_numpy(dVdz_interp(z)) * (1 + z) ** 1.7
#     return dVdz_interp(z) * (1 + z) ** 1.7


# z_normalization = torch.trapz(z_distribution_unnormalized(z_axis), z_axis)


# def z_distribution(z):
#     return dVdz_interp(z) * (1 + z) ** 1.7 / z_normalization


def log_prob(x, transforms, base_dist):
    x = x.clone()
    J = 0
    for t in transforms:
        y = t(x)
        J += t.log_abs_det_jacobian(x, y)
        x = y

    log_prob = base_dist.log_prob(x).sum(-1) + J
    return log_prob, y


def log_selection_bias(model, selection_data):
    x = selection_data[:, : model.d]
    logp, _ = log_prob(
        x, model.flows, model.base_dist(model.base_mean, model.base_logvar.exp())
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
        model.base_dist(model.base_mean, model.base_logvar.exp()),
    )
    logp = logp.view(x.shape[:-1])
    q = torch.tensor(1.0)
    if model.d >= 3 and gw_data.shape[-1] > 4:
        q = q * gw_data[:, :, 4] / 1e9  # z_prior, keep in mind.
    if model.d >= 4 and gw_data.shape[-1] > 5:
        q = q * gw_data[:, :, 5]
    logq = q.log()
    ll = torch.logsumexp(logp - prior_weight * logq, dim=0) - math.log(len(logp))

    sb = None
    if sb_weight > 0:
        if sel_data is not None:
            sb = log_selection_bias(model, sel_data)
            sb = sb.mean(0)
            loss = ll - sb * sb_weight
        else:
            print("Warning: no selection bias data provided.")
    else:
        loss = ll

    return loss, ll, sb


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class NormalizingFlow(Model):
    def __init__(self, dataset, flows, lr=1.0e-3):
        super().__init__()
        self.flows = flows
        self.d = dataset.dimensionality
        self.base_mean = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.base_logvar = nn.Parameter(torch.zeros(self.d), requires_grad=True)
        self.base_dist = torch.distributions.Normal
        self.lr = lr
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
            x, self.flows, self.base_dist(self.base_mean, self.base_logvar.exp())
        )
        return lp

    def parameters(self):
        return self.trainable_flows.parameters()

    def _process_metrics(self, metrics, prefix):
        return {f"{prefix}_{k}": v.mean() for k, v in metrics.items() if v is not None}

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        metrics = self._process_metrics(metrics, "train")
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss, metrics = self.step(batch, batch_idx)
        metrics = self._process_metrics(metrics, "val")
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        metrics = self._process_metrics(metrics, "test")
        self.log_dict(metrics)
        return loss


class HierarchicalNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx):
        (x,) = batch
        x = x[:, :, : self.d].clone()
        lp, y = log_prob(
            x.view(-1, self.d),
            self.flows,
            self.base_dist(self.base_mean, self.base_logvar.exp()),
        )
        lp = lp.view(x.shape[:-1])
        lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        loss = -lp.mean(0)
        metrics = {"loss": loss, "log-likelihood": lp}
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
                self.base_mean,
                self.base_logvar.exp(),
            )
            lp = lp.view(x.shape[:-1])
            lp = torch.logsumexp(lp, dim=0) - math.log(lp.shape[0])
        else:
            lp, y = log_prob(
                x, self.flows, self.base_dist, self.base_mean, self.base_logvar.exp()
            )
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

            log_prob = (
                self.base_dist(self.base_mean, self.base_logvar.exp())
                .log_prob(x)
                .sum(-1)
                + J
            )
            total_log_prob += log_prob

        return total_log_prob


class HierarchicalNormalizingFlowSB(NormalizingFlow):
    def __init__(self, sb_weight, prior_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sb_weight = sb_weight
        self.prior_weight = prior_weight

    def step(self, batch, batch_idx):
        (gw_data,), (sel_data,) = batch

        total_log_prob, log_prob, log_selection_bias = log_prob_sb(
            gw_data,
            sel_data,
            self,
            self.sb_weight,
            self.prior_weight,
        )

        loss = -total_log_prob

        metrics = {
            "loss": loss,
            "log_likelihood": log_prob,
            "log_likelihood_selection_bias": log_selection_bias,
        }

        loss = loss.mean()
        return loss, metrics

    def log_prob(self, gw_data, sel_data=None):

        if type(gw_data) in (tuple, list):
            (gw_data,) = gw_data
        if type(sel_data) in (tuple, list):
            (sel_data,) = sel_data

        if len(gw_data.shape) == 2:
            print("Got 2D data, reshaping to 3D")
            gw_data = gw_data[None]

        loss, log_prob, log_sb = log_prob_sb(
            gw_data,
            sel_data,
            self,
            self.sb_weight,
            self.prior_weight,
        )
        return log_prob

    def forward(self, *args, **kwargs):
        return self.log_prob(*args, **kwargs)
