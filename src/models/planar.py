import pyro.distributions.transforms as T
import torch
from torch import nn
import math
from torch.distributions import constraints
import pyro
from pyro.distributions.transforms.planar import ConditionedPlanar
from pyro.distributions.torch_transform import TransformModule


def planar(
    *args,
    num_layers=1,
    **kwargs,
):
    return [T.planar(*args, **kwargs) for _ in range(num_layers)]


def gaussian_kl(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.ones_like(mu1)

    return 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)


def gaussian_h(logvar1):
    return -0.5 - 0.5 * (math.log(2 * math.pi) + logvar1)


class BayesianPlanar(ConditionedPlanar, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim):
        super().__init__(self._params)

        self.bias = nn.Parameter(
            torch.Tensor(
                1,
            )
        )
        self.u = nn.Parameter(
            torch.Tensor(
                input_dim,
            )
        )
        self.w = nn.Parameter(
            torch.Tensor(
                input_dim,
            )
        )

        self.bias_logvar = nn.Parameter(
            torch.Tensor(
                1,
            )
        )
        self.u_logvar = nn.Parameter(
            torch.Tensor(
                input_dim,
            )
        )
        self.w_logvar = nn.Parameter(
            torch.Tensor(
                input_dim,
            )
        )
        self.input_dim = input_dim
        self.reset_parameters()

        self._entropy = None

    def _params(self):
        bias_std = torch.exp(self.bias_logvar / 2)
        u_std = torch.exp(self.u_logvar / 2)
        w_std = torch.exp(self.w_logvar / 2)

        # c = math.log(2 * math.pi)
        # self._entropy = (
        #     0.5 * (self.bias_logvar.sum() + c + 1)
        #     + 0.5 * (self.u_logvar.sum() + c + 1)
        #     + 0.5 * (self.w_logvar.sum() + c + 1)
        # )
        # self

        # bias = self.bias + bias_std * torch.randn_like(bias_std)
        bias = self.bias
        self._entropy = (
            gaussian_kl(self.u, u_std**2, self.u, torch.ones_like(u_std)).sum()
            + gaussian_kl(self.w, w_std**2, self.w, torch.ones_like(w_std)).sum()
        )
        # self._entropy = (
        #     0.5 * (self.bias_logvar.sum() + c + 1)
        #     + 0.5 * (self.u_logvar.sum() + c + 1)
        #     + 0.5 * (self.w_logvar.sum() + c + 1)
        # )
        u = self.u + u_std * torch.randn_like(u_std)
        w = self.w + w_std * torch.randn_like(w_std)
        return bias, u, w

    def entropy(self):
        e = self._entropy
        self._entropy = None
        return e

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.u.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.u.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        self.bias_logvar.data.fill_(-32)
        self.u_logvar.data.fill_(-32)
        self.w_logvar.data.fill_(-32)


def bayesian_planar(
    *args,
    num_layers=1,
    **kwargs,
):
    return [BayesianPlanar(*args, **kwargs) for _ in range(num_layers)]
