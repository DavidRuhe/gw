"""
Torch/Pyro Implementation of (Unconditional) Normalizing Flow. In particular, as seen in RealNVP (z = x * exp(s) + t), where half of the 
dimensions in x are linearly scaled/transfromed as a function of the other half.
"""
import torch
from torch import nn
from pyro.nn import DenseNN
import pyro.distributions as dist
import pyro.distributions.transforms as T
import itertools
from functools import partial, reduce
import pytorch_lightning as pl
import operator

from torch.distributions.utils import lazy_property
from pyro.distributions import constraints


def pinv(p):
    """Reverses a permutation."""
    return sorted(range(len(p)), key=p.__getitem__)


class RealNVPBlock(T.AffineCoupling):
    def __init__(self, permutation, *args, dim=-1, **kwargs) -> None:
        super().__init__(*args, dim=dim, **kwargs)

        bijective = True
        volume_preserving = True

        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.permutation = permutation
        self.dim = dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, -self.dim)

    @lazy_property
    def inv_permutation(self):
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            self.permutation.size(0), dtype=torch.long, device=self.permutation.device
        )
        return result

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        x = x.index_select(self.dim, self.permutation)
        y = super()._call(x)
        y = y.index_select(self.dim, self.inv_permutation)
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        y = y.index_select(self.dim, self.inv_permutation)
        x = super()._inverse(y)
        x = x.index_select(self.dim, self.permutation)
        return x


class RealNVP(pl.LightningModule):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=16,
        num_layers=12,
        flow_length=16,
        lr=5e-4,
        split_dim=None,
    ):
        super().__init__()
        if split_dim is None:
            split_dim = input_dim // 2
        self.base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        self.param_dims = [input_dim - split_dim, input_dim - split_dim]
        # Define series of bijective transformations
        permutations = list(itertools.permutations(range(input_dim)))
        j = 0

        self.transforms = []
        # for i in range(flow_length):
        #     j = i % len(permutations)
        #     p = list(permutations[j])
        #     # self.transforms.append(T.Permute(torch.tensor(p)))
        #     # self.transforms.append(
        #     #     T.AffineCoupling(
        #     #         split_dim,
        #     #         DenseNN(split_dim, [hidden_dim] * num_layers, self.param_dims),
        #     #     )
        #     # )
        #     # self.transforms.append(T.Permute(torch.tensor(pinv(p))))
        #     breakpoint()
        #     print([hidden_dim] * num_layers, self.param_dims)
        #     self.transforms.append(
        #         RealNVPBlock(
        #             torch.tensor(p),
        #             split_dim,
        #             DenseNN(split_dim, [hidden_dim] * num_layers, self.param_dims),
        #         )
        #     )
        self.transforms = realnvp(input_dim, flow_length, hidden_dims=(16, 16))
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


import torch
from torch.distributions.utils import _sum_rightmost

from pyro.nn import DenseNN

from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.utils import clamp_preserve_gradients


def realnvp_block(
    input_dim, permutation, hidden_dims=None, split_dim=None, dim=-1, **kwargs
):
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError(
                "event shape {} must have same length as event_dim {}".format(
                    input_dim, -dim
                )
            )
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1) :], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [
            8 * event_shape[dim] * extra_dims,
            8 * event_shape[dim] * extra_dims,
        ]

    print(hidden_dims, (event_shape[dim] - split_dim) * extra_dims)
    hypernet = DenseNN(
        split_dim * extra_dims,
        hidden_dims,
        [
            (event_shape[dim] - split_dim) * extra_dims,
            (event_shape[dim] - split_dim) * extra_dims,
        ],
    )
    return RealNVPBlock(permutation, split_dim, hypernet, dim=dim, **kwargs)


def realnvp(input_dim, num_transforms, split_dim=None, hidden_dims=None):
    if split_dim is None:
        split_dim = input_dim // 2

    permutations = list(itertools.permutations(range(input_dim)))
    j = 0

    transforms = []
    for i in range(num_transforms):
        j = i % len(permutations)
        p = torch.tensor(list(permutations[j]))
        block = realnvp_block(input_dim, p, hidden_dims, split_dim)

        transforms.append(block)

        # self.transforms.append(Permute(torch.tensor(p)))
        # transforms.append(T.permute(input_dim, torch.tensor(p)))
        # transforms.append(T.affine_coupling(input_dim, hidden_dims, split_dim))
        # self.transforms.append(
        #     AffineCoupling(
        #         split_dim,
        #         DenseNN(split_dim, [hidden_dim] * num_layers, self.param_dims),
        #     )
        # )
        # transforms.append(T.permute(input_dim, torch.tensor(pinv(p))))

    return T.ComposeTransformModule(transforms)
    # return transforms
