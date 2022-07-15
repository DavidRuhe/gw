# import math

# import pyro.distributions as dist
import pyro.distributions.transforms as T

# import pytorch_lightning as pl
# import torch
from nn.dense_nn import DenseNN

# from torch import nn

# from utils import count_parameters


# def hierarchical_log_prob(flow, x):
#     """ "
#     :param flow: A :class:`~pyro.distributions.transforms.SplineCoupling` object
#     :param x: A sample
#     :return: The log probability of the sample
#     """

#     return torch.logsumexp(
#         flow.log_prob(x.reshape(-1, 1)).view(x.shape), dim=-1
#     ) - math.log(x.shape[-1])


# def spline_coupling(
#     input_dim, split_dim=None, hidden_dims=None, count_bins=8, bound=3.0
# ):
#     """
#     A helper function to create a
#     :class:`~pyro.distributions.transforms.SplineCoupling` object for consistency
#     with other helpers.

#     :param input_dim: Dimension of input variable
#     :type input_dim: int

#     """

#     if split_dim is None:
#         split_dim = input_dim // 2

#     if hidden_dims is None:
#         hidden_dims = [input_dim * 10, input_dim * 10]

#     nn = DenseNN(
#         split_dim,
#         hidden_dims,
#         param_dims=[
#             (input_dim - split_dim) * count_bins,
#             (input_dim - split_dim) * count_bins,
#             (input_dim - split_dim) * (count_bins - 1),
#             (input_dim - split_dim) * count_bins,
#         ],  # 3K - 1
#     )

#     return T.SplineCoupling(input_dim, split_dim, nn, count_bins, bound)


# class SplineCouplingFlow(pl.LightningModule):
#     def __init__(self, d=1, objective="mle", precision=1, hierarchical=True):
#         super().__init__()

#         self.objective = objective
#         self.precision = precision
#         assert self.objective in ["mle", "map"]

#         self.base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
#         # self.spline_transform = spline_coupling(d)
#         self.spline_transform = spline_coupling(d)
#         self.flow = dist.TransformedDistribution(
#             self.base_dist, [self.spline_transform]
#         )
#         self.hierarchical = hierarchical

#         print(f"Number of parameters: {count_parameters(self)}")

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters())

#     def step(self, batch, batch_idx):
#         (x,) = batch
#         if self.hierarchical:
#             log_prob = hierarchical_log_prob(self.flow, x)

#         else:
#             log_prob = self.flow.log_prob(x)

#         loss = -log_prob.mean()

#         self.log("nll", loss, prog_bar=True)

#         if self.objective == "map":
#             weights = torch.cat(
#                 [p.flatten() for p in self.parameters() if p.requires_grad]
#             )
#             prior = self.precision * weights.dot(weights)
#             self.log("prior", prior, prog_bar=True)
#             loss += prior

#         return loss

#     def training_step(self, batch, batch_idx):
#         return self.step(batch, batch_idx)

#     def validation_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         return loss


def spline_coupling(
    input_dim,
    *args,
    num_layers=1,
    **kwargs,
):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.SplineCoupling` object for consistency
    with other helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int

    """
    # return [T.SplineCoupling(input_dim, split_dim, nn, count_bins, bound)]
    return [T.spline_coupling(input_dim, *args, **kwargs) for _ in range(num_layers)]
