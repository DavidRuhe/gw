from models.realnvp import realnvp
from pyro.distributions.transforms import spline_coupling
import pyro.distributions.transforms as T


def spline_realnvp(input_dim, num_transforms, count_bins, split_dim=None, hidden_dims=None):
    count_bins=8
    num_transforms=4
    transform = []
    transform.append(spline_coupling(input_dim, split_dim=split_dim, hidden_dims=hidden_dims, count_bins=count_bins))
    transform.append(realnvp(input_dim, num_transforms, split_dim=split_dim, hidden_dims=hidden_dims))
    # Compose the transform
    return T.ComposeTransformModule(transform)