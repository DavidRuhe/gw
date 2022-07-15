import itertools
import pyro.distributions.transforms as T
from pyro.nn import DenseNN


def realnvp(num_transforms, input_dim, split_dim=None, hidden_dims=[32]):

    if split_dim is None:
        split_dim = input_dim // 2

    param_dims = [input_dim - split_dim, input_dim - split_dim]

    transforms = [
        T.AffineCoupling(split_dim, DenseNN(split_dim, hidden_dims, param_dims))
        for _ in range(num_transforms)
    ]
    perms = [T.permute(input_dim) for _ in range(num_transforms)]
    flows = list(itertools.chain(*zip(transforms, perms)))[:-1]

    return flows
