import pyro.distributions.transforms as T


def planar(
    *args,
    num_layers=1,
    **kwargs,
):
    return [T.planar(*args, **kwargs) for _ in range(num_layers)]
