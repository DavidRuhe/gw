from .base import cfg
from types import SimpleNamespace


cfg.simulator = SimpleNamespace(
    **dict(
        name="GaussianSimulator",
        output_path=f"/Users/druhe/Projects/gw/data/gaussian.npy",
    )
)
