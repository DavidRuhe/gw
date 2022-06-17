from .base import cfg
from types import SimpleNamespace


cfg.simulator = SimpleNamespace(
    **dict(
        name="PowerPlusPeakSimulator",
        output_path=f"/Users/druhe/Projects/gw/data/gaussianpeak.npy",
        num_events=1025,
        num_posterior_samples=1024,
    )
)
