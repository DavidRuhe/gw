from types import SimpleNamespace


cfg = SimpleNamespace()
cfg.seed = 42

cfg.simulator = SimpleNamespace(
    **dict(
        name="GaussianSimulator", output_path=f"/Users/druhe/Projects/gw/data/gaussian.npy"
    )
)
# cfg.simulator.name = "GaussianSimulator"
# cfg.simulator.datapath = "/Users/druhe/Projects/gw/data/gaussian.npy"
