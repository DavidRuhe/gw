from types import SimpleNamespace


cfg = SimpleNamespace()
cfg.seed = 42
cfg.dataroot = (
    "/Users/druhe/Projects/gw/MLPopulation/data/Combined_GWTC_m1m2chieffz.npz"
)
cfg.batch_size = 512
cfg.dataset = "circles"
cfg.lr = 1e-3
cfg.epochs = 64
