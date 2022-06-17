from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.experiment = SimpleNamespace(
    **dict(
        name="MarginalFlowExperiment",
    )
)

cfg.dataset = SimpleNamespace(**dict(name="ToyDataset", dataset_name="double_moons", fold=0))
