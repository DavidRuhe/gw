import argparse

from configs.parse import add_arguments

# from argparse import Namespace
# from types import SimpleNamespace

# class Config(Namespace):
#     def __init__(self, name, **kwargs):
#         super().__init__(name=name, **kwargs)

#     def pop(self, k):
#         return self.__dict__.pop(k)


# cfg = SimpleNamespace()
# cfg.seed = 42
# cfg.dataroot = (
#     "/Users/druhe/Projects/gw/MLPopulation/data/Combined_GWTC_m1m2chieffz.npz"
# )
# cfg.batch_size = 512
# cfg.dataset = "circles"
# cfg.lr = 1e-3
# cfg.epochs = 64

base_parser = argparse.ArgumentParser(add_help=False)
base_cfg = dict(
    seed=42,
    batch_size=512,
)
add_arguments(base_parser, base_cfg)
