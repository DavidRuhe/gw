import argparse
from configs.base import base_parser
from configs.parse import add_group

base_args, _ = base_parser.parse_known_args()

parser = argparse.ArgumentParser(parents=[base_parser])

parser.add_argument("--train", default=False, action="store_true")

dataset_cfg = dict(
    object="ToyDataset",
    name="moons",
    fold=0,
)
add_group(parser, base_args, dataset_cfg, "dataset")
