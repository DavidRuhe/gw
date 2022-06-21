import os
import unittest
import argparse
from configs.base import base_parser
from configs.parse import add_group

base_args, _ = base_parser.parse_known_args()

parser = argparse.ArgumentParser(parents=[base_parser])

parser.add_argument("--train", default=False, action="store_true")
parser.add_argument('--batch_size', default=32, type=int)

dataset_cfg = dict(
    object="ToyDataset",
    name="moons",
    fold=0,
)
add_group(parser, base_args, dataset_cfg, "dataset")


class TestConfig(unittest.TestCase):
    def test_toy_run(self):
        command = "WANDB_DISABLED=TRUE python toy.py -C configs/toy/cfg_toy.py" 
        result = os.system(command)
        self.assertEqual(result, 0)