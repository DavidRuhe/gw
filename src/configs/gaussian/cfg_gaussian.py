import os
import unittest
import argparse
from configs.base import base_parser
from configs.parse import add_group

base_args, _ = base_parser.parse_known_args()

parser = argparse.ArgumentParser(parents=[base_parser])
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--train", default=False, action="store_true")

data_path = os.path.join("../datasets/", "gaussian")
dataset_cfg = dict(
    object="GaussianDataset",
    path=data_path,
    fold=0,
)
add_group(parser, base_args, dataset_cfg, "dataset")

trainer_cfg = dict(max_epochs=float("inf"))
add_group(parser, base_args, trainer_cfg, "trainer")


class TestConfig(unittest.TestCase):
    def test_run(self):
        command = f"WANDB_DISABLED=TRUE python gaussian.py -C {os.path.relpath(__file__)} --dataset.path=../unittest_data/gaussian/ --trainer.max_epochs=1 --train"
        result = os.system(command)
        self.assertEqual(result, 0)
