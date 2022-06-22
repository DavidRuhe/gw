import argparse
import os
import unittest

from configs.base import base_parser
from configs.parse import add_group

base_args, _ = base_parser.parse_known_args()
parser = argparse.ArgumentParser(parents=[base_parser])

sim_cfg = dict(
    object="SyntheticSimulator",
    output_path=f"/Users/druhe/Projects/gw/datasets/synthetic",
    num_events=2048,
    num_posterior_samples=1024,
)
add_group(parser, base_args, sim_cfg, "simulator")


class TestConfig(unittest.TestCase):
    def test_synthetic_sim(self):
        command = "python simulate.py -C configs/simulate/synthetic.py --simulator.output_path=../unittest_data/synthetic"
        result = os.system(command)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
