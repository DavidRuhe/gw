import os
import unittest
import argparse
from configs.base import base_parser
from configs.parse import add_group

base_args, _ = base_parser.parse_known_args()
parser = argparse.ArgumentParser(parents=[base_parser])

sim_cfg = dict(
    object="GaussianSimulator",
    output_path=f"/Users/druhe/Projects/gw/data/gaussian.npy",
)
add_group(parser, base_args, sim_cfg, "simulator")


class TestConfig(unittest.TestCase):
    def test_gaussian_sim(self):
        command = "python simulate.py -C configs/simulate/gaussian.py --simulator.output_path=../unittest_data/gaussian.npy"
        result = os.system(command)
        self.assertEqual(result, 0)



if __name__ == "__main__":
    unittest.main()