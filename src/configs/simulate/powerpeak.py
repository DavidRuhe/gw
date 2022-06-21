import argparse
import os
import unittest

from configs.base import base_parser
from configs.parse import add_group

base_args, _ = base_parser.parse_known_args()
parser = argparse.ArgumentParser(parents=[base_parser])

sim_cfg = dict(
    object="PowerPlusPeakSimulator",
    output_path=f"/Users/druhe/Projects/gw/datasets/powerpeak",
    num_events=1025,
    num_posterior_samples=1024,
)

add_group(parser, base_args, sim_cfg, "simulator")

trainer_cfg = dict(max_epochs=float("inf"))
add_group(parser, base_args, trainer_cfg, "trainer")


class TestConfig(unittest.TestCase):
    def test_powerpluspeak_sim(self):
        command = "python simulate.py -C configs/simulate/powerpeak.py --simulator.output_path=../unittest_data/powerpeak --trainer.max_epochs=1"
        result = os.system(command)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
