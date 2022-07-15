import os
import unittest


def msg(cmd):
    return f"\n\nFailed command:\n{cmd}"


class TestTrainConfigs(unittest.TestCase):
    def test_train_moons_realnvp(self):
        cmd = (
            "python train_marginal.py"
            " -D configs/data/toy.yml"
            " -M configs/models/realnvp.yml"
            " -E configs/evaluation/flow_heatmap_2d.yml configs/evaluation/flow_marginals.yml"
            " --trainer.max_epochs=1 --train"
        )
        result = os.system(cmd)
        self.assertEqual(result, 0, msg=msg(cmd))
