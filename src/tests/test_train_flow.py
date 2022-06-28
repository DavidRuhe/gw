import os
import unittest


def msg(cmd):
    return f"\n\nFailed command:\n{cmd}"


class TestTrainConfigs(unittest.TestCase):
    def test_train_circles_spline_coupling(self):
        cmd = (
            "python /Users/druhe/Projects/gw/src/train_flow.py --train=False --ckpt=None"
            "--seed=42 --dataset.object=data.toy.ToyDataset --dataset.name=circles --loader.object=torch.utils.data.DataLoader"
            "--loader.batch_size=1024 --flow.object=pyro.distributions.transforms.spline_coupling"
            "--flow.input_dim=2  --model.lr=0.01 --trainer.object=pytorch_lightning.Trainer --trainer.max_epochs=5001"
            "--evaluation.flow_heatmap_2d.object=evaluation.flow_heatmap_2d.flow_heatmap_2d --evaluation.flow_marginals.object=evaluation.flow_marginals.flow_marginals"
        )
        result = os.system(cmd)
        self.assertEqual(result, 0, msg=msg(cmd))

    def test_train_circles_realnvp(self):
        cmd = (
            "python /Users/druhe/Projects/gw/src/train_flow.py --train=True --ckpt=None"
            "--seed=42 --dataset.object=data.toy.ToyDataset --dataset.name=circles --loader.object=torch.utils.data.DataLoader"
            "--loader.batch_size=1024 --flow.object=models.realnvp.realnvp --flow.num_transforms=4 --model.lr=0.01"
            "--flow.input_dim=2 --flow.hidden_dims=[16,16] --trainer.object=pytorch_lightning.Trainer --trainer.max_epochs=1"
            "--evaluation.flow_heatmap_2d.object=evaluation.flow_heatmap_2d.flow_heatmap_2d --evaluation.flow_marginals.object=evaluation.flow_marginals.flow_marginals "
        )
        result = os.system(cmd)
        self.assertEqual(result, 0, msg=msg(cmd))

    def test_train_m1_spline_coupling(self):
        cmd = (
            "python /Users/druhe/Projects/gw/src/train_flow.py --train=True --ckpt=None"
            " --seed=42 --dataset.object=data.m1.M1Dataset --dataset.path=/Users/druhe/Projects/gw/MLPopulation/data/Combined_GWTC_m1m2chieffz.npz"
            " --loader.object=torch.utils.data.DataLoader --loader.batch_size=1024 --model.object=models.flow_marginal.NormalizingFlow"
            " --model.lr=0.0005 --flow.object=pyro.distributions.transforms.spline_coupling --flow.input_dim=2"
            " --trainer.object=pytorch_lightning.Trainer --trainer.max_epochs=2048 --evaluation.flow_marginal_kdeplots.object=evaluation.flow_marginal_kdeplots.marginal_kdeplots"
            " --evaluation.flow_marginals.object=evaluation.flow_marginals.flow_marginals"
        )
        result = os.system(cmd)
        self.assertEqual(result, 0, msg=msg(cmd))
