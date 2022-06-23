# import os
# import unittest


# def msg(cmd):
#     return f"\n\nFailed command:\n{cmd}"


# class TestTrainConfigs(unittest.TestCase):
#     def test_train_m1_spline_coupling(self):
#         cmd = "python train.py -D configs/data/m1.yml -M configs/models/spline_coupling.yml --train --trainer.max_epochs=1"
#         result = os.system(cmd)
#         self.assertEqual(result, 0, msg=msg(cmd))


#     def test_train_toy_spline_coupling(self):
#         cmd = "python train.py -D configs/data/toy.yml -M configs/models/spline_coupling.yml --train --trainer.max_epochs=1"
#         result = os.system(cmd)
#         self.assertEqual(result, 0, msg=msg(cmd))
