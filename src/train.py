import torch
from functools import partial
import sys
import os
import argparse
from copy import copy
import importlib
from configs.parse import add_arguments, add_group, unflatten
from utils import set_seed
import yaml
import logging
import shutil
import logging
import matplotlib.pyplot as plt
from pytorch_lightning import callbacks, loggers
import pandas as pd
import seaborn as sns

from torch import nn
import pytorch_lightning as pl
import pyro.distributions as dist
import pyro.distributions.transforms as T

logging.basicConfig(level=logging.INFO)
from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from pyro.distributions.transforms import NeuralAutoregressive


import tempfile

USE_WANDB = (
    "WANDB_ENABLED" in os.environ and os.environ["WANDB_ENABLED"].lower() == "true"
)

if USE_WANDB:
    import wandb


def load_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)


def load_object(object):
    module, object = object.rsplit(".", 1)
    module = importlib.import_module(module)
    object = getattr(module, object)
    return object


def object_from_config(config, key):
    config = config[key]
    object = load_object(config.pop("object"))
    return object


def evaluate(dir, model, test_dataset):
    d = test_dataset.dimensionality
    grid = torch.linspace(-3, 3, 64)
    if d == 2:
        X_flow = model.sample(1024).detach().numpy()
        plt.title(r"Joint Distribution")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        # plt.scatter(X[:, 0], X[:, 1], label="data", alpha=0.5)
        plt.scatter(
            X_flow[:, 0], X_flow[:, 1], color="firebrick", label="flow", alpha=0.5
        )
        plt.legend()
        plt.savefig("test.png")
        plt.close()

        X = torch.meshgrid(grid, grid, indexing="xy")
        prob = model.log_prob(torch.stack(X, -1).view(-1, 2)).exp()
        X = torch.stack(X, -1).view(-1, 2)
        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "z": prob})
        data = df.pivot(index="x", columns="y", values="z")
        sns.heatmap(data)

        # sns.kdeplot(data=df, x="x", y="y")
        # plt.pcolormesh(*X, prob.numpy())
        plt.savefig(os.path.join(dir, "pcolormesh.png"))

    else:
        X = grid[:, None]
        prob = model.log_prob(X).exp()
        plt.plot(prob.numpy())
        plt.title("$p(\\theta \mid w)$")
        plt.savefig(os.path.join(dir, "prob.png"))
        plt.close()


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask):
        super().__init__()
        self.s_fc1 = nn.Linear(input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, output_dim)
        self.t_fc1 = nn.Linear(input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, output_dim)
        self.mask = mask

    def forward(self, x):
        x_m = x * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
        y = x_m + (1 - self.mask) * (x * torch.exp(s_out) + t_out)
        log_det_jacobian = s_out.sum(dim=1)
        return y, log_det_jacobian

    def backward(self, y):
        y_m = y * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
        x = y_m + (1 - self.mask) * (y - t_out) * torch.exp(-s_out)
        return x


class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers=6):
        super().__init__()
        assert n_layers >= 2, "num of coupling layers should be greater or equal to 2"

        self.modules = []
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        for _ in range(n_layers - 2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
        self.module_list = nn.ModuleList(self.modules)

    def forward(self, x):
        ldj_sum = 0  # sum of log determinant of jacobian
        for module in self.module_list:
            x, ldj = module(x)
            ldj_sum += ldj
        return x, ldj_sum

    def backward(self, z):
        for module in reversed(self.module_list):
            z = module.backward(z)
        return z


# class Model(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         d = 1
#         base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
#         self.spline_transform = T.spline_coupling(d, count_bins=16)
#         self.coupling = T.affine_coupling(d, hidden_dims=(10, 10))
#         # self.transform = nn.ModuleList(
#         #     [T.affine_coupling(2, (64, 64)) for _ in range(3)]
#         # )
#         # self.transform = nn.ModuleList([T.neural_autoregressive(2)])
#         # arn = AutoRegressiveNN(2, [40], param_dims=[16] * 3)
#         # self.transform = NeuralAutoregressive(arn, hidden_units=16)
#         # # self.transform = nn.ModuleList([T.block_autoregressive(2)])
#         self.flow = dist.TransformedDistribution(base_dist, [self.spline_transform, self.coupling])
#         # mask = torch.tensor([0, 1])
#         # self.model = RealNVP(2, 2, 256, mask=torch.ones(2))
#         # from torch import distributions
#         # self.prior_z = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))


#     def step(self, batch, batch_idx):
#         # self.flow.clear_cache()
#         (x,) = batch
#         log_prob = self.flow.log_prob(x)
#         loss = -log_prob.mean()

#         self.log("nll", loss, prog_bar=True)

#         return loss

#     def training_step(self, batch, batch_idx):
#         return self.step(batch, batch_idx)

#     def validation_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters())


def main(config):
    set_seed(config["seed"])
    dataset = partial(object_from_config(config, "dataset"), **config.pop("dataset"))
    train_dataset = dataset(split="train")
    valid_dataset = dataset(split="valid")
    test_dataset = dataset(split="test")

    set_seed(config["seed"])
    loader = partial(object_from_config(config, "loader"), **config.pop("loader"))
    train_loader = loader(train_dataset, shuffle=True)
    valid_loader = loader(valid_dataset, shuffle=False)
    test_loader = loader(test_dataset, shuffle=False)

    set_seed(config["seed"])
    ckpt = config.pop("ckpt")
    if ckpt is not None:
        model_object = object_from_config(config, "model")
        model = model_object.load_from_checkpoint(ckpt)
    else:
        model = object_from_config(config, "model")(
            **config.pop("model"),
            input_dim=train_dataset.dimensionality,
            hierarchical=train_dataset.hierarchical,
        )
    # model = Model()

    set_seed(config["seed"])
    trainer = object_from_config(config, "trainer")(
        **config.pop("trainer"),
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", mode="min")],
        logger=loggers.CSVLogger(config["dir"]),
        deterministic=True,
    )

    set_seed(config["seed"])
    logging.info("Sanity checking evaluation.")
    with torch.no_grad():
        evaluate(config["dir"], model, test_dataset)

    set_seed(config["seed"])
    if config["train"]:
        trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    set_seed(config["seed"])
    with torch.no_grad():
        evaluate(config["dir"], model, test_dataset)


def load_config(config_file, attribute="cfg"):
    try:
        module = importlib.import_module(
            config_file.replace("/", ".").replace(".py", "")
        )

        cfg = copy(getattr(module, attribute))
    except ModuleNotFoundError:
        raise Exception(f"File {config_file} not found.")
    except AttributeError:
        raise Exception(f"Cannot access 'cfg' attribute of {config_file}.")
    return cfg


def pop_config_from_sys_argv(key, default=None):
    argv = sys.argv
    if key in argv:
        config_file = argv[argv.index(key) + 1]
        index = argv.index(key)
        sys.argv = sys.argv[:index] + sys.argv[index + 2 :]
    elif default is not None:
        config_file = default
    else:
        raise Exception(f'No "{key}" config file specified.')
    return config_file


if __name__ == "__main__":

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--train", action="store_true", default=False)
    base_parser.add_argument("--ckpt", type=str, default=None)
    base_config = load_yaml("configs/base.yml")
    add_arguments(base_parser, base_config)
    base_args, _ = base_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[base_parser])

    data_config_path = pop_config_from_sys_argv("-D")
    data_config = load_yaml(data_config_path)
    add_group(parser, base_args, data_config, "dataset")

    loader_config_path = pop_config_from_sys_argv(
        "-L", default="configs/loaders/loader.yml"
    )
    loader_config = load_yaml(loader_config_path)
    add_group(parser, base_args, loader_config, "loader")

    model_config_path = pop_config_from_sys_argv("-M")
    model_config = load_yaml(model_config_path)
    add_group(parser, base_args, model_config, "model")

    trainer_config_path = pop_config_from_sys_argv(
        "-T", default="configs/trainers/trainer.yml"
    )
    trainer_config = load_yaml(trainer_config_path)
    add_group(parser, base_args, trainer_config, "trainer")

    args = parser.parse_args()
    config = unflatten(vars(args))
    set_seed(config["seed"])

    exception = None
    with tempfile.TemporaryDirectory() as tmpdir:
        config["dir"] = tmpdir
        if USE_WANDB:
            wandb.init(config=args)
        try:
            main(config)
        except (Exception, KeyboardInterrupt) as e:
            exception = e

        if USE_WANDB:
            wandb.finish()
            if os.path.exists(tmpdir):
                os.system(
                    f"wandb sync {tmpdir} --clean --clean-old-hours 0 --clean-force"
                )
        else:
            if not isinstance(exception, KeyboardInterrupt):
                target = shutil.move(tmpdir, "../local_runs/")
                logging.info(f"Copied results to {target}")

    if exception is None:
        logging.info("Run finished!")
    else:
        raise (exception)
