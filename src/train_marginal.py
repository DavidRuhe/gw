import argparse
import importlib
import logging
import os
import shutil
import sys
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from pytorch_lightning import callbacks, loggers

from configs.parse import add_arguments, add_group, unflatten
from utils import set_seed

logging.basicConfig(level=logging.INFO)

USE_WANDB = (
    "WANDB_ENABLED" in os.environ and os.environ["WANDB_ENABLED"].lower() == "true"
)

if USE_WANDB:
    import wandb


def main(config):

    set_seed(config["seed"])
    dataset = partial(object_from_config(config, "dataset"), **config.pop("dataset"))
    train_dataset = dataset(split="train")
    valid_dataset = dataset(split="valid")
    test_dataset = dataset(split="test")

    set_seed(config["seed"])
    loader = partial(object_from_config(config, "loader"), **config.pop("loader"))
    train_loader = loader(train_dataset, shuffle=True, drop_last=True)
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
        )

    trainer = object_from_config(config, "trainer")(
        **config.pop("trainer"),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=128)
        ],
        logger=loggers.CSVLogger(config["dir"]),
        deterministic=True,
    )

    set_seed(config["seed"])
    if config["train"]:
        trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    # Evaluation
    if "evaluation" in config:
        evaluation_config = config["evaluation"]
        for evaluation in config["evaluation"]:
            object_from_config(evaluation_config, evaluation)(
                dir=config['dir'],
                dataloader=test_loader,
                model=model,
                **evaluation_config[evaluation],
            )


def load_object(object):
    module, object = object.rsplit(".", 1)
    module = importlib.import_module(module)
    object = getattr(module, object)
    return object


def object_from_config(config, key):
    config = config[key]
    object = load_object(config.pop("object"))
    return object


def load_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)


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

    evaluation_config_paths = pop_config_from_sys_argv("-E", default="")
    evaluation_config_paths = evaluation_config_paths.split(",")

    for path in evaluation_config_paths:
        if len(path) == 0:
            continue
        name, ext = os.path.splitext(os.path.basename(path))
        evaluation_config = load_yaml(path)
        add_group(parser, base_args, evaluation_config, f"evaluation.{name}")

    args = parser.parse_args()
    config = unflatten(vars(args))
    set_seed(config["seed"])

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
