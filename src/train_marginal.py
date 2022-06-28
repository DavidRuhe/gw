import argparse
import importlib
import logging
import os
import shutil
import sys
import tempfile
from functools import partial
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

    dataset = partial(object_from_config(config, "dataset"), **config.pop("dataset"), hierarchical=False)
    train_dataset = dataset(split="train")
    valid_dataset = dataset(split="valid")
    test_dataset = dataset(split="test")

    loader = partial(object_from_config(config, "loader"), **config.pop("loader"))
    train_loader = loader(train_dataset, shuffle=True)
    valid_loader = loader(valid_dataset, shuffle=False)
    test_loader = loader(test_dataset, shuffle=False)

    ckpt = config.pop("ckpt")
    model_object = object_from_config(config, "model")
    if ckpt is not None:
        model = model_object.load_from_checkpoint(ckpt)
    else:
        model = model_object(
            **config.pop("model"),
            input_dim=train_dataset.dimensionality,
        )

    checkpoint = callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", dirpath=config["dir"]
    )

    trainer = object_from_config(config, "trainer")(
        **config.pop("trainer"),
        callbacks=[checkpoint],
        # logger=loggers.CSVLogger(config["dir"]),
        deterministic=True,
    )

    if config["train"]:
        trainer.fit(model, train_loader, valid_loader)
        trainer.test(model, test_loader, ckpt_path='best')
    else:
        trainer.test(model, test_loader)

    # Evaluation
    if "evaluation" in config:
        evaluation_config = config["evaluation"]
        for evaluation in config["evaluation"]:
            object_from_config(evaluation_config, evaluation)(
                dir=config["dir"],
                dataset=test_dataset,
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


def pop_configs_from_sys_argv(key, default=None):
    argv = sys.argv
    if key in argv:
        config_files = []
        index = argv.index(key)
        start_index = index
        while not argv[index + 1].startswith("-"):
            config_files.append(argv[index + 1])
            index += 1
            if index == len(argv) - 1:
                break
        sys.argv = sys.argv[:start_index] + sys.argv[index + 1 :]

    elif default is not None:
        config_files = default
    else:
        raise Exception(f'No "{key}" config file specified.')
    return config_files


if __name__ == "__main__":

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--train", action="store_true", default=False)
    base_parser.add_argument("--ckpt", type=str, default=None)
    base_config = load_yaml("configs/base.yml")
    add_arguments(base_parser, base_config)
    base_args, _ = base_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[base_parser])

    (data_config_path,) = pop_configs_from_sys_argv("-D")
    data_config = load_yaml(data_config_path)
    add_group(parser, base_args, data_config, "dataset")

    (loader_config_path,) = pop_configs_from_sys_argv(
        "-L", default=["configs/loaders/loader.yml"]
    )
    loader_config = load_yaml(loader_config_path)
    add_group(parser, base_args, loader_config, "loader")

    (model_config_path,) = pop_configs_from_sys_argv("-M")
    model_config = load_yaml(model_config_path)
    add_group(parser, base_args, model_config, "model")

    (trainer_config_path,) = pop_configs_from_sys_argv(
        "-T", default=["configs/trainers/trainer.yml"]
    )
    trainer_config = load_yaml(trainer_config_path)
    add_group(parser, base_args, trainer_config, "trainer")

    evaluation_config_paths = pop_configs_from_sys_argv("-E", default=[])

    for path in evaluation_config_paths:
        name, ext = os.path.splitext(os.path.basename(path))
        evaluation_config = load_yaml(path)
        add_group(parser, base_args, evaluation_config, f"evaluation.{name}")

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