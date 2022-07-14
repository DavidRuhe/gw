import argparse
import importlib
import logging
import os
import shutil
import sys
import tempfile
from functools import partial
from typing import Iterable
from pytorch_lightning import callbacks, loggers

import yaml

from configs.parse import add_arguments, add_group, flatten, unflatten
from utils import count_parameters, set_seed

logging.basicConfig(level=logging.INFO)

USE_WANDB = (
    "WANDB_ENABLED" in os.environ and os.environ["WANDB_ENABLED"].lower() == "true"
)

if USE_WANDB:
    import wandb


class CSVLogger(loggers.CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_image(self, key, images, step=None, **kwargs) -> None:

        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')

        for i, fig in enumerate(images):
            if step is not None:
                save_file = os.path.join(self.log_dir, f"{key}_{step}_{i}.png")
            else:
                save_file = os.path.join(self.log_dir, f"{key}_{i}.png")

            fig.savefig(save_file, bbox_inches="tight", **kwargs)
            fig.clf()



class EvaluationLoop(callbacks.Callback):
    def __init__(self, evaluation_config, dataset):
        self.evaluation_config = evaluation_config
        self.dataset = dataset
        self.evaluations = {
            evaluation: object_from_config(evaluation_config, evaluation)
            for evaluation in evaluation_config
        }

    def _loop(self, trainer, model, mode):
        if trainer.current_epoch % 1 == 0:
            for k, evaluation in self.evaluations.items():
                evaluation(
                    trainer=trainer,
                    dataset=self.dataset,
                    model=model,
                    mode=mode,
                    **self.evaluation_config[k],
                )

    def on_validation_epoch_end(self, trainer, model):
        self._loop(trainer, model, mode="val")

    def on_test_epoch_end(self, trainer, model):
        self._loop(trainer, model, mode="test")

    def on_train_epoch_end(self, trainer, model):
        self._loop(trainer, model, mode="train")


class SBSchedule(callbacks.Callback):
    def __init__(self, sb_schedule):
        assert (
            len(sb_schedule) % 2 == 0
        ), "SB schedule must be even with epoch, value pairs."
        self.schedule = {
            sb_schedule[i]: sb_schedule[i + 1] for i in range(0, len(sb_schedule), 2)
        }

    def on_epoch_start(self, trainer, model):
        if trainer.current_epoch in self.schedule:
            model.sb_weight = self.schedule[trainer.current_epoch]
            print("\nUpdating SB weight to", model.sb_weight, "\n")
        return super().on_epoch_start(trainer, model)


def main(config, experiment):

    dataset = object_from_config(config, "dataset")(**config["dataset"])

    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    run_validation = val_loader is not None
    run_test = test_loader is not None

    ckpt = config["model"].pop("ckpt")
    flows = object_from_config(config, "flow")(**config.pop("flow"))
    model_object = object_from_config(config, "model")

    if ckpt is not None:
        model = model_object.load_from_checkpoint(
            ckpt, flows=flows, d=dataset.dimensionality
        )
    else:
        model = model_object(
            dataset=dataset,
            flows=flows,
            **config.pop("model"),
        )

    print(f"Parameters: {count_parameters(model)}")

    callback_chain = []
    sb_schedule = config["trainer"].pop("sb_schedule")
    if sb_schedule is not None:
        callback_chain.append(SBSchedule(sb_schedule))

    if "evaluation" in config:
        evaluate = EvaluationLoop(
            evaluation_config=config["evaluation"],
            dataset=dataset,
        )

        callback_chain.append(evaluate)

    if run_validation > 0:
        monitor = "val_loss"
    else:
        monitor = "train_loss"

    checkpoint = callbacks.ModelCheckpoint(
        monitor=monitor,
        mode="min",
        dirpath=config["dir"],
        **config["trainer"].pop("checkpoint"),
    )
    callback_chain.append(checkpoint)
    earlystop = callbacks.EarlyStopping(
        monitor=monitor, **config["trainer"].pop("earlystopping")
    )
    callback_chain.append(earlystop)
    from pytorch_lightning.callbacks import ProgressBar

    class MeterlessProgressBar(ProgressBar):
        def _update_bar(self, bar):
            bar.dynamic_ncols = False
            bar.ncols = 0
            return bar

        def init_train_tqdm(self):
            bar = self._update_bar(super().init_train_tqdm())
            return bar

        def init_validation_tqdm(self):
            bar = self._update_bar(super().init_validation_tqdm())
            return bar

        def init_sanity_tqdm(self):
            bar = self._update_bar(super().init_sanity_tqdm())
            return bar

        def init_predict_tqdm(self):
            bar = self._update_bar(super().init_predict_tqdm())
            return bar

        def init_test_tqdm(self):
            bar = self._update_bar(super().init_test_tqdm())
            return bar

    bar = MeterlessProgressBar(refresh_rate=1)
    callback_chain.append(bar)

    if USE_WANDB:
        logger = loggers.WandbLogger(experiment=experiment, dir=config["dir"])
    else:
        logger = CSVLogger(config["dir"])

    trainer = object_from_config(config, "trainer")(
        **config.pop("trainer"),
        # callbacks=[checkpoint, earlystop, evaluate, bar],
        callbacks=callback_chain,
        logger=logger,
        deterministic=True,
        # enable_progress_bar=False,
    )

    if config["train"]:

        if run_validation:
            trainer.fit(model, train_loader, val_loader)
            (result,) = trainer.validate(model, val_loader, ckpt_path="best")
            if "val_loss" in result:
                assert result["val_loss"] == checkpoint.best_model_score.item()
        else:
            trainer.fit(model, train_loader)
            result = {}
        if run_test:
            if len(test_loader) > 0:
                (result,) = trainer.test(model, test_loader, ckpt_path="best")
    else:
        if run_test:
            (result,) = trainer.test(model, test_loader)
        else:
            result = {}

    return result


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


def command_from_config(config):
    cmd = f"python {__file__} "
    for key, value in config.items():
        cmd += f"--{key}={value} "
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--experiment.name", type=str, default=None)
    parser.add_argument("--experiment.group", type=str, default=None)
    (run_config_path,) = pop_configs_from_sys_argv("-C", default=["configs/base.yml"])
    run_config = load_yaml(run_config_path)
    add_arguments(parser, run_config)

    (data_config_path,) = pop_configs_from_sys_argv("-D")
    data_config = load_yaml(data_config_path)
    run_args = run_config["dataset"] if "dataset" in run_config else {}
    add_group(parser, run_args, data_config, "dataset")

    (loader_config_path,) = pop_configs_from_sys_argv(
        "-L", default=["configs/loaders/loader.yml"]
    )
    run_args = run_config["loader"] if "loader" in run_config else {}
    loader_config = load_yaml(loader_config_path)
    add_group(parser, run_args, loader_config, "loader")

    (model_config_path,) = pop_configs_from_sys_argv(
        "-M", default=["configs/models/flow_marginal.yml"]
    )
    run_args = run_config["model"] if "model" in run_config else {}
    model_config = load_yaml(model_config_path)
    add_group(parser, run_args, model_config, "model")

    (flow_config_path,) = pop_configs_from_sys_argv("-F")
    run_args = run_config["flow"] if "flow" in run_config else {}
    flow_config = load_yaml(flow_config_path)
    add_group(parser, run_args, flow_config, "flow")

    (trainer_config_path,) = pop_configs_from_sys_argv(
        "-T", default=["configs/trainers/trainer.yml"]
    )
    run_args = run_config["trainer"] if "trainer" in run_config else {}
    trainer_config = load_yaml(trainer_config_path)
    add_group(parser, run_args, trainer_config, "trainer")

    evaluation_config_paths = pop_configs_from_sys_argv("-E", default=[])

    for path in evaluation_config_paths:
        name, ext = os.path.splitext(os.path.basename(path))
        evaluation_config = load_yaml(path)
        add_group(parser, run_args, evaluation_config, f"evaluation.{name}")

    args = parser.parse_args()
    config = unflatten(vars(args))
    print("\n", yaml.dump(config, default_flow_style=False), "\n")
    set_seed(config["seed"])

    exception = None
    with tempfile.TemporaryDirectory() as tmpdir:
        config["dir"] = tmpdir
        config["command"] = command_from_config(vars(args))

        if USE_WANDB:
            assert (
                config["experiment"]["name"] is not None
            ), 'Please provide "experiment.name"'
            experiment = wandb.init(config=args, dir=tmpdir, **config["experiment"])
        else:
            experiment = None
        try:
            result = main(config, experiment)
            if result is not None:
                config = {**config, **result}
        except Exception as e:
            exception = e

        # Save config
        with open(os.path.join(tmpdir, "config.yml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        if USE_WANDB:
            wandb.save(os.path.join(tmpdir, "*.ckpt"))
            wandb.finish()
        else:
            if not isinstance(exception, KeyboardInterrupt):
                target = shutil.move(tmpdir, "../local_runs/")
                logging.info(f"Copied results to {target}")

    if exception is None:
        logging.info("Run finished!")
    else:
        raise (exception)
