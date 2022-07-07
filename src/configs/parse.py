import importlib
from types import SimpleNamespace
import sys
import argparse
import random
from copy import copy
import ast
from utils import set_seed


import utils
import functools


def flatten(d, parent_key="", sep="."):
    items = []
    if isinstance(d, SimpleNamespace):

        d = d.__dict__
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) or isinstance(v, SimpleNamespace):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten(d: dict, sep="."):
    assert isinstance(d, dict)
    result = dict()
    for key, value in d.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        assert isinstance(obj, SimpleNamespace)
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    attr = rgetattr(obj, pre)
    assert isinstance(attr, SimpleNamespace)
    return setattr(attr if pre else obj, post, val)


def parse_arg_default(default_type):
    def parse_arg(arg):
        # check if list
        if str(arg)[0] == "[" and str(arg)[-1] == "]":
            return ast.literal_eval(arg)
        elif default_type == type(None):
            return arg
        else:
            return default_type(arg)

    return parse_arg


def get_config_from_sys_argv():
    argv = sys.argv
    if "--config" in argv:
        config_file = argv[argv.index("--config") + 1]
        index = argv.index("--config")
    elif "-C" in argv:
        config_file = argv[argv.index("-C") + 1]
        index = argv.index("-C")
    else:
        raise Exception("No config file specified (use -C or --config).")

    sys.argv = sys.argv[:index] + sys.argv[index + 2 :]
    return config_file


def parse_cfg():
    config_file = get_config_from_sys_argv()

    cfg = copy(
        importlib.import_module(config_file.replace("/", ".").replace(".py", "")).cfg
    )

    # This enables all arguments to be shown in the help menu.
    parser = argparse.ArgumentParser()
    all_args = flatten(cfg)

    for k, v in all_args.items():
        if k in all_args:
            parser.add_argument("--" + k, type=parse_arg_default(type(v)), default=v)

    args = parser.parse_args()

    for k, v in vars(args).items():
        if v != rgetattr(cfg, k):
            print(f"Updating {k} to {v}")
            rsetattr(cfg, k, v)

    if args.seed > -1:
        cfg.seed = args.seed

    if cfg.seed < 0:
        cfg.seed = random.randint(0, sys.maxsize)

    print("Seed: ", cfg.seed)

    utils.set_seed(cfg.seed)
    return cfg


def add_arguments(parser, cfg):
    for k, v in cfg.items():
        parser.add_argument(f"--" + k, type=parse_arg_default(type(v)), default=v)


def add_group(parser, base_args, cfg, group_name):

    group = parser.add_argument_group(group_name)
    for k, v in cfg.items():
        # help = None
        if k in base_args:
            # help = argparse.SUPPRESS
            default = base_args[k]
        else:
            default = v

        help = f"Default: {v}"

        group.add_argument(
            f"--{group_name}." + k,
            type=parse_arg_default(type(v)),
            default=default,
            help=help,
            # required=v is None
        )


def parse_args():
    config_file = get_config_from_sys_argv()
    try:
        parser = copy(
            importlib.import_module(
                config_file.replace("/", ".").replace(".py", "")
            ).parser
        )
    except ModuleNotFoundError:
        raise Exception(f"File {config_file} not found.")
    except AttributeError:
        raise Exception(f"Cannot access 'parser' attribute of {config_file}.")

    args = parser.parse_args()
    args = unflatten(vars(args))
    set_seed(args['seed'])
    return args
