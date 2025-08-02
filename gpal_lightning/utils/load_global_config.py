import argparse
import copy
import logging
import os

import yaml

from gpal_lightning import const
# from zpilot_lightning.data.exceptions import DatasetException
from gpal_lightning.neural_network.global_config import GlobalConfig


def get_checkpoint_path(root):
    checkpoint = os.path.join(
        root, const.CHECKPOINT_PATH, const.CHECKPOINT_NAME_LAST + const.FILE_EXTENSION)
    return checkpoint


# def get_config_path(root):
#     config = os.path.join(root, const.CONFIG_NAME + const.CONFIG_EXTENSION)
#     return config


def load_global_config(args: argparse.Namespace, override: bool = True, dump_config: bool = True) -> GlobalConfig:
    """
    This helper is used in model training/inference and evaluation pipeline to init global config
    Args:
        args: args from users
        override: used in model training, if an arg is given by user, is should override the
        default value in config.yaml
        dump_config: whether dump config to save path or not

    Returns: global_config instance

    """

    with open(args.config, "r") as infile:
        config = yaml.safe_load(infile)

    # config = add_freeze_config(config, args)
    config = add_load_config(config, args)
    # print("config.save", config['save'])

    updated_config = copy.deepcopy(config)
    # updated_config["Global"] = dict(config["Global"].items())
    # print(args)
    for key, val in vars(args).items():
        if key in ['tasks']:
            updated_config[key] = [name.upper() for name in val]
            continue
        if key not in ["config", "load_from"]:
            updated_config[key] = val
    # print("updated_config.save", updated_config['save'])

    # print("global_config.save", updated_config.save)
    global_config = GlobalConfig(updated_config)
    if override:
        config = updated_config

    if dump_config and const.is_rank_zero():
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, const.CONFIG_NAME + const.CONFIG_EXTENSION), "w") as outfile:
            print(outfile)
            yaml.dump(config, outfile, default_flow_style=None, sort_keys=False)

    global_config.dump_path = global_config.save
    return global_config


def add_freeze_config(config, args):

    if args.freeze:
        assert "Backbone" in config or "Backbones" in config
        backbone_configs = config.get(
            "Backbone", config.get("Backbones", None))
        # BFS to find the "type" key
        stack = [backbone_configs]
        while stack:
            configs = stack.pop()
            if isinstance(configs, dict):
                if "layers_config" in configs or "type" in configs:
                    configs["freeze"] = "all"
                else:
                    for sub_config in configs.values():
                        stack.append(sub_config)
    return config


def add_load_config(config, args):
    """Warnings: if you --save path already has a checkpoint, trainer will go to resume_training mode,
    this behavior is for preemption resuming purpose"""
    config.setdefault("Global", {})["load_from"] = None
    config["Global"]["to_resume"] = False
    if get_checkpoint_path(args.save) and os.path.isfile(get_checkpoint_path(args.save)):
        config["Global"]["load_from"] = args.save
        config["Global"]["to_resume"] = True
    elif hasattr(args, "load_from"):
        config["Global"]["load_from"] = args.load_from
        config["Global"]["to_resume"] = False
    elif hasattr(args, "resume_from"):
        config["Global"]["load_from"] = args.resume_from
        config["Global"]["to_resume"] = True
    return config
