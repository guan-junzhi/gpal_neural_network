"""
Provides the registry instance for task modules, each task should have all of the below modules

configparser, evaluation, postprocess, preprocess, kpi, dataset, head, loss, logger, task
"""
from typing import Union

from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.config_parsers import BaseConfigParser
from gpal_lightning.utils import Registry

ATTRIBUTES = Registry("Attribute")
CALLBACKS = Registry("Callback")
CONFIGPARSERS = Registry("ConfigParser")
EVALUATORS = Registry("Evaluator")
POSTPROCESSES = Registry("PostProcessing")
PREPROCESSES = Registry("PreProcessing")
KPIS = Registry("Kpi")
KPI_FORMATTERS = Registry("KPIFormatter")

DATASETS = Registry("Dataset")
HEADS = Registry("Head")
LOSSES = Registry("Loss")
LOGGERS = Registry("Logger")
TASKS = Registry("Task")


def build(
    global_config: GlobalConfig,
    task_config: Union[dict, BaseConfigParser],
    registry: Registry,
    task_name: str,
    *args,
    **kwargs,
):
    """This function is used to build task module given registry

    Args:
        global_config: instance contains global parameters
        task_config: instance of task config parser
        registry: Registry module of current task
        task_name: used to indicate which task module to use
        *args: used for attribute_config
        **kwargs: some kwargs used to initial task modules

    Returns: initialized class object

    """
    cls_name = registry.name
    cls_obj = registry.get(task_name.upper() + cls_name)

    if cls_name == "Task":
        kwargs["name"] = task_name

    sub_name = kwargs.pop("sub_name", "")

    obj_name = task_name.upper() + sub_name + cls_name
    # print(f"obj_name = {obj_name} {task_config}")
    cls_obj = registry.get(obj_name)
    # print(args)
    # print(kwargs)

    return cls_obj(global_config, task_config, *args, **kwargs)


__all__ = [
    "build",
    "ATTRIBUTES",
    "CALLBACKS",
    "CONFIGPARSERS",
    "EVALUATORS",
    "POSTPROCESSES",
    "PREPROCESSES",
    "KPI_FORMATTERS",
    "KPIS",
    "DATASETS",
    "HEADS",
    "LOGGERS",
    "LOSSES",
    "TASKS",
]
