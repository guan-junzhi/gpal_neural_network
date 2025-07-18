import logging

import yaml
from torch import distributed

from gpal_lightning import const
from gpal_lightning.data.dataloader_helpers.datasets_properties import set_datasets_properties
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks import builder
from gpal_lightning.neural_network.tasks.register_tasks import register_tasks


def build_tasks(global_config: GlobalConfig, phase: str = const.PHASE_TRAINING, tasks_root: str = "") -> list:
    """Helper function to build task for train/eval/inference

    Args:
        global_config: global config instance
        phase: used to determine whether to init datasets/heads and loggers
        tasks_root: path contains the source code of all tasks
    Returns:
        list of built tasks
    """
    tasks = global_config.Tasks
    built_tasks = []
    for task_name, task_config in tasks.items():
        print(task_name, task_config)
        print(tasks_root)
        if tasks_root:
            try:
                register_tasks(tasks_root, task_name.lower())
            except ModuleNotFoundError as error:
                logging.warning(
                    f"task: {task_name} registration failed due to {error}")
        logging.info(f"Building task {task_name}")
        task = builder.build(global_config, task_config,
                             builder.TASKS, task_name)

        if phase in [const.PHASE_TRAINING, const.PHASE_VALIDATION, const.PHASE_INFERENCE]:
            task.build_head()
            task.build_logger()

        built_tasks.append(task)

    return built_tasks


def build_tasks_datasets(global_config: GlobalConfig, built_tasks: list, phase: str = const.PHASE_TRAINING):

    for task in built_tasks:
        if should_build_datasets(global_config, task.name):
            task.build_datasets(phase)
    set_datasets_properties(global_config, built_tasks)


def should_build_datasets(global_config: GlobalConfig, task_name: str):
    if global_config.tasks and task_name not in global_config.tasks:
        return False
    return True
