import logging

from typing import Callable, List
from torch.utils.data import DataLoader

from gpal_lightning import const
from gpal_lightning.data.dataloader_helpers.gpal_collate import gpal_collate
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.build_task import build_tasks_datasets
# from gpal_lightning.data.dataloader_helpers.sampler.stream_data_sampler import StreamDataSampler


def build_val_or_infer_dataloaders(tasks: dict, phase: str, image_per_gpu: int, collate_fn: Callable = gpal_collate):
    """Build a list of dataloaders corresponding to the tasks. This function only support val or infer datasets.

    Args:
        tasks: dict of (name, Task).
        phase: str, needs to be either `const.PHASE_VALIDATION` or `const.PHASE_INFERENCE`
        image_per_gpu: int, batch size per GPU.

    Returns:
        dataloaders: list
        datalaoder_start_idx: dict, {task_name: start_index}
    """
    dataloaders = []
    dataloader_start_idx = {}
    for task_name, task in tasks.items():
        dataloader_start_idx[task_name] = len(dataloaders)
        if phase == const.PHASE_VALIDATION:
            datasets = task.val_datasets
        elif phase == const.PHASE_INFERENCE:
            datasets = task.infer_datasets
        else:
            raise ValueError(
                f"phase {phase} is not in ({const.PHASE_VALIDATION}, {const.PHASE_INFERENCE})")
        for dataset in datasets:
            if len(dataset) == 0:
                logging.warning(f"No data find in {task_name}, skipped")
                continue
            if hasattr(task.task_config, "batch_size"):
                image_per_gpu = task.task_config.batch_size
            if hasattr(task.task_config, "test_batch_size"):
                image_per_gpu = task.task_config.test_batch_size
            if hasattr(task.task_config, "num_workers"):
                num_workers = task.task_config.num_workers
            else:
                num_workers = 0  # DYN: avoid DataLoader worker fork issues with DDP (32+ workers fork-bombs with model on CPU)

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=image_per_gpu,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                persistent_workers=(num_workers > 0),
                pin_memory=True,
            )

            # for d in dataloader:
            #     print(d['meta'][0]['frame_num'], len(d['meta']) )
            #     continue

            dataloaders.append(dataloader)

    return dataloaders, dataloader_start_idx
