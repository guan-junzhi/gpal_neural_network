import itertools
import random
from collections import defaultdict
from typing import Callable
import numpy as np
from torch import distributed
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning.utilities import rank_zero_info
import logging

from gpal_lightning import const
from gpal_lightning.data.dataloader_helpers.gpal_collate import gpal_collate
from gpal_lightning.data.dataloader_helpers.clip_sampler import ClipSampler


class RatioDataloader:
    """Ratio_dataloader warps several pytorch Dataloader objs, which aims to sample a batch data from one of them based on frequency and ratio"""

    def __init__(self,
                 global_config,
                 tasks: list,
                 curr_iteration: int = 0,
                 collate_fn: Callable = gpal_collate,
                 ):
        self.global_config = global_config
        self.dataloaders = defaultdict(list)
        self.ratio = defaultdict(list)
        self.curr_iteration = curr_iteration
        self.task_frequency = {}
        self.collate_fn = collate_fn

        try:
            self.global_rank = distributed.get_rank()
        except (RuntimeError, AssertionError):
            self.global_rank = -1

        for task in tasks:
            if not task.task_config.frequency:
                continue
            if not task.train_datasets:
                continue
            self.task_frequency[task.name] = task.task_config.frequency

        for task in tasks:
            if not task.task_config.frequency:
                continue
            use_concat_dataset = getattr(
                task.task_config, 'use_concat_dataset', False)
            if use_concat_dataset:
                dataset_name_list, concat_list, ratio_list, data_num_list, num_worker, datalists = [], [], [], [], 0, []
                for dataset in task.train_datasets:
                    batch_size = global_config.image_per_gpu
                    worker_init_fn = getattr(dataset, "worker_init_fn", None)
                    if hasattr(task.task_config, "batch_size"):
                        batch_size = task.task_config.batch_size
                    concat_list.append(dataset)
                    ratio_list.append(dataset.ratio)
                    # clip数量
                    data_num_list.append(len(dataset))
                    num_worker += dataset.worker
                    dataset_name_list.append(dataset.dataset_name)
                    datalists += dataset.dataset

                default_num_worker_limit = 8
                num_worker_limit = getattr(
                    task.task_config, 'num_worker_limit', default_num_worker_limit)
                prefetch_factor = getattr(
                    task.task_config, 'prefetch_factor', 2)
                logging.info(
                    f"{task} use_concat_dataset num_worker = {num_worker} limit to {num_worker_limit} prefetch_factor {prefetch_factor}")
                num_worker = min(num_worker, num_worker_limit)

                print(
                    f"{task} use_concat_dataset num_worker = {num_worker} limit to {num_worker_limit} prefetch_factor {prefetch_factor}")

                # Due to the dataset ratio was calculate in a cumsum manner in datasets_preperties,
                # we need recalculate it to ecah dataset ratios here.
                for i, ratio in enumerate(ratio_list):
                    if i == 0:
                        continue
                    else:
                        # import pdb;pdb.set_trace()
                        ratio_list[i] = ratio - sum(ratio_list[:i])
                concat_dataset = ConcatDataset(concat_list)
                setattr(concat_dataset, 'camera_name', dataset.camera_name)
                if task.name not in ["DRIVING_BEV_DYN"]:
                    weighted_list = []
                    for i, ratio in enumerate(ratio_list):
                        weighted = ratio * sum(data_num_list) / data_num_list[i]
                        weighted_list.extend([weighted] * data_num_list[i])
                        rank_zero_info(
                            f"dataset_name: {dataset_name_list[i]}, "
                            f"has ratio: {round(ratio, 3)} in concat dataset "
                        )
                    sampler = WeightedRandomSampler(
                        weighted_list, len(weighted_list), replacement = (len(ratio_list) > 1))
                else:
                    sampler = ClipSampler(
                        datalists, default_resample_len=1600, batch_size=batch_size, length_range=[5, 25], rank = self.global_rank)

                dataloader = DataLoader(
                    dataset=concat_dataset,
                    batch_size=batch_size,
                    num_workers=num_worker,
                    sampler=sampler,
                    shuffle=False,
                    pin_memory=getattr(task.task_config, 'pin_memory', True),
                    drop_last=True,
                    collate_fn=self.collate_fn,
                    worker_init_fn=worker_init_fn,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=True
                )
                self.dataloaders[task.name].append(dataloader)
                self.ratio[task.name].append(1.)
            else:
                raise NotImplementedError
        # we don't want to re-init the iterator
        # because we need to track the iterator.curr_iteration during training
        self.iter = RatioBasedIter(self)

    def __iter__(self):
        return self.iter

    def __len__(self):
        """1 epoch means iterates through the task with longest sum(len(datasets))"""
        return sum([sum([len(x) for x in dataloader])
                   for dataloader in self.dataloaders.values()])


class RatioBasedIter:
    """This iter helper will decide which task and task dataloader to sample data"""

    def __init__(self, ratio_dataloader):
        self.dataloaders = ratio_dataloader.dataloaders
        self.iter_dataloaders = {
            task_name: [
                iter(dataloader) for dataloader in dataloaders] for task_name,
            dataloaders in self.dataloaders.items()}
        self.ratio = ratio_dataloader.ratio
        self.curr_iteration = ratio_dataloader.curr_iteration
        self.task_frequency = ratio_dataloader.task_frequency
        self.task_iterations = {
            task_name: int(self.curr_iteration * frequency)
            for task_name, frequency in self.task_frequency.items()
        }
        self.tasks = list(self.task_frequency.keys())
        try:
            self.global_rank = distributed.get_rank()
        except (RuntimeError, AssertionError):
            self.global_rank = -1
        self.task_index = itertools.cycle(range(len(self.tasks) + 1))
        self.prev_task = None

    def __iter__(self):
        return self

    def __next__(self):
        # choose which task to sample, we support task_frequency > 1.
        # when task_frequency > 1, the same task may be sampled more than once per curr_iteration.
        curr_task = None
        while curr_task is None:
            # if prev_task has reached enough iterations we move to the next task,
            # otherwise, we sample the prev_task again.
            if self.prev_task is None or \
                    self.task_iterations[self.prev_task] > \
                    int(self.task_frequency[self.prev_task] * self.curr_iteration):
                task_index = next(self.task_index)
                if task_index >= len(self.tasks):
                    task_index = next(self.task_index)
                    self.curr_iteration += 1
                candidate_task = self.tasks[task_index]
                task_frequency = self.task_frequency[candidate_task]
                if task_frequency == 0:
                    continue
                self.prev_task = candidate_task
                if self.task_iterations[candidate_task] > \
                        int(self.task_frequency[candidate_task] * self.curr_iteration):
                    continue
                curr_task = candidate_task
            else:
                curr_task = self.prev_task
        self.task_iterations[curr_task] += 1

        dataloaders = self.dataloaders[curr_task]
        iter_dataloaders = self.iter_dataloaders[curr_task]
        ratios = self.ratio[curr_task]
        dataloader_idx = np.searchsorted(ratios, random.random())
        try:
            data_blob = next(iter_dataloaders[dataloader_idx])
        except StopIteration:
            # python doesn't have has_next method for iterator
            iter_dataloaders[dataloader_idx] = iter(
                dataloaders[dataloader_idx])
            data_blob = next(iter_dataloaders[dataloader_idx])

        camera_name = dataloaders[dataloader_idx].dataset.camera_name

        return {
            "curr_iteration": self.curr_iteration,
            "camera_name": camera_name,
            "curr_task": curr_task,
            **data_blob,
        }

    # Python 2 compatibility
    next = __next__

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders.values())
