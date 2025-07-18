import json
import math

import numpy as np

def set_datasets_properties(global_config, tasks: list):
    """This function is used to calculate the dataset properties,
    such as ratio, workers"""
    total_frequency = sum(task.task_config.frequency for task in tasks)
    total_workers = global_config.workers_per_gpu

    for task in tasks:
        if not hasattr(task, "train_datasets"):
            continue
        datasets = task.train_datasets
        if not datasets:
            continue
        phase = datasets[0].phase
        if phase == "training":
            workers = round(total_workers * task.task_config.frequency / total_frequency)
        else:
            workers = total_workers // len(tasks)
        workers = max(workers, 1)

        assert all(
            len(dataset) for dataset in datasets
        ), "Datasets: {0} has/have len(dataset) == 0".format( 
            [dataset.sql_filter for dataset in datasets if len(dataset) == 0]
        )

        if any(dataset.ratio != 0.0 for dataset in datasets):
            assert all(
                dataset.ratio != 0.0 for dataset in datasets
            ), "dataset ratio should be all zero or all none-zero in config"

            # normalize ratio and turn it to a cumulative list
            ratios = [dataset.ratio for dataset in datasets]
            ratios = np.cumsum(ratios) / sum(ratios)
            for dataset, ratio in zip(datasets, ratios):
                dataset.ratio = ratio

        else:
            num_sample = [len(dataset) for dataset in datasets]
            ratios = get_sampler_ratio(num_sample)
            for ratio, dataset in zip(ratios, datasets):
                dataset.ratio = ratio
        # set up the dataset usage percentage based on accumulative ratio
        dataset_usage_percentage = _get_task_datasets_percentage(task.name, datasets)
        for idx, dataset in enumerate(datasets):
            percentage = dataset_usage_percentage[idx]
            dataset.worker = max(math.ceil(workers * percentage), 1)

def _get_task_datasets_percentage(task_name, datasets):
    dataset_usage_percentage = []
    for idx, dataset in enumerate(datasets):
        if idx == 0:
            dataset_usage_percentage.append(dataset.ratio)
        else:
            dataset_usage_percentage.append(dataset.ratio - datasets[idx - 1].ratio)

    return dataset_usage_percentage


def get_sampler_ratio(num_sample):
    ratios = list(np.cumsum(num_sample) / np.sum(num_sample))
    ratios[-1] = 1.0
    return ratios
