import logging
from typing import List
from typing import Union

import numpy as np
import torch

from gpal_lightning import const
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.datasets.base_dataset import BaseDataset

class ImageBaseDataset(BaseDataset):
    """ BaseDataset is a iterable-style dataset. (https://pytorch.org/docs/stable/data.html)

    Workflow:
    1. fetch a datablob given idx.
    2. pass datablob to _data_preprocess, this method will call task.preprocess with below
    """

    def __init__(self,
                 global_config: GlobalConfig,
                 task_config,
                 preprocess,
                 dataset_name: str,
                 camera_name: str,
                 root_dir: str,
                 phase: str,
                 shuffle: bool = True,
                 shuffle_seed: int = 0,
                 sql_filter: str = "",
                 pseudo_labels_path: Union[str, list] = None,
                 worker: int = 0,
                 ratio: float = 0.0):
        """
        Args:
            global_config: object from CONSTANT class
            task_config: object from task ConfigParser class
            preprocess: preprocess object of current task
            dataset_name: a str
            # TODO: support this feature
            camera_name: used to filter data from data source
            root_dir: data prefix
            phase: it can and only can be from "training" and "validation"
            worker: number of workers for the dataloader of this dataset, this value is calculated automatically.
            ratio: accumulated sampling ratio of this dataset, this can be calculated automatically based on the
            length of dataset, if you want to assign the ratio manully, please make sure the ratio of each dataset
            is in ascending order, and the ratio of last dataset is 1.0
        """

        # self.sequential_data = task_config.sequential_data
        self.image_processor = None
        self.augmentations = None

        super().__init__(global_config=global_config,
                     task_config=task_config,
                     preprocess=preprocess,
                     dataset_name=dataset_name,
                     phase=phase,
                     camera_name=camera_name,
                     root_dir=root_dir,
                     shuffle=shuffle,
                     shuffle_seed=shuffle_seed,
                     sql_filter=sql_filter,
                     ratio=ratio,
                     worker=worker,
                     pseudo_labels_path=pseudo_labels_path,
                     )

    def __len__(self):
        return len(self.dataset)

    def _data_preprocess(self, data_blob):
        processed_results = self.preprocess.process(data_blob["label"],
                                                    data_blob[const.AUGMENTATIONS],
                                                    data_blob[const.METADATA],
                                                    None)

        data_blob["label"] = _covert2ndarry_fp32(processed_results["label"])
        data_blob["mask"] = _covert2ndarry_fp32(processed_results["mask"])

        return data_blob


def _covert2ndarry_fp32(data):
    """ 
    Args: 
        data: unknown obj and unknown dtype
    Returns:
        np.ndarray with dtype.float32

    """

    if isinstance(data, np.ndarray):
        if data.dtype == np.float32:
            return data
        else:
            return data.astype(np.float32)
    else:
        return np.array(data, dtype=np.float32)
    