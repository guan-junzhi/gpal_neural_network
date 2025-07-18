from abc import ABC, abstractmethod

import numpy as np

from gpal_lightning.neural_network.global_config import GlobalConfig


class BasePostProcess(ABC):
    """
    Interface for all postprocessing modules
    """

    def __init__(self, global_config: GlobalConfig, task_config):
        """

        Args:
            global_config: instance contains global parameters
            task_config: instance contains task parameters
        """
        self.global_config = global_config
        self.task_config = task_config

    @abstractmethod
    def process(self, vectors: np.ndarray, metadata: dict, is_gt: bool = False) -> dict:
        """

        Args:
            vectors: numpy format array from network output
            metadata: dictionary contains the metadata info of the image(vector)
            is_gt: boolean key indicates whether current vector is gt or not,
            different postprocess logic can be applied

        Returns: human-readble dictionary for evaluation/visualization

        """
        raise NotImplementedError
