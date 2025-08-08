from abc import ABC, abstractmethod

from gpal_lightning.neural_network.global_config import GlobalConfig


class BasePreProcess(ABC):
    """Preprocess interface for all tasks"""

    def __init__(self, global_config: GlobalConfig, task_config):
        self.global_config = global_config
        self.task_config = task_config

    @abstractmethod
    def process(self, label: dict, augmentations: dict, metadata: dict) -> dict:
        raise NotImplementedError


