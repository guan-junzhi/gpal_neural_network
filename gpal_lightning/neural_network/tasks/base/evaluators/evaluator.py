from abc import ABC, abstractmethod

from gpal_lightning.data.dataset_identifier import DatasetIdentifier
from gpal_lightning.neural_network.global_config import GlobalConfig


class BaseEvaluator(ABC):
    """
    Base evaluator class for kpi computation.
    Workflow:
        1. intializaed evaluator instance
        2. pass pred and true json into with process API one by one
        3. call generate_kpi API to calculate aggregate KPI
    """

    def __init__(self, global_config: GlobalConfig, task_config):
        self.global_config = global_config
        self.task_config = task_config
        self._dataset_identifier = None
        self._dedups = set()

    @property
    def dataset_identifier(self):
        """camera id is used to control the switable batch norm"""
        return self._dataset_identifier

    @dataset_identifier.setter
    def dataset_identifier(self, dataset_identifier: DatasetIdentifier):
        self._dataset_identifier = dataset_identifier

    @abstractmethod
    def generate_kpi(self) -> dict:
        """This API will be called after all images are processed, and it will return
        a dict contains all kpis
        """

    @abstractmethod
    def process(self, pred: dict, true: dict, metadata: dict) -> None:
        """This API will take in the pred/true and metadata info of one image,
        and store their info in the evaluator instance

        Args:
            pred: dict contains pred info of one image
            true: dict contains true info of one image
            metadata: metadata of this image

        Returns: None

        """
