from abc import ABC, abstractmethod

from gpal_lightning.data.dataset_identifier import DatasetIdentifier
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.utils.clean_kpi import clean_kpi


class BaseKPIFormatter(ABC):
    """This is the base KPI formatter class"""

    def __init__(self, global_config: GlobalConfig, task_config):
        self.global_config = global_config
        self.task_config = task_config

    @abstractmethod
    def format_kpi(self, kpi: dict, dataset_identifier: DatasetIdentifier) -> str:
        """

        Args:
            kpi: kpi of single task
            dataset_identifier: instance contains the information of current dataset
        Returns: formatted str, will be dumped into a txt file

        """

    def filter_kpi_to_log(self, kpi: dict):
        """
        Args:
            kpi: kpi of a single task

        Returns:
            filtered_kpi: dict, a new kpi dict whose keys may be modified for logging, values
                that are not float may be removed.
            kpi_count: int, total number of kpis in filtered_kpi, recursive.
        """
        return clean_kpi(kpi)
