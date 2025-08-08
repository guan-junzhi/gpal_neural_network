from abc import ABC

from torch import nn

from gpal_lightning.neural_network.global_config import GlobalConfig


class BaseLayer(nn.Module, ABC):
    """BaseLayer defines the basic attributes shared by all layers,
    all layers used in the pipeline should inherit from this class
    """

    def __init__(self, global_config: GlobalConfig, camera_name: str = None, batch_size: int = -1):
        """

        Args:
            global_config: instance contains the global parameters
            camera_name: camera id for current layer, used by camera related operations
            batch_size: used to control the batch size whiling doing onnx conversion. otherwise a
            reshape operation will be stored in the static graph.
        """
        super().__init__()
        self.global_config = global_config
        self._camera_name = camera_name
        self._batch_size = batch_size

    @property
    def batch_size(self):
        """batch size is used to control the onnx model batch"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    @property
    def camera_name(self):
        """camera id is used to control the switable batch norm"""
        return self._camera_name

    @camera_name.setter
    def camera_name(self, camera_name: str):
        self._camera_name = camera_name
