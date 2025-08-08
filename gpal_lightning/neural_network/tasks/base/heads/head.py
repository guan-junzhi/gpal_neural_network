from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import nn

from gpal_lightning.neural_network.global_config import GlobalConfig


# TODO: consider use BaseModule
class BaseHead(nn.Module, ABC):
    """Base task head class, all task head should inherit from it
    forward_test and forward_train is required for all children class
    """

    def __init__(self, global_config: GlobalConfig, task_config, loss_func, freeze_module: bool = False):
        """

        Args:
            global_config: instance contains global parameters
            task_config: instance contains task parameters
            loss_func: nn module class contains the loss computation
            freeze_module: bool variable, all parameters in this module will be freezed while
        """
        super().__init__()
        self.global_config = global_config
        self.task_config = task_config
        self.loss = loss_func(global_config, task_config)
        self._camera_name = None
        self._batch_size = -1
        self.freeze_module = freeze_module
        self.head: Union[nn.Module, nn.Sequential, None] = None

        self._setup()
        if freeze_module:
            self._freeze_module()
        assert self.head is not None, "Task head should be network but it's None"

    def _freeze_module(self):
        """
        NOTE: set requires_grad to False only freeze parameter,
        not the running mean/var in norm layer.
        """
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @abstractmethod
    def _setup(self):
        ...

    @property
    def batch_size(self):
        """batch size is used to control the onnx model batch"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size
        for module in self.head.modules():
            module.batch_size = batch_size

    @property
    def camera_name(self):
        """camera id is used to control the switable batch norm"""
        return self._camera_name

    @camera_name.setter
    def camera_name(self, camera_name: str):
        self._camera_name = camera_name
        for module in self.head.modules():
            module.camera_name = camera_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
