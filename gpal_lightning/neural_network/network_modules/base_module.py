from abc import ABC
from typing import List, Union

from torch import nn

from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.network_modules.utils.state_resetable import StateResetable


class BaseModule(StateResetable, nn.Module, ABC):
    """BaseModule is used to replace the nn.Module, all necks/task heads should inherti it
    instead of nn.Module

    Definations:
    Model: The overall pytorch model contains modules like backbone, neck and heads
    Module: necks, heads
    Layer: permutations of convolution, normalization and activaton layers.

    a Model has backbone + multiple modules
    a module has multiple layers

    _setup() method is required for all child class, this method will be called
    to initialize the layers in the module, and all layers should be stored in the
    self.layers variable(a nn.ModuleDict instance). in the forward method please use self.layers to
    access layers.
    """

    def __init__(
        self,
        global_config: GlobalConfig,
        task_config=None,
        camera_name: str = None,
        batch_size: int = -1,
        freeze_module: bool = False,
        feature_id: list = None,
    ):
        """

        Args:
            global_config: global parameters
            task_config: optional config for task configs, it's used for task heads
            camera_name: control the camera id for layers in the module
            batch_size: control the batch size for layers in the module
            freeze_module: bool variable, all parameters in this module will be freezed while
            this paremter = True
        """
        super().__init__()
        self.global_config = global_config
        self.task_config = task_config
        self._camera_name = camera_name
        self._batch_size = batch_size
        self._feature_id = feature_id
        self.freeze_module = freeze_module
        self.layers = nn.ModuleDict()
        self._setup()
        self._init_weight()
        if freeze_module:
            self._freeze_module()

    def _init_weight(self) -> None:
        """initialize the conv and norm layers. Override for customized operation"""
        for layer in self.layers.values():
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def _setup(self):
        """Required method for all children class, all layers setup should be in this
        method."""

    def _freeze_module(self):
        """
        NOTE: set requires_grad to False only freeze parameter,
        not the running mean/var in norm layer.
        """
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @property
    def batch_size(self):
        """batch size is used to control the onnx model batch"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size
        for layer in self.layers.values():
            for module in layer.modules():
                module.batch_size = batch_size

    @property
    def camera_name(self):
        """camera id is used to control the switable batch norm"""
        return self._camera_name

    @camera_name.setter
    def camera_name(self, camera_name: str):
        self._camera_name = camera_name
        for layer in self.layers.values():
            for module in layer.modules():
                module.camera_name = camera_name

    @property
    def feature_id(self):
        return self._feature_id

    @feature_id.setter
    def feature_id(self, feature_id: Union[int, List]):
        if feature_id is not None:
            if isinstance(feature_id, int):
                feature_id = [feature_id]
            assert all(isinstance(x, int) for x in feature_id)
            self._feature_id = feature_id
