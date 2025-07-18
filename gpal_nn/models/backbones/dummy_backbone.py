from gpal_lightning.neural_network.network_modules.backbones.builder import BACKBONES
from gpal_lightning.neural_network.network_modules.base_module import BaseModule


@BACKBONES.register_module()
class DummyBackbone(BaseModule):
    def forward(self, x):
        return x

