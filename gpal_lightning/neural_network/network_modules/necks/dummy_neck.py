from gpal_lightning.neural_network.network_modules.necks.builder import NECKS
from gpal_lightning.neural_network.network_modules.base_module import BaseModule


@NECKS.register_module()
class DummyNeck(BaseModule):
    def forward(self, x):
        return x
