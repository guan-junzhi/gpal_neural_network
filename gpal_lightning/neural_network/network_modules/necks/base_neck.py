from abc import ABC

from gpal_lightning.neural_network.network_modules.base_module import BaseModule


class BaseNeck(BaseModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone_feature_id: list = []

    @property
    def backbone_feature_id(self):
        return self._backbone_feature_id

    @backbone_feature_id.setter
    def backbone_feature_id(self, backbone_feature_id):
        self._backbone_feature_id = backbone_feature_id
