from gpal_lightning.neural_network.tasks.builder import PREPROCESSES
from gpal_lightning.neural_network.tasks.base.preprocesses.preprocess import BasePreProcess


@PREPROCESSES.register_module()
class DRIVING_BEV_STAPreProcessing(BasePreProcess):
    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)

    def process(self, data, phase, **kwargs):
        return
