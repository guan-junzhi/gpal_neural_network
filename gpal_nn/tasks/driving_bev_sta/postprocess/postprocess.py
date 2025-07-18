from gpal_lightning.neural_network.tasks.builder import POSTPROCESSES
from gpal_lightning.neural_network.tasks.base.postprocesses.postprocess import (
    BasePostProcess,
)


@POSTPROCESSES.register_module()
class DRIVING_BEV_STAPostProcessing(BasePostProcess):
    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)

    def process(self, vectors, metadata: dict, is_gt: bool = False) -> dict:
        pass
