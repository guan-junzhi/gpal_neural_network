from gpal_lightning.neural_network.tasks.base.loggers.logger import BaseLogger
from gpal_lightning.neural_network.tasks.builder import LOGGERS


@LOGGERS.register_module()
class RADAR4D_NN_SDKLogger(BaseLogger):
    ...
