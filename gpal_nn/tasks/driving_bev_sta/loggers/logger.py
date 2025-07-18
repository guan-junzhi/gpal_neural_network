from gpal_lightning.neural_network.tasks.base.loggers.logger import BaseLogger
from gpal_lightning.neural_network.tasks.builder import LOGGERS


@LOGGERS.register_module()
class DRIVING_BEV_STALogger(BaseLogger):
    ...
