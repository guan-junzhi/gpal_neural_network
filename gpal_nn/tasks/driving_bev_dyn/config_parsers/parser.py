from gpal_lightning.neural_network.tasks.base.config_parsers.config_parser import BaseConfigParser
from gpal_lightning.neural_network.tasks.builder import CONFIGPARSERS


@CONFIGPARSERS.register_module()
class DRIVING_BEV_DYNConfigParser(BaseConfigParser):
    def __init__(self, global_config, task_config):
        super().__init__(global_config=global_config, task_config=task_config)
        self._parse_attribute(task_config)

    def _load_constants(self) -> None:
        pass

    def _parse_attribute(self, task_config: dict):
        pass
