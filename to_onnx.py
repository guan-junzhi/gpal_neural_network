# import yaml

# from zpilot_lightning import const
# from zpilot_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.plugins.onnx_convert import PytorchToOnnx
# from zpilot_lightning.utilities.args_parser import ArgumentParserHelper
# from zpilot_lightning.neural_network.tasks.build_task import build_tasks
# from zpilot_lightning.utilities.load_global_config import load_global_config

# from zpilot_nn.models import backbones, necks, transformers

from gpal_lightning import const
from gpal_lightning.neural_network.runners.runner import Runner
from gpal_lightning.neural_network.tasks.build_task import build_tasks
from gpal_lightning.neural_network.network_modules.gpnet import GpNet

from gpal_lightning.utils.load_global_config import load_global_config
from gpal_lightning.utils.args_parser import ArgumentParserHelper

# from gpal_nn.models import necks, backbones, transformers
from gpal_nn.models import backbones, transformers

def to_onnx():
    args = ArgumentParserHelper.parse()
    config_path = args.config
    save_path = args.save

    global_config = load_global_config(args, override = True, dump_config = False)

    tasks = build_tasks(global_config,
                        phase="onnx_gen",
                        tasks_root="gpal_nn.tasks")
    tasks = [task for task in tasks if task.task_config.frequency != 0]
    task_names = [task.name for task in tasks]
    for task in tasks:
        task.build_head()
    PytorchToOnnx.to_onnx(global_config, tasks, save_path)

if __name__ == '__main__':
    to_onnx()