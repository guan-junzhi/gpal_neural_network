from gpal_lightning import const
from gpal_lightning.neural_network.runners.runner import Runner
from gpal_lightning.neural_network.tasks.build_task import build_tasks
from gpal_lightning.neural_network.network_modules.gpnet import GpNet

from gpal_lightning.utils.load_global_config import load_global_config
from gpal_lightning.utils.args_parser import ArgumentParserHelper

# from gpal_nn.models import necks, backbones, transformers
from gpal_nn.models import backbones, transformers

# import torch.multiprocessing as mp
# if mp.current_process().name == 'MainProcess':
#     mp.set_start_method('spawn')


def evaluate():
    args = ArgumentParserHelper.parse()
    dump_config = "save" in args and args.save != args.load_from
    global_config = load_global_config(
        args, override=False, dump_config=dump_config)

    global_config.validation = True
    tasks = build_tasks(global_config,
                        phase="validation",
                        tasks_root="gpal_nn.tasks")
    print(tasks)
    net = GpNet(global_config, tasks)

    for t in tasks:
        if hasattr(t, 'callbacks'):
            net.callbacks.extend(t.callbacks)

    runner = Runner(global_config, net, const.PHASE_VALIDATION)
    runner.eval()


if __name__ == '__main__':
    evaluate()
