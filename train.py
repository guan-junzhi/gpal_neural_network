from pytorch_lightning import seed_everything
from gpal_lightning.neural_network.runners.runner import Runner
from gpal_lightning.utils.args_parser import ArgumentParserHelper
from gpal_lightning.utils.load_global_config import load_global_config
from gpal_lightning.neural_network.network_modules.gpnet import GpNet
from gpal_lightning.neural_network.tasks.build_task import build_tasks

from gpal_lightning import const

# from gpal_nn.models import necks, backbones, transformers
from gpal_nn.models import  backbones, transformers


def train():
    # Parse argument
    args = ArgumentParserHelper.parse()
    print(args)
    # Iitial constant
    global_config = load_global_config(args)
    
    # fix seed                    
    if global_config.seed:
        seed_everything(global_config.seed, True)  
        
    # Build tasks
    tasks = build_tasks(global_config,
                        phase="training",
                        tasks_root="gpal_nn.tasks")
    net = GpNet(global_config, tasks)
    
    runner = Runner(global_config, net, const.PHASE_TRAINING)
    runner.train()

if __name__=="__main__":
    train()