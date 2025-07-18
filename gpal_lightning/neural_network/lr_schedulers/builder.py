import inspect

from torch.optim import lr_scheduler

from gpal_lightning.utils.registry import Registry
from gpal_lightning.neural_network.lr_schedulers.GradualWarmupScheduler import GradualWarmupScheduler

LRSCHEDULER = Registry("LrScheduler")

LRSCHEDULER.register_module("MultiStepLR", module=lr_scheduler.MultiStepLR)
LRSCHEDULER.register_module(
    "CosineAnnealingLR", module=lr_scheduler.CosineAnnealingLR)
LRSCHEDULER.register_module("ExponentialLR", module=lr_scheduler.ExponentialLR)
LRSCHEDULER.register_module(
    "GradualWarmupScheduler", module=GradualWarmupScheduler)


def build_lr_scheduler(lr_scheduler_config, optimizer):
    lr_scheduler_config = lr_scheduler_config.copy()
    lr_scheduler_types = lr_scheduler_config.pop("type")
    if not isinstance(lr_scheduler_types, list):
        lr_scheduler_types = [lr_scheduler_types]

    if lr_scheduler_types[0] == 'GradualWarmupScheduler':
        lr_scheduler_types.insert(0, lr_scheduler_config['after_scheduler'])

    lr_scheulder = None
    for lr_scheduler_type in lr_scheduler_types:
        lr_scheduler_cls = LRSCHEDULER.get(lr_scheduler_type)
        required_args = inspect.getfullargspec(lr_scheduler_cls).args
        kwargs = {}
        for args in required_args:
            if args in lr_scheduler_config:
                kwargs[args] = lr_scheduler_config[args]
        if "after_scheduler" in required_args:
            kwargs["after_scheduler"] = lr_scheulder
        lr_scheulder = lr_scheduler_cls(optimizer, **kwargs)

    return lr_scheulder
