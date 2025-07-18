from torch import optim

from gpal_lightning.utils.registry import Registry

OPTIMIZERS = Registry("Optimizer")
OPTIMIZERS.register_module(name="SGD", module=optim.SGD)
OPTIMIZERS.register_module(name="Adadelta", module=optim.Adadelta)
OPTIMIZERS.register_module(name="Adagrad", module=optim.Adagrad)
OPTIMIZERS.register_module(name="Adam", module=optim.Adam)
OPTIMIZERS.register_module(name="ASGD", module=optim.ASGD)
OPTIMIZERS.register_module(name="AdamW", module=optim.AdamW)
OPTIMIZERS.register_module(name="Adamax", module=optim.Adamax)


def build_optimizer(optimizer_config, params):
    optimizer_config = optimizer_config.copy()
    optimizer_type = optimizer_config.pop("type")
    optimizer_cls = OPTIMIZERS.get(optimizer_type)
    return optimizer_cls(params, **optimizer_config)
