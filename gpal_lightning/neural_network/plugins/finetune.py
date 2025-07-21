from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info


class Finetune(BaseFinetuning):
    """This is the call back to finetune and to freeze/unfreeze parameter."""

    def __init__(self, train_bn=False):
        super().__init__()
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module):

        for module_name, module in pl_module.model.items():
            if module.freeze_module:
                self.to_freeze(module_name, module, train_bn=self.train_bn)

            freeze_module_names = pl_module.get_freeze_module_names(
                module_name)

            if freeze_module_names:
                # Deal with speical case
                if freeze_module_names == "all":
                    self.to_freeze(module_name, module, train_bn=self.train_bn)
                elif freeze_module_names == "bn":
                    rank_zero_info(
                        f"[Finetune] Freezing {module_name}'s bn eval statistics")
                    continue
                else:
                    assert isinstance(
                        freeze_module_names, list
                    ), f"[Finetune] Only support list-type freeze parameter got {type(freeze_module_names)} instead"

                    for freeze_module_name in freeze_module_names:
                        module_to_freeze = module.get_submodule(
                            freeze_module_name)
                        module_to_freeze_name = ".".join(
                            [module_name, freeze_module_name])
                        self.to_freeze(module_to_freeze_name,
                                       module_to_freeze, train_bn=self.train_bn)

    def finetune_function(self, pl_module, epoch, optimizer):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.train_bn:
            return

        for module_name, module in pl_module.model.items():
            if module.freeze_module:
                self.eval(module)

            freeze_module_names = pl_module.get_freeze_module_names(
                module_name)

            if freeze_module_names:
                if freeze_module_names == "all" or freeze_module_names == "bn":
                    self.eval(module)
                else:
                    for freeze_module_name in freeze_module_names:
                        module_to_freeze = module.get_submodule(
                            freeze_module_name)
                        self.eval(module_to_freeze)

    def to_freeze(self, modules_name, modules, train_bn):
        rank_zero_info(
            f"[Finetune] Freezing module: {modules_name} with train_bn: {train_bn}")
        super().freeze(modules, train_bn)

    def eval(self, modules):
        modules.eval()
