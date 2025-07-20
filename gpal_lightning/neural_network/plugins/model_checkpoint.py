
import os
from collections import OrderedDict

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info

from gpal_lightning import const
from gpal_lightning.utils.get_checkpoint_path import get_checkpoint_path
import logging


def _search_prefix(dict_to_search, prefix):
    dict_after_search = OrderedDict()
    prefix = f"model.{prefix}."
    for key, val in dict_to_search.items():
        if key.startswith(prefix):
            dict_after_search[key[len(prefix):]] = val
    return dict_after_search

def Simplify(dict_to_search):
    while len(set([k.split('.')[0] for k in dict_to_search])) == 1:
        dict_to_search = {'.'.join(k.split('.')[1:]): dict_to_search[k] for k in dict_to_search}
    return dict_to_search


class EpochModelCheckpoint(ModelCheckpoint):
    def __init__(self, load_from, to_resume, *args, **kwargs):
        self.load_from: str = load_from
        self.to_resume: bool = to_resume
        self.checkpoint = None
        self.save_checkpoint_before_validation = kwargs.pop(
            "save_checkpoint_before_validation", False)
        self._last_saved_epoch = -1
        self.current_mode = kwargs.pop("mode")
        if self.load_from:
            # self.load_from = get_checkpoint_path(load_from)
            self.load_from = load_from
            assert os.path.isfile(
                self.load_from), f"[Checkpoint] Checkpoint path: {self.load_from} not a file"
            self.checkpoint = torch.load(self.load_from, map_location="cpu")
            rank_zero_info(
                f"[Checkpoint] Loading checkpoint from path: {self.load_from}")
        super().__init__(*args, **kwargs)

    def _load_model_weights(self, module, module_name, state_dict_to_load):
        rank_zero_info(f"[Checkpoint] Loading module weight: {module_name}")
        try:
            module.load_state_dict(state_dict_to_load, strict=True)
        except Exception as e:
            logging.error(
                "load state dict %s ", e)

    def setup(self, trainer, pl_module, stage):
        assert stage.name in [
            "FITTING",
            "VALIDATING",
            "PREDICTING",
        ], "[Checkpoint] Only support training/validating/predict usage"

        if stage.name in ["VALIDATING", "PREDICTING"]:
            assert self.checkpoint, "[Checkpoint] No checkpoint found load_from path"
            self.to_resume = False
            state_dict = self.checkpoint["state_dict"]
            self._load_model_weights(pl_module, "ALL", state_dict)

        elif self.to_resume and self.checkpoint:
            # Resume from or -c -s will always read all
            state_dict = self.checkpoint["state_dict"]
            self._load_model_weights(pl_module, "ALL", state_dict)
            # Load Current Iteration
            pl_module.curr_iteration = self.checkpoint["curr_iteration"]
            # Load Epoch
            trainer.fit_loop.epoch_progress.current.completed = self.checkpoint["epoch"]
            # # Load Global Step
            # trainer.fit_loop.epoch_loop.global_step = self.checkpoint["global_step"]
            print(trainer.fit_loop.epoch_loop.global_step)

        elif self.load_from:
            state_dict = self.checkpoint["state_dict"]
            
            if not pl_module._load_module_dict:
                self._load_model_weights(pl_module, "ALL", state_dict)
            else:
                for module_name, module in pl_module.model.items():
                    load_module_names = pl_module.get_load_module_names(
                        module_name)
                    print(module_name, load_module_names)
                    
                    if load_module_names:
                        if load_module_names == "all":
                            
                            # print(_search_prefix(state_dict, module_name).keys())
                            print(module)
                            dict_for_module = Simplify(_search_prefix(state_dict, module_name))
                            self._load_model_weights(
                                module, module_name, dict_for_module)
                        else:
                            assert isinstance(load_module_names, list), (
                                f"[Checkpoint] Only support list-type load parameter"
                                f" got {type(load_module_names)} instead"
                            )

                            for load_module_name in load_module_names:
                                module_to_load = module.get_submodule(
                                    load_module_name)
                                prefix = f"{module_name}.{load_module_name}"
                                self._load_model_weights(
                                    module_to_load, prefix, _search_prefix(state_dict, prefix))

    def on_train_start(self, trainer, pl_module):
        if self.to_resume:
            rank_zero_info(
                f"[Checkpoint] Resume training states from: {self.load_from}")
            # Load Optimizer and LR
            pl_module.optimizers().load_state_dict(
                self.checkpoint["optimizer_states"][0])
            if torch.__version__ < const.PYTORCH_UPGRADE_VERSION:
                for _ in range(self.checkpoint["lr_schedulers"][0]["_step_count"]):
                    pl_module.lr_schedulers().step()
                pl_module.lr_schedulers().optimizer = pl_module.optimizers()
            else:
                pl_module.lr_schedulers().load_state_dict(
                    self.checkpoint["lr_schedulers"][0])
            # Load grad scaler
            for task, grad_scaler in pl_module.grad_scalers.items():
                if task in self.checkpoint.get("grad_scalers", {}):
                    grad_scaler.load_state_dict(
                        self.checkpoint["grad_scalers"][task])
                else:
                    rank_zero_info(
                        f"[Checkpoint] Task: {task} fp16 grad scaler initiate from config")

    def on_validation_start(
        self,
        trainer="pl.Trainer",
        pl_module="pl.LightningModule",
    ) -> None:
        """Save a checkpoint at the beginning of the validation stage."""
        if (
            self._every_n_epochs >= 1
            and (pl_module.current_epoch + 1) % self._every_n_epochs == 0
            and self._last_saved_epoch != pl_module.current_epoch
            and self.mode == const.PHASE_TRAINING
        ):
            self.filename = f"epoch={pl_module.current_epoch}"
            self.save_checkpoint(trainer)
            self._last_saved_epoch = trainer.current_epoch


class IterationModelCheckpoint(EpochModelCheckpoint):
    """This checkpoint callback is used to save checkpoint in iteration instead of epoch"""

    def __init__(self, *args, save_every_n_iterations=-1, **kwargs):
        super().__init__(*args, **kwargs)
        if save_every_n_iterations <= 0:
            raise ValueError(
                "[Checkpoint] Save_every_n_iterations should be greater than 0.")
        self.save_every_n_iterations = save_every_n_iterations
        self._last_saved_iteration = -1
        self.filename = ""

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        # dataloader_idx,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module,
                                   outputs, batch, batch_idx)
        if (
            pl_module.curr_iteration % self.save_every_n_iterations == 0
            and self._last_saved_iteration != pl_module.curr_iteration
        ):
            self.filename = os.path.join(
                self.dirpath, f"epoch={pl_module.current_epoch}-step={pl_module.curr_iteration}_{const.CHECKPOINT_NAME_LAST}{const.FILE_EXTENSION}")
            self._save_checkpoint(trainer, self.filename)
            self._last_saved_iteration = pl_module.curr_iteration

    def on_train_epoch_end(self, trainer, pl_module, unused=None) -> None:
        """Skip the epoch end save"""
