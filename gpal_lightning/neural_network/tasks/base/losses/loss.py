from abc import ABC

import torch
from torch import nn

from gpal_lightning.neural_network.global_config import GlobalConfig


class BaseLoss(nn.Module, ABC):
    """Base loss class, all task loss should inherit from it,
    it compute the loss given preds, trues and masks"""

    def __init__(self, global_config: GlobalConfig, task_config):
        super().__init__()
        self.global_config = global_config
        self.task_config = task_config

    def forward(self, preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor) -> dict:
        """

        Args:
            preds: preds tensor from model
            trues: trues tensor from dataloader
            masks: masks tensor from preprocess modules, used for mask invalid labels

        Returns: dictionary contains the loss of current batch.

        """
        loss = {}
        for attribute_obj in self.task_config.attributes.values():
            loss.update(attribute_obj.loss(preds, trues, masks))
        loss.update({"total_loss": sum(loss.values())})
        return loss
