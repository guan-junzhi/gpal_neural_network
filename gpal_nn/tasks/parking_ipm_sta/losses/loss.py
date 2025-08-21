import torch
import numpy as np
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.losses.loss import BaseLoss
from gpal_lightning.neural_network.tasks.builder import LOSSES
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf
import time
from tools_scripts.data_format_cvt import ShowDataStruct
import pickle as pkl
import torch.nn as nn


# point_sigma = 5.6, line_sigma = 3.2, line_pad = dummy
class Crit1_WideRange_L2_Loss(nn.Module):
    def __init__(self, w1=1.0, w2=1.0, reduction="none"):
        super(Crit1_WideRange_L2_Loss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.reduction = reduction
        self.c_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, input_point, input_line, target_gtmap):
        target_point = target_gtmap[:, 0:1, :, :]
        target_line = target_gtmap[:, 1:2, :, :]
        loss_p = self.c_loss(input_point, target_point)
        loss_l = self.c_loss(input_line, target_line)
        loss = self.w1 * loss_p + self.w2 * loss_l
        return loss


@LOSSES.register_module()
class PARKING_IPM_STALoss(BaseLoss):
    def __init__(self, global_config: GlobalConfig, task_config):
        super(PARKING_IPM_STALoss,
              self).__init__(global_config, task_config)
        self.criterion_1 = Crit1_WideRange_L2_Loss(
            w1=1.0, w2=1.0, reduction='sum')

    def forward(self, preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor) -> dict:
        """

        Args:
            preds: preds tensor from model
            trues: trues tensor from dataloader
            masks: masks tensor from preprocess modules, used for mask invalid labels

        Returns: dictionary contains the loss of current batch.

        """

        # trues_cat = torch.stack(trues, dim=0)
        # print(ShowDataStruct("preds", preds))
        # print(ShowDataStruct("trues", trues))
        if isinstance(trues, torch.Tensor):
            loss = {"total_loss": self.criterion_1(input_point=preds[0], input_line=preds[1],
                                                   target_gtmap=trues)}
        else:
            loss = {"total_loss": 0.0}

        return loss
