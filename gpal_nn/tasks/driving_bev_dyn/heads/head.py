import torch
import numpy as np
from torch import nn
from gpal_lightning.neural_network.tasks.builder import HEADS
from gpal_lightning.neural_network.tasks.base.heads.head import BaseHead
from gpal_nn.tasks.driving_bev_dyn.losses.loss import DRIVING_BEV_DYNLoss
import torch.nn.functional as F
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.heads.fast_decoder_head import FastDecoderHead
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf


class SeqFeatureFuser(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers_config = layers_config

        self.conv_fuser = nn.Sequential(
            nn.Conv2d(
            self.layers_config["in_channels"], self.layers_config["out_channels"], kernel_size=3, stride=1, padding=1, bias=False
        ), 
        nn.BatchNorm2d(self.layers_config["out_channels"]), nn.ReLU(True))
        
    def forward(self, prev_feats, cur_feats, cur2prev):
        x = torch.cat([cur_feats, cur_feats], dim = 1)
        return self.conv_fuser(x)


@HEADS.register_module()
class DRIVING_BEV_DYNHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=DRIVING_BEV_DYNLoss):
        self.task_config = task_config
        self.is_track_task = True  # 区分当前是否是Track任务
        self.head_conv = 64

        self.fuser_config = {"in_channels": 2048, "out_channels": 1024}
        self.head_config = {"in_channels": 1024,
                            "num_stages": 6, "out_channels": 21, "upsample": 4}

        self.feature_bank = None
        super(DRIVING_BEV_DYNHead, self).__init__(
            global_config, task_config, loss_func)

    def _setup(self):
        self.head = nn.ModuleDict()
        self.head["fuser"] = SeqFeatureFuser(self.fuser_config)
        self.head["center_head"] = FastDecoderHead(self.head_config)
    def load_state_dict(self, state_dict, strict=True):
        if len(self.head) == 1:
            self.head["center_head"].load_state_dict(state_dict, strict)
        else:
            for head_name, head in self.head.items():
                state_dict_sub = {k.replace(f"{head_name}.", ""): state_dict[k]
                                for k in state_dict if head_name in k}
                head.load_state_dict(state_dict_sub, strict)

    def forward(self, x: torch.Tensor, calib=None) -> torch.Tensor:
        # B,HW,C = x.shape
        # x = x.permute(0,2,1).reshape(B,C,96,240)
        if self.feature_bank == None:
            self.feature_bank = torch.zeros_like(x).detach().clone()
        prev_feats = self.feature_bank.clone()
        # self.feature_bank = x.detach().clone()

        cur2prev = torch.from_numpy(np.eye(3)).to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x_fuser = self.head["fuser"](prev_feats, x, cur2prev)
        x_decode = self.head["center_head"](x_fuser)
        batch_dict = {'head_conv': x_decode[:, 6:], "hm_cen": x_decode[:, :6]}
        return [batch_dict]
