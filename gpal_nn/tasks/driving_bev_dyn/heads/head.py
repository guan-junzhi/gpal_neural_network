import torch
from torch import nn
from gpal_lightning.neural_network.tasks.builder import HEADS
from gpal_lightning.neural_network.tasks.base.heads.head import BaseHead
from gpal_nn.tasks.driving_bev_dyn.losses.loss import DRIVING_BEV_DYNLoss
import torch.nn.functional as F
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.heads.fast_decoder_head import FastDecoderHead
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf



@HEADS.register_module()
class DRIVING_BEV_DYNHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=DRIVING_BEV_DYNLoss):
        self.task_config = task_config
        self.is_track_task = True  # 区分当前是否是Track任务
        self.head_conv = 64

        self.head_config = {"in_channels": 1024,
                            "num_stages": 6, "out_channels": 21, "upsample": 4}

        super(DRIVING_BEV_DYNHead, self).__init__(
            global_config, task_config, loss_func)

    def _setup(self):
        self.head = nn.ModuleDict()
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
        x = self.head["center_head"](x)
        batch_dict = {'head_conv': x[:, 6:], "hm_cen": x[:, :6]}
        return [batch_dict]
