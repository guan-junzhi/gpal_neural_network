import torch
from torch import nn
from gpal_lightning.neural_network.tasks.builder import HEADS
from gpal_lightning.neural_network.tasks.base.heads.head import BaseHead
from gpal_nn.tasks.radar4d_nn_sdk.losses.loss import RADAR4D_NN_SDKLoss
import torch.nn.functional as F


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        BatchNorm2d = nn.SyncBatchNorm
        bn_mom = 0.1
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes,
                               kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):

        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out


@HEADS.register_module()
class RADAR4D_NN_SDKHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=RADAR4D_NN_SDKLoss):
        super(RADAR4D_NN_SDKHead, self).__init__(
            global_config, task_config, loss_func)
        # num_classes=1, inplanes=128, interplanes=64, scale_factor=8
        self.task_config = task_config

    def _setup(self):
        self.head = nn.ModuleDict()
        self.head_type = {}
        for head_name, head_config in self.task_config.Head.items():
            head_type = head_config.pop(
                "head_type") if "head_type" in head_config else head_config["type"]
            if head_type == "SlotHead":
                self.num_classes = head_config['num_classes']
                self.scale_factor = head_config['scale_factor']

                self.seg_p = segmenthead(
                    head_config['inplanes'], head_config['interplanes'], head_config['num_classes'])
                self.seg_l = segmenthead(
                    head_config['inplanes'], head_config['interplanes'], head_config['num_classes'])
            else:
                raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        for head_name, head in self.head.items():
            head.load_state_dict(state_dict, strict)

        state_dict_seg_p = {
            k.replace("seg_p.", ""): state_dict[k] for k in state_dict if "seg_p." in k}
        self.seg_p.load_state_dict(state_dict_seg_p, strict)
        state_dict_seg_l = {
            k.replace("seg_l.", ""): state_dict[k] for k in state_dict if "seg_l." in k}
        self.seg_l.load_state_dict(state_dict_seg_l, strict)

    def forward(self, x: torch.Tensor, calib=None) -> torch.Tensor:
        branch_point = self.seg_p(x)
        branch_line = self.seg_l(x)
        height = x.shape[-2] * self.scale_factor
        width = x.shape[-1] * self.scale_factor
        branch_point = F.interpolate(branch_point,
                                     size=[height, width],
                                     mode='bilinear')
        branch_line = F.interpolate(branch_line,
                                    size=[height, width],
                                    mode='bilinear')
        return branch_point, branch_line
