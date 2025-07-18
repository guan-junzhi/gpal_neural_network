import torch
from torch import nn
from gpal_lightning.neural_network.tasks.builder import HEADS
from gpal_lightning.neural_network.tasks.base.heads.head import BaseHead
from gpal_nn.tasks.driving_bev_sta.heads.maptr.instance_decoder import MapInstanceDetectorHead
from gpal_nn.tasks.driving_bev_sta.losses.loss import DRIVING_BEV_STALoss


@HEADS.register_module()
class DRIVING_BEV_STAHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=DRIVING_BEV_STALoss):
        super(DRIVING_BEV_STAHead, self).__init__(
            global_config, task_config, loss_func)

    def _setup(self):
        self.head = nn.ModuleDict()
        self.head_type = {}
        for head_name, head_config in self.task_config.Head.items():
            head_type = head_config.pop(
                "head_type") if "head_type" in head_config else head_config["type"]
            if head_type == "MapInstanceDetectorHead":
                head_cls = MapInstanceDetectorHead
            else:
                raise NotImplementedError

            self.head[head_name] = head_cls if head_type == "Sparse4DHead" else head_cls(
                self.global_config, self.task_config, **head_config)
            self.head_type[head_name] = head_type

    def load_state_dict(self, state_dict, strict=True):
        for head_name, head in self.head.items():
            head.load_state_dict(state_dict, strict)

    def forward(self, x: torch.Tensor, calib=None) -> torch.Tensor:
        outputs = []
        for head_name, head in self.head.items():
            output = head(x, calib)
            head_type = self.head_type[head_name]
            outputs.append(output)
        if isinstance(outputs[0], torch.Tensor) and not torch.onnx.is_in_onnx_export() and not self.global_config.dump_calibset:
            outputs = torch.cat(outputs, dim=1)
        return outputs
