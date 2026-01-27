import torch
from torch import nn
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.network_modules.base_module import BaseModule
from gpal_lightning.neural_network.tasks.base.config_parsers.config_parser import (
    BaseConfigParser,
)
from tools_scripts.data_format_cvt import ShowDataStruct
import torchvision.models as models
import torch.nn.functional as F


class FastDecoderHead(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers_config = layers_config
        self.heads = {}
        self._setup()

    def _setup(self):
        trunk = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(
            self.layers_config["in_channels"], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        
        in_channels_list = self.layers_config["in_channels_list"]
        feat_out_channels = self.layers_config["feat_out_channels"]
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feat_out_channels, kernel_size=1),
                    nn.BatchNorm2d(feat_out_channels, momentum=0.1, eps=1e-5),
                    # nn.ReLU(inplace=True)
                )
            )
            
        self.heads = nn.ModuleDict()
        self.head_config = self.layers_config["HEAD"]
        for head_name, head_out_channels in self.head_config.items():

            self.heads[head_name] = nn.Sequential(
                nn.Conv2d(feat_out_channels, feat_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feat_out_channels, momentum=0.1, eps=1e-5),
                nn.Conv2d(feat_out_channels, int(head_out_channels), kernel_size=1, padding=0),
            )
            

        self.out = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.layers_config["out_channels"], kernel_size=1, padding=0),
        )

    def load_state_dict(self, state_dict, strict=True):
        prefix = "head.head1."
        unexpected_keys = []
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.split(prefix)[-1]
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            if param.shape != own_state[name].shape:
                own_state[name][:param.shape[0], ...].copy_(param)
                logging.warn("While copying the parameter named {}, " "whose dimensions in the model are {} and " "whose dimensions in the checkpoint are {}., only copying part {} param".format(name, own_state[name].size(), param.size(), param.size()))
            else:
                own_state[name].copy_(param)

        err_msg = []
        if unexpected_keys:
            err_msg.append("unexpected key in source state_dict: {}\n".format(", ".join(unexpected_keys)))

        err_msg = "\n".join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)

    def forward(self, x):
        # Apply ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Apply lateral convolutions
        x3 = self.lateral_convs[2](x3)
        x2 = self.lateral_convs[1](x2)
        x1 = self.lateral_convs[0](x1)
        # Apply output convolutions
        x2 = F.interpolate(x3,scale_factor=2,mode='bilinear')+x2
        x1 = F.interpolate(x2,scale_factor=2,mode='bilinear')+x1
        x = F.interpolate(x1,scale_factor=2,mode='bilinear')

        # Apply heads
        head_outputs = {}
        for head_name, head in self.heads.items():
            head_outputs[head_name] = head(x)
        # Apply out
        # x = self.out(x)
        return head_outputs


if __name__ == "__main__":
    # x = torch.randn((4, 3, 640, 640)).cuda()
    # bb = ResNet(34).cuda()
    # y = bb(x)
    # print(y.shape)

    import pickle as pkl
    # inputs = pkl.load(open("Simple_FPN.pkl", 'rb'))

    x = pkl.load(open("head_x.pkl", 'rb'))

    print(ShowDataStruct("x", x))

    head_config = {"in_channels": 256, "num_stages": 6, "out_channels": 21}

    head = FastDecoderHead(head_config)
    head = head.cuda()
    
    y = head(x)
    print(ShowDataStruct("y", y))
