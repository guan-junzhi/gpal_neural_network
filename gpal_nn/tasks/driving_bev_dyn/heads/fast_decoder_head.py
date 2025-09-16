import torch
from torch import nn
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.network_modules.base_module import BaseModule
from gpal_lightning.neural_network.tasks.base.config_parsers.config_parser import (
    BaseConfigParser,
)
from tools_scripts.data_format_cvt import ShowDataStruct
import torchvision.models as models

class FastDecoderHead(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers_config = layers_config
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

        upsample_ratio = self.layers_config.get("upsample", 1)
        upsample_layers = []
        while upsample_ratio > 1:
            upsample_layers.append(
                nn.ConvTranspose2d(
                    self.layers_config["in_channels"],
                    128 if upsample_ratio == 2 else self.layers_config["in_channels"],
                    4,
                    2,
                    1,
                    bias=False,
                )
            )
            upsample_layers.append(nn.BatchNorm2d(128 if upsample_ratio == 2 else self.layers_config["in_channels"]))
            upsample_layers.append(nn.ReLU(True))
            upsample_ratio /= 2

        self.upsample = nn.Sequential(*upsample_layers)

        self.out = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
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


        # Apply upsampling
        x = self.upsample(x)
        # Apply out
        x = self.out(x)
        return x


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
