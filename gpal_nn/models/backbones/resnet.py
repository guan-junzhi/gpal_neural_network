import torchvision.models as models
import torch
import torch.nn as nn
from gpal_lightning.neural_network.network_modules.backbones.builder import BACKBONES
from gpal_lightning.neural_network.network_modules.base_module import BaseModule


@BACKBONES.register_module()
class ResNet(BaseModule):
    def __init__(self, global_config, num_layers, need_neck):
        # super(Resnet, self).__init__()
        super(ResNet, self).__init__(global_config)

        if num_layers == 18:
            self.model = models.resnet18(pretrained=False)
        elif num_layers == 34:
            self.model = models.resnet34(pretrained=False)
        elif num_layers == 50:
            self.model = models.resnet50(pretrained=False)
        elif num_layers == 101:
            self.model = models.resnet101(pretrained=False)
        else:
            raise ValueError(
                "Unsupported ResNet version. Choose from 18, 32, 50, or 101.")

        self.need_neck = need_neck
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool
        self.fc = self.model.fc
        self.m = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1)
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        if self.need_neck:
            return c2, c3, c4
        else:
            return [self.m(c4)]

    def _get_export_layers_channel(self):
        if self.need_neck:
            self.export_layers_channel = [128, 256, 512]

        else:
            self.export_layers_channel = [256]
        return self.export_layers_channel


if __name__ == "__main__":
    x = torch.randn((4, 3, 640, 640)).cuda()
    bb = ResNet(34).cuda()
    y = bb(x)
    print(y.shape)
