import torchvision as tv
import torch
import torch.nn as nn
from gpal_lightning.neural_network.network_modules.backbones.builder import BACKBONES
from gpal_lightning.neural_network.network_modules.base_module import BaseModule


@BACKBONES.register_module()
class ResNet(BaseModule):
    def __init__(self, global_config, num_layers, need_neck):
        # super(Resnet, self).__init__()
        super(ResNet, self).__init__(global_config)

        d = num_layers
        if d == 18:
            '''
            b, 3, h, w --> b, 512, h / 32, w / 32
            '''
            self.m = nn.Sequential(
                *list(tv.models.resnet18(pretrained=True).children())[:-2],
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1)
            )

        elif d == 34:
            '''
            b, 3, h, w --> b, 512, h / 32, w / 32
            '''
            self.m = nn.Sequential(
                *list(tv.models.resnet34(pretrained=True).children())[:-2],
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1)
            )
        elif d == 50:
            '''
            b, 3, h, w --> b, 2048, h / 32, w / 32
            '''
            self.m = nn.Sequential(
                *list(tv.models.resnet50(pretrained=True).children())[:-2],
                nn.Conv2d(2048, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.Conv2d(1024, 1024, 3, padding=1)
            )
        elif d == 101:
            '''
            b, 3, h, w --> b, 2048, h / 32, w / 32
            '''
            self.m = nn.Sequential(
                *list(tv.models.resnet101(pretrained=True).children())[:-2],
                nn.Conv2d(2048, 1024, 1),
                nn.BatchNorm2d(1024),
                nn.Conv2d(1024, 1024, 3, padding=1)
            )
        else:
            raise NotImplementedError

        # print(list(self.m))
        # exit(1)

    def forward(self, x):
        return [self.m(x)]


if __name__ == "__main__":
    x = torch.randn((4, 3, 640, 640)).cuda()
    bb = ResNet(34).cuda()
    y = bb(x)
    print(y.shape)
