import torchvision.models as models
import torch
import torch.nn as nn
from gpal_lightning.neural_network.network_modules.backbones.builder import BACKBONES
from gpal_lightning.neural_network.network_modules.base_module import BaseModule
from tools_scripts.data_format_cvt import ShowDataStruct


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


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.upsample = nn.Sequential(
            # 降低通道数至原1/4
            nn.Conv2d(1024, 256, 1),                  # 预降维
            nn.ConvTranspose2d(256, 256, 4, 2, 1),     # 反卷积
            # 3x3卷积增强特征融合
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = self.conv1(x_to_upsample)
        x = self.conv2(x)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv3(x_to_upsample)


@BACKBONES.register_module()
class EncoderRes50(BaseModule):
    def __init__(self, global_config, out_channels):
        super(EncoderRes50, self).__init__(global_config)

        # import pickle as pkl
        # pkl.dump((global_config, out_channels), open("EncoderRes50.pkl", 'wb'))
        # exit(1)
        self.C = out_channels
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.depth_layer = nn.Conv2d(128, self.C, kernel_size=1, padding=0)
        # self.upsampling_layer = UpsamplingConcat(0, 0)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        # x = self.upsampling_layer(x2, x1)
        # x = self.depth_layer(x)

    # 0<class 'torch.Tensor'> : torch.Size([4, 512, 40, 96]) torch.float32
    # 1<class 'torch.Tensor'> : torch.Size([4, 1024, 20, 48]) torch.float32
    # 2<class 'torch.Tensor'> : torch.Size([4, 2048, 10, 24]) torch.float32
        return [x1,x2,x3]


@BACKBONES.register_module()
class EncoderRes34(BaseModule):
    def __init__(self, global_config, out_channels):
        super(EncoderRes34, self).__init__(global_config)

        # import pickle as pkl
        # pkl.dump((global_config, out_channels), open("EncoderRes50.pkl", 'wb'))
        # exit(1)
        self.C = out_channels
        resnet = models.resnet34(pretrained=True)
        print(resnet.children())
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        print(list(resnet.children()))
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.depth_layer = nn.Conv2d(128, self.C, kernel_size=1, padding=0)
        # self.upsampling_layer = UpsamplingConcat(0, 0)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        # x = self.upsampling_layer(x2, x1)
    #     # x = self.depth_layer(x)
    # 0<class 'torch.Tensor'> : torch.Size([4, 128, 40, 96]) torch.float32
    # 1<class 'torch.Tensor'> : torch.Size([4, 256, 20, 48]) torch.float32
    # 2<class 'torch.Tensor'> : torch.Size([4, 512, 10, 24]) torch.float32
        return [x1,x2,x3]


if __name__ == "__main__":
    # x = torch.randn((4, 3, 640, 640)).cuda()
    # bb = ResNet(34).cuda()
    # y = bb(x)
    # print(y.shape)

    import pickle as pkl
    inputs = pkl.load(open("EncoderRes50.pkl", 'rb'))
    print("hello")
    # r50 = EncoderRes50(*inputs)
    r50 = EncoderRes34(*inputs)
    r50 = r50.cuda()
    x = torch.randn((20, 3, 320, 768)).cuda()
    while True:
        y = r50(x)

        print("hello")
        print(ShowDataStruct("y", y))

    #     0<class 'torch.Tensor'> : torch.Size([4, 64, 40, 96]) torch.float32