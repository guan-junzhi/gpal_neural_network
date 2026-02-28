import torch
from torch import nn
from gpal_lightning.neural_network.network_modules.base_module import BaseModule
from gpal_lightning.neural_network.network_modules.necks.builder import NECKS
from tools_scripts.data_format_cvt import ShowDataStruct


@NECKS.register_module()
class Simple_FPN(BaseModule):
    def __init__(self,
                 global_config,
                 layers_config: dict,
                 freeze_module: bool = False,
                 ):

        # import pickle as pkl
        # pkl.dump((global_config, layers_config, freeze_module), open("Simple_FPN.pkl", 'wb'))
        # exit(1)

        self.global_config = global_config
        self.layers_config = layers_config
        super().__init__(global_config, freeze_module=freeze_module)
        hidden_dim = layers_config['out_channel']
        self.conv_channels = layers_config['conv_channels']
        # self.conv_channels = [512, 1024, 2048]
        self.use_relu = layers_config.get('use_relu', False)
        self.drop_smooth_conv = layers_config.get('drop_smooth_conv', False)
        self.include_input = layers_config.get('include_input', False)
        input_proj_list = []
        for _ in range(len(self.conv_channels)):
            in_channels = self.conv_channels[_]
            if self.use_relu:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim, momentum=0.1, eps=1e-5),
                    nn.ReLU(inplace=True)
                ))
            else:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim, momentum=0.1, eps=1e-5)
                ))
        # input_proj_list.append(nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
        # ))
        self.input_proj = nn.ModuleList(input_proj_list)
        
        self.p5_to_p4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.p4_to_p3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.smoothp3 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(hidden_dim, momentum=0.1, eps=1e-5))
        # self.smoothp3_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.smoothp4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                               nn.BatchNorm2d(256, momentum=0.1, eps=1e-5))
        # self.smoothp4_0 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                                 nn.BatchNorm2d(256, momentum=0.1, eps=1e-5))
        # self.smoothp4_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.smoothp5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                               nn.BatchNorm2d(256, momentum=0.1, eps=1e-5))
        # self.smoothp5_0 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                                 nn.BatchNorm2d(256, momentum=0.1, eps=1e-5))
        # self.smoothp5_down = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.smoothp6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                               nn.BatchNorm2d(256, momentum=0.1, eps=1e-5))

        if self.use_relu:
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)


    def _setup(self):
        """Doing layer setup with Layer Factory here"""

    def forward(self, x):
        # import pickle as pkl
        # pkl.dump(x, open("Simple_FPN_x.pkl", 'wb'))
        # exit(1)
        output = []
        for l in range(len(self.conv_channels)):
            output.append(self.input_proj[l](x[l]))
        p3, p4, p5 = output[0], output[1], output[2]

        if self.use_relu:
            p4 = self.relu1(self.p5_to_p4(p5) + p4)
            p3 = self.relu2(self.p4_to_p3(p4) + p3)
        else:
            p4 = self.p5_to_p4(p5) + p4
            p3 = self.p4_to_p3(p4) + p3

        if not self.drop_smooth_conv:
            p3 = self.smoothp3(p3)

        if self.include_input:
            x.extend([p3, p4, p5])
        else:
            x = [p3, p4, p5]
        # return [x[idx] for idx in self.feature_id]
        return [p3]


if __name__ == "__main__":
    # x = torch.randn((4, 3, 640, 640)).cuda()
    # bb = ResNet(34).cuda()
    # y = bb(x)
    # print(y.shape)

    import pickle as pkl
    inputs = pkl.load(open("Simple_FPN.pkl", 'rb'))
    fpn = Simple_FPN(*inputs)
    fpn = fpn.cuda()
    x = pkl.load(open("Simple_FPN_x.pkl", 'rb'))

    print(ShowDataStruct("x", x))
    y = fpn(x)
    print(ShowDataStruct("y", y))
