import torch
import torch.nn as nn
import torch.nn.functional as F
from gpal_lightning.neural_network.network_modules.necks.builder import NECKS
from gpal_lightning.neural_network.network_modules.base_module import BaseModule


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        # self.upsample_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.ReLU6(inplace=True)
                )
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU6(inplace=True)
                )
            )

            # self.upsample_convs.append(
            #     nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            # )

    def forward(self, inputs):
        laterals = [conv(inputs[i])
                    for i, conv in enumerate(self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                scale_factor=2,
                mode='bilinear'
            )
            # upsampled = self.upsample_convs[i - 1](laterals[i])
            # laterals[i - 1] += upsampled

        outputs = []
        for i in range(len(laterals)):
            outputs.append(self.output_convs[i](laterals[i]))

        return outputs


class PAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels):
        super(PAN, self).__init__()
        self.num_levels = num_levels
        self.convs = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i in range(num_levels):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            if i < num_levels - 1:
                self.downsample_layers.append(
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=2, padding=1)
                )

    def forward(self, features):
        pan_features = []
        for i in range(self.num_levels):
            if i == 0:
                x = self.convs[i](features[i])
            else:
                downsampled = self.downsample_layers[i -
                                                     1](pan_features[i - 1])
                merged = features[i] + downsampled
                x = self.convs[i](merged)
            pan_features.append(x)
        return pan_features[0]


class PAN_UpMerge(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels):
        super(PAN_UpMerge, self).__init__()
        self.num_levels = num_levels
        self.convs = nn.ModuleList()
        # self.upsample_convs = nn.ModuleList()

        # 为每个层级创建卷积(包括最高层)
        for _ in range(num_levels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU6(inplace=True)
                )
            )
            # self.upsample_convs.append(
            #     nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            # )

    def forward(self, features):
        """
        输入特征顺序为[P3, P4, P5] (尺寸从大到小)
        输出特征顺序保持[N3, N4, N5]
        """
        pan_features = []

        # 从高层向低层处理 (P5 -> P4 -> P3)
        for i in reversed(range(self.num_levels)):
            if i == self.num_levels - 1:
                x = self.convs[i](features[i])
            else:
                upsampled = F.interpolate(
                    pan_features[0],
                    scale_factor=2,
                    mode='nearest'
                )
                # upsampled = self.upsample_convs[i](pan_features[0])
                merged = features[i] + upsampled
                x = self.convs[i](merged)

            pan_features.insert(0, x)

        return pan_features[1]


class PAFPN(nn.Module):
    def __init__(self, backbone_out_channels, fpn_out_channels=256):
        super(PAFPN, self).__init__()
        self.fpn = FPN(backbone_out_channels, fpn_out_channels)
        self.pan = PAN(
            in_channels=fpn_out_channels,
            out_channels=fpn_out_channels,
            num_levels=len(backbone_out_channels)
        )

    def forward(self, features):
        fpn_outputs = self.fpn(features)
        pan_output = self.pan(fpn_outputs)
        return pan_output


@NECKS.register_module()
class PAFPN_Up(BaseModule):
    def __init__(self, global_config, backbone_out_channels, fpn_out_channels=256):
        super(PAFPN_Up, self).__init__(global_config)
        self.fpn = FPN(backbone_out_channels, fpn_out_channels)
        self.pan = PAN_UpMerge(
            in_channels=fpn_out_channels,
            out_channels=fpn_out_channels,
            num_levels=len(backbone_out_channels)
        )

    def forward(self, features):
        fpn_outputs = self.fpn(features)
        pan_output = self.pan(fpn_outputs)
        return [pan_output]


if __name__ == "__main__":
    backbone_out_channels = [512, 1024, 2048]
    model = PAFPN_Up(backbone_out_channels)

    C3 = torch.randn(2, 512, 64, 64)
    C4 = torch.randn(2, 1024, 32, 32)
    C5 = torch.randn(2, 2048, 16, 16)

    outputs = model([C3, C4, C5])

    for i, feat in enumerate(outputs):
        print(f"Output N{i + 3} shape: {feat.shape}")

    import onnx
    from onnxsim import simplify

    onn_name = 'PAFPN_Up.onnx'
    torch.onnx.export(model, [C3, C4, C5],
                      onn_name,
                      export_params=True,
                      verbose=10,
                      input_names=['input'],
                      output_names=['output'],
                      )

    onnx_model = onnx.load(onn_name)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onn_name)
