import torch
from torch import nn
from gpal_lightning.neural_network.tasks.builder import HEADS
from gpal_lightning.neural_network.tasks.base.heads.head import BaseHead
from gpal_nn.tasks.driving_bev_dyn.losses.loss import DRIVING_BEV_DYNLoss
import torch.nn.functional as F
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.heads.pointtransformersiamese import PointnetTransformerSiamese
from gpal_nn.tasks.driving_bev_dyn.heads.fast_decoder_head import FastDecoderHead
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf


BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ImageNeck(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.block = BasicBlock
        self.inplanes = 64  # ori 128
        self.model_cfg = model_cfg
        self.crop_area = self.model_cfg.get("CROP_AREA", [[0, 96], [0, 160]])
        self.layers = self.model_cfg['LAYER_NUMS']
        self.layer_strides = self.model_cfg['LAYER_STRIDES']
        self.down_filters = self.model_cfg['DOWN_FILTERS']
        self.up_filters = self.model_cfg['UPSAMPLE_FILTERS']
        self.cat_layers = self.model_cfg['CAT_FILTERS']
        self.num_bev_features = self.model_cfg['num_bev_features']
        self.relu = nn.ReLU()
        # radar
        # self.conv1 = nn.Conv2d(input_channels, self.down_filters[0], kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1   = nn.BatchNorm2d(self.down_filters[0], momentum=BN_MOMENTUM)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.sp_atten_rd = SpatialAttention(3)
        # self.sp_atten_im = SpatialAttention(3)
        # self.adaptive_fusion = AdaptiveFeatureFusion()
        # self.conv_down   = nn.Conv2d(self.down_filters[0], self.down_filters[0] // 4, kernel_size=1, bias=False)
        # self.conv_up     = nn.Conv2d(self.down_filters[0] // 4, self.down_filters[0], kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # self.conv2 = nn.Conv2d(self.down_filters[0], self.down_filters[0], kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2   = nn.BatchNorm2d(self.down_filters[0], momentum=BN_MOMENTUM)
        if 'Fusion_Sptial' in model_cfg:
            self.sf = True
            # self.spatial_fusion = Spatial_fusion(3)
        else:
            self.sf = False
        self.conv_flag = False
        if 'IMAGE_VOXEL_SIZE' not in model_cfg:
            self.conv_flag = True
            self.conv3 = nn.Conv2d(
                256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # fusion
        self.bn3 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        # self.conv4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn4   = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.cbam = CBAM(self.up_filters[0])

        # downsample8x
        self.layer1 = self._make_layer(
            self.block, self.down_filters[0], self.layers[0], stride=self.layer_strides[0])
        self.layer2 = self._make_layer(
            self.block, self.down_filters[1], self.layers[1], stride=self.layer_strides[1])
        self.layer3 = self._make_layer(
            self.block, self.down_filters[2], self.layers[2], stride=self.layer_strides[2])
        self.layer4 = self._make_layer(
            self.block, self.down_filters[3], self.layers[3], stride=self.layer_strides[3])

        # upsample
        self.conv_up_level1 = nn.Conv2d(
            512, 128, kernel_size=1, stride=1, padding=0)
        self.conv_up_level2 = nn.Conv2d(
            256, 128, kernel_size=1, stride=1, padding=0)
        self.conv_up_level3 = nn.Conv2d(
            320, 64, kernel_size=1, stride=1, padding=0)

        # self.up_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.upsample0 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.upsample1 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.conv_up_level4 = nn.Conv2d(self.cat_layers[3], self.up_filters[3], kernel_size=1, stride=1, padding=0)
        # self.conv_up_level5 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.shared_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # self.share_up_conv= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, batch_dict):
        # radar 192 640
        # x = batch_dict['spatial_features']  # 96 192 640

        # x  = self.conv1(x)

        # x  = self.bn1(x)   # 96 192 640
        # # x = x*x0  # 1, 64, 192, 640 | 0.32
        # x  = self.relu(x)  # 1, 64, 192, 640 | 0.32
        # x2 = self.conv2(x)
        # x2 = self.bn2(x2)
        # x1 = self.global_pool(x2)
        # # x0 = self.sp_atten_rd(x)
        # x1 = self.conv_down(x1)
        # x1 = self.relu(x1)
        # x1 = self.conv_up(x1)
        # x1 = self.sig(x1)
        # x2 = x2*x1  # 1, 64, 192, 640 | 0.32
        # x2 = self.relu(x2)  # 1, 64, 96, 320 | 0.64
        # x2 = F.normalize(x2, p=2, dim=1)

        # image 96 160
        # x3 = batch_dict['backbone_2d_img']  # 64*4=256 96 160
        x3 = batch_dict

        # x4 = self.sp_atten_im(x3)
        if self.conv_flag:
            x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        # x3 = F.normalize(x3, p=2, dim=1)

        # fusion
        # ---------------------------- two range 拼接方式  ---------------------------- #
        # way1
        # HR, WR = x2.size()[2:]
        # NI, CI, _, WI = x3.size()
        # XI = torch.zeros(NI, CI, HR, WR).to(x3)
        # XI[:, :, :, :WI] += x3  # 64 96 320 | 0.64  <-
        # if x2.size()[2:] == x3.size()[2:]:
        #     WI = WI//2
        # if self.sf:
        #     x2, XI = self.spatial_fusion(x2, XI)
        # x2 = torch.cat([x2, XI], dim=1)  # 128 96 320 <-
        # x2 = self.conv4(x2)
        # x2 = self.bn4(x2)
        # x2 = self.relu(x2)
        # ---------------------------- two range 拼接方式  ---------------------------- #
        x2 = x3
        # Down Sample 8X
        # [N,  64, 96, 240] <- [N, 64, 96 240]
        out_layer1 = self.layer1(x2)
        out_layer2 = self.layer2(out_layer1)  # [N, 128, 48, 120] <-
        out_layer3 = self.layer3(out_layer2)  # [N, 256, 24,  60] <-
        out_layer4 = self.layer4(out_layer3)  # [N, 512, 12,  30] <-

        # # ~FPN cat-channel
        # # Up Sample 8X  此时都在 det范围内，若作用于occ任务，需要切片过去
        # up_level1      = F.interpolate(out_layer4, scale_factor=2, mode='bilinear', align_corners=True)  # 1 512 24 80
        # concat_level1  = torch.cat((up_level1, out_layer3), dim=1)   # [1, 768, 24, 80]
        # up_level2      = F.interpolate(self.conv_up_level1(concat_level1), scale_factor=2, mode='bilinear', align_corners=True)  # [1, 128, 48, 160]

        # concat_level2  = torch.cat((up_level2, out_layer2), dim=1)  # [1, 256, 48, 160]
        # conv_up_level2 = self.conv_up_level2(concat_level2)  # [1, 64, 48, 160]
        # out_leveal_2   = F.interpolate(conv_up_level2, scale_factor=2, mode='bilinear', align_corners=True)  # [1, 64, 96, 320]

        # out_leveal_1   = F.interpolate(up_level2, scale_factor=2, mode='bilinear', align_corners=True)  # [1, 128, 96, 320]
        # det_cat        = torch.cat((x2, out_leveal_1, out_leveal_2, out_layer1), dim=1)  # 1, 384=(64+128+64+64), 96, 320]
        # # det_cat        = torch.cat((out_leveal_1, out_leveal_2, out_layer1), dim=1)  # 1, 384=(128+64+64), 96, 320]
        # conv_up_level3 = self.conv_up_level3(det_cat)  # [1, 64, 96, 320]  ---- det

        # out_layer4     = self.up_1(out_layer4) # 512 -> 256
        # up_level1      = F.interpolate(out_layer4, scale_factor=2, mode='bilinear', align_corners=True)  # 256
        # concat_level1  = torch.cat((up_level1, out_layer3), dim=1)   # 512
        # up_level2      = F.interpolate(self.conv_up_level1(concat_level1), scale_factor=2, mode='bilinear', align_corners=True)  # 128

        # concat_level2  = torch.cat((up_level2, out_layer2), dim=1)  # 256
        # conv_up_level2 = self.conv_up_level2(concat_level2)  # 256 -> 128
        # out_leveal_2   = F.interpolate(conv_up_level2, scale_factor=2, mode='bilinear', align_corners=True)  # 128
        # out_leveal_1   = F.interpolate(up_level2, scale_factor=2, mode='bilinear', align_corners=True)  # 128
        # det_cat        = torch.cat((out_leveal_1, out_leveal_2, out_layer1), dim=1)  # 1, 320=(128+64+64), 96, 320]
        # conv_up_level3 = self.conv_up_level3(det_cat) # 320->64
        # det_cat1        = torch.cat((x3, conv_up_level3), dim=1)  # 1, 128=(64+64), 96, 240]
        # shared_conv    = self.shared_conv(det_cat1)
        # breakpoint()

        out_layer4 = self.upsample0(out_layer4)
        concat_level1 = torch.cat((out_layer4, out_layer3), dim=1)  # 512
        up_level2 = self.upsample1(
            self.conv_up_level1(concat_level1))  # 512 128 128

        concat_level2 = torch.cat(
            (up_level2, out_layer2), dim=1)  # 128+128=256
        conv_up_level2 = self.conv_up_level2(concat_level2)  # 256 -> 128
        out_leveal_2 = self.upsample2(conv_up_level2)  # 128 -> 128
        out_leveal_1 = self.upsample3(up_level2)  # 128 -> 128
        # 1, 320=(128+64+64), 96, 320]
        det_cat = torch.cat((out_leveal_1, out_leveal_2, out_layer1), dim=1)

        conv_up_level3 = self.conv_up_level3(det_cat)
        # 1, 128=(64+64), 96, 240]
        det_cat1 = torch.cat((x3, conv_up_level3), dim=1)
        shared_conv_for_occ = self.shared_conv(det_cat1)

        # breakpoint()

        # 注意网格 size和相关的尺寸
        # 其上,上采样再取
        # 其下,直接除以倍数再取

        # occ
        # occ_lvldiv2    = F.interpolate(conv_up_level3[...,:WI], scale_factor=2, mode='bilinear', align_corners=True)    # [1, 128, 192, 320]
        # # occ_catdiv2    = self.conv_up_level4(torch.cat((x[...,:WI*2], occ_lvldiv2), dim=1))                     # [1,  64, 192, 320] <-
        # occ_catdiv2    = self.conv_up_level4(occ_lvldiv2) # [1,  64, 192, 320] <-
        # shared_conv    = self.shared_conv(occ_catdiv2)

        # occ_lvldiv1    = F.interpolate(shared_conv, scale_factor=2, mode='bilinear', align_corners=True)  # [1, 64, 384, 640]
        # out            = self.conv_up_level5(occ_lvldiv1)  # [1, 64, 384, 640]
        W0 = int(self.crop_area[0][0])
        W1 = int(self.crop_area[0][1])
        L0 = int(self.crop_area[1][0])
        L1 = int(self.crop_area[1][1])

        spatial_features_scale = [1, 2, 4, 4, 4, 8, 8]
        spatial_features_2d = [
            shared_conv_for_occ,
            shared_conv_for_occ,
            # out_leveal_2[..., :WI],
            # conv_up_level3,
            # out_leveal_1[..., :WI],
            # up_level2[..., :int(WI/2)],
            # conv_up_level2[..., :int(WI/2)],
            out_leveal_2[:, :, W0:W1, L0:L1],  # 0.64
            conv_up_level3,  # for det task
            out_leveal_1[:, :, W0:W1, L0:L1],
            up_level2[:, :, int(W0/2):int(W1/2), int(L0/2):int(L1/2)],
            conv_up_level2[:, :, int(W0/2):int(W1/2), int(L0/2):int(L1/2)],
        ]

        """ 
            date: 20250123
            shared_conv,      # 1 occ-match | 1, 64, 96, 240
            shared_conv,      # 2 occ-match | 1, 64, 96, 240
            out_leveal_2,     # 4 occ-not   | 1, 128, 80, 192 
            conv_up_level3,   # 4 occ-not   | 1, 64, 96, 240
            out_leveal_1,     # 4 occ-not   | 1, 128, 80, 192 
            up_level2,        # 8 occ-not   | 1, 128, 40, 96  
            conv_up_level2,   # 8 occ-not   | 1, 128, 40, 96  
        """

        return spatial_features_scale, spatial_features_2d


class CenterHead(nn.Module):
    def __init__(self, head_conv, shared_input_channels):
        super(CenterHead, self).__init__()
        self.heads = {'hm_cen': 6}

        for head in sorted(self.heads):
            if head == 'hm_cen':
                fpn_channels = 64
            else:
                fpn_channels = 64
            num_output = self.heads[head]
            fc = nn.Sequential(nn.Conv2d(fpn_channels, head_conv, kernel_size=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(head_conv, num_output, kernel_size=1, bias=True))
            self.__setattr__('fpn_{}'.format(head), fc)

        self.shared_conv = nn.Sequential(
            nn.Conv2d(shared_input_channels, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )

    def generate_predicted_hm_cen(self, x_dict, batch_size, **kwargs):
        # 是有梯度的
        hm = x_dict['hm_cen']  # 这里的key是yaml的
        _, _, H, W = hm.size()
        hm = hm.sigmoid()
        heat = F.max_pool2d(hm, (3, 3), stride=1, padding=1)
        keep = (hm == heat).float()
        hm = hm * keep

        return hm.view(batch_size, 1, hm.shape[1], hm.shape[2], hm.shape[3])

    def forward(self, x):
        ret = {}
        self.forward_ret_dict = {}
        spatial_features_2d = x
        shared_conv = self.shared_conv(spatial_features_2d)
        ret['head_conv'] = shared_conv

        for head in self.heads:
            fdn_input = shared_conv
            fpn_out = self.__getattr__('fpn_{}'.format(head))(fdn_input)
            ret[head] = fpn_out  # hm_cen

        # self.forward_ret_dict['target'] = ret

        # if self.training:
        #     if self.is_track_task:  # 如果是多帧，会对两个热力图进行合并,但是没有使用
        #         B, C, H, W = data_dict['gt_curr_hm_cen'].size()
        #         self.forward_ret_dict['gt_curr_hm_cen'] = torch.cat([
        #             torch.unsqueeze(data_dict['gt_curr_hm_cen'], dim=1),
        #             torch.unsqueeze(data_dict['gt_prev_hm_cen'], dim=1)], dim=1).view(B*2, C, H, W)  #
        #     else:
        #         raise NotImplementedError
        #         self.forward_ret_dict['gt_curr_hm_cen'] = data_dict['gt_curr_hm_cen']
        #     self.forward_ret_dict['track'] = self.is_track_task

        # if self.is_track_task:
        #     data_dict['hm_cen_pred'] = self.generate_predicted_hm_cen(
        #         batch_size=data_dict['batch_size'])
        # else:
        #     raise NotImplementedError
        #     if self.training:
        #         self.forward_ret_dict['gt_curr_cen_offset'] = data_dict['gt_curr_cen_offset']
        #         self.forward_ret_dict['gt_curr_direction'] = data_dict['gt_curr_direction']
        #         self.forward_ret_dict['gt_curr_z_coor'] = data_dict['gt_curr_z_coor']
        #         self.forward_ret_dict['gt_curr_dim'] = data_dict['gt_curr_dim']
        #         self.forward_ret_dict['gt_curr_indices_center'] = data_dict['gt_curr_indices_center']
        #         self.forward_ret_dict['gt_curr_obj_mask'] = data_dict['gt_curr_obj_mask']
        #         self.forward_ret_dict['gt_curr_vel'] = data_dict['gt_curr_vel']

        #     if (not self.training) or self.predict_boxes_when_training:
        #         batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #             batch_size=data_dict['batch_size'])
        #         data_dict['batch_cls_preds'] = batch_cls_preds
        #         data_dict['batch_box_preds'] = batch_box_preds
        #         data_dict['cls_preds_normalized'] = True

        hm_cen_pred = self.generate_predicted_hm_cen(ret,
                                                     batch_size=int(spatial_features_2d.shape[0]))
        ret["hm_cen_pred"] = hm_cen_pred
        return ret


@HEADS.register_module()
class DRIVING_BEV_DYNHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=DRIVING_BEV_DYNLoss):
        self.task_config = task_config
        self.is_track_task = True  # 区分当前是否是Track任务
        self.head_conv = 64

        # self.neck_cfg = dict(
        #     NAME="ImageNeck",
        #     # NAME="FusionOccSim",
        #     Fusion_Sptial=True,
        #     CROP_AREA=[[8, 88], [16, 208]],
        #     LAYER_NUMS=[2, 2, 2, 2],
        #     LAYER_STRIDES=[1, 2, 2, 2],
        #     DOWN_FILTERS=[64, 128, 256, 512],
        #     CAT_FILTERS=[768, 256, 384, 64],
        #     UPSAMPLE_FILTERS=[128, 64, 64, 64],
        #     num_bev_features=[64, 64, 128, 64, 128, 128, 128]
        # )
        self.head_config = {"in_channels": 256,
                            "num_stages": 6, "out_channels": 1, "upsample": 4}

        super(DRIVING_BEV_DYNHead, self).__init__(
            global_config, task_config, loss_func)

    def _setup(self):
        self.head = nn.ModuleDict()
        self.head["seg_head"] = FastDecoderHead(self.head_config)
    def load_state_dict(self, state_dict, strict=True):
        if len(self.head) == 1:
            self.head["seg_head"].load_state_dict(state_dict, strict)
        else:
            for head_name, head in self.head.items():
                state_dict_sub = {k.replace(f"{head_name}.", ""): state_dict[k]
                                for k in state_dict if head_name in k}
                head.load_state_dict(state_dict_sub, strict)

    def forward(self, x: torch.Tensor, calib=None) -> torch.Tensor:
        print(x.shape)
        # B,HW,C = x.shape
        # x = x.permute(0,2,1).reshape(B,C,96,240)
        x = self.head["seg_head"](x)
        batch_dict = {'seg': x}
        return [batch_dict]
