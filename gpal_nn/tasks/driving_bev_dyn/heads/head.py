import torch
import numpy as np
from torch import nn
import copy
from gpal_lightning.neural_network.tasks.builder import HEADS
from gpal_lightning.neural_network.tasks.base.heads.head import BaseHead
from gpal_nn.tasks.driving_bev_dyn.losses.loss import DRIVING_BEV_DYNLoss
import torch.nn.functional as F
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.heads.fast_decoder_head import FastDecoderHead
from gpal_nn.models.transformers.od_view_transform import gridcloud3d
from gpal_nn.models.base_modules.basic_henet_module import BasicHENetStageBlock
from gpal_nn.models.base_modules.conv_module import ConvModule2d

class SeqFeatureFuser(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers_config = layers_config

        self.conv_fuser = nn.Sequential(
            nn.Conv2d(self.layers_config["in_channels"], self.layers_config["out_channels"],
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layers_config["out_channels"]), nn.ReLU(True))

    def forward(self, prev_feats, cur_feats, cur2prev):
        x = torch.cat([prev_feats, cur_feats], dim = 1)
        return self.conv_fuser(x)



class TinySEBlock(nn.Module):
    """Block similar to SEBlock but only with one layer of conv2d.

    Args:
        in_channels:  The number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.att(x)


class BevFuseModule(nn.Module):
    """BevFuseModule fuses features using convolutions and SE block.

    Args:
        input_c: The number of input channels.
        fuse_c: The number of channels after fusion.
    """

    def __init__(self, input_c: int, fuse_c: int):
        super().__init__()
        self.reduce_conv = ConvModule2d(
            input_c,
            fuse_c,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(fuse_c, eps=1e-3, momentum=0.01),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv2 = ConvModule2d(
            fuse_c,
            fuse_c,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=nn.BatchNorm2d(fuse_c, eps=1e-3, momentum=0.01),
            act_layer=nn.ReLU(inplace=True),
        )
        self.seblock = TinySEBlock(fuse_c)

    def forward(self, x: torch.Tensor):
        x = self.reduce_conv(x)
        x = self.conv2(x)
        pts_feats = self.seblock(x)
        return pts_feats



@HEADS.register_module()
class DRIVING_BEV_DYNHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=DRIVING_BEV_DYNLoss):
        self.task_config = task_config

        self.dyn_od_head_cfg = self.task_config.Head["dyn_od_head"]
        self.fuser_config = self.dyn_od_head_cfg["fuser_config"] # {"in_channels": 256, "out_channels": 128}
        self.head_config = self.dyn_od_head_cfg["head_config"] # {"in_channels": 128, "num_stages": 6, "out_channels": 21, "upsample": 8}
        self.feature_fuser_config = self.dyn_od_head_cfg.get("feature_fuser_config", None) # {"in_channels": 128, "block_num": 2, "attention_block_num": 2, "mlp_ratio": 4.0, "mlp_ratio_attn": 4.0, "act_layer": "GELU", "use_layer_scale": True, "layer_scale_init_value": 1e-5, "extra_act": False, "block_cls": "BasicHENetStageBlock"}
        
        self.feature_bank = None
        self.prev_metas = None
        self.cnt = 0
        
        transformer_config = copy.deepcopy(global_config.Transformer["transformer_config"])
        self.point_cloud_range = transformer_config["bev_map_range"]
        self.voxel_size = transformer_config["bev_map_voxel_size"]
        self.voxel_size[0] = self.voxel_size[0]
        self.voxel_size[1] = self.voxel_size[1]
        self.grid_size = [
            int(round((self.point_cloud_range[3]-self.point_cloud_range[0])/self.voxel_size[0],2)),
            int(round((self.point_cloud_range[4]-self.point_cloud_range[1])/self.voxel_size[1],2))
        ]  # [H, W] 48, 120 / 96 120 (fisheye)
        
        xyz_camA = gridcloud3d(1, 1, self.grid_size[1], self.grid_size[0], norm=False, device='cpu')
        xyz_camA[:, :, 0] = xyz_camA[:, :, 0] * self.voxel_size[0] + self.voxel_size[0]/2 + self.point_cloud_range[0]
        xyz_camA[:, :, 1] = xyz_camA[:, :, 1] * self.voxel_size[1] + self.voxel_size[1]/2 + self.point_cloud_range[1]
        xyz_camA[:, :, 2] = xyz_camA[:, :, 2] * self.voxel_size[2] + self.voxel_size[2]/2 + self.point_cloud_range[2]
        self.xyz_camA = xyz_camA[:, :, [0, 1, 3], :]
        
        super(DRIVING_BEV_DYNHead, self).__init__(
            global_config, 
            task_config, 
            loss_func
        )

    def _setup(self):
        self.head = nn.ModuleDict()
        self.head["seq_fuser"] = SeqFeatureFuser(self.fuser_config)
        self.head["center_head"] = FastDecoderHead(self.head_config)
        if self.feature_fuser_config is not None:
            self.head["feature_fuser"] = BevFuseModule(
                self.feature_fuser_config["in_channels"],
                self.feature_fuser_config["out_channels"],
            )
    
    def load_state_dict(self, state_dict, strict=True):
        if len(self.head) == 1:
            self.head["center_head"].load_state_dict(state_dict, strict)
        else:
            for head_name, head in self.head.items():
                state_dict_sub = {k.replace(f"{head_name}.", "", 1): state_dict[k]
                                  for k in state_dict if head_name in k}
                head.load_state_dict(state_dict_sub, strict)

    def GetCur2Prev(self, cur_metas, dts):
        B = len(cur_metas)
        rt = torch.zeros([B,3,3], device = "cpu", dtype = torch.float)
        rt[:,0,0] = 1.0
        rt[:,1,1] = 1.0
        rt[:,2,2] = 1.0
        yaw_batch = torch.tensor([ele["ego_yaw_rate"] for ele in cur_metas])
        speed_batch = torch.tensor([ele["ego_speed"] for ele in cur_metas])

        final_yaw = dts * yaw_batch
        speed_vec = final_yaw * 0.5
        pos_x = torch.cos(speed_vec) * speed_batch * dts
        pos_y = torch.sin(speed_vec) * speed_batch * dts

        rt[:,0,0] = torch.cos(final_yaw)
        rt[:,1,1] = torch.cos(final_yaw)
        rt[:,0,1] = -torch.sin(final_yaw)
        rt[:,1,0] = torch.sin(final_yaw)
        rt[:,0,2] = pos_x
        rt[:,1,2] = pos_y

        return rt

    def SeqCheck(self, prev_metas, cur_metas, tth = 0.25):
        B = len(cur_metas)
        seq_flag = torch.zeros(B, device = "cpu", dtype = torch.bool)
        dt = torch.zeros(B, device = "cpu", dtype = torch.float)
        if prev_metas is None:
            return seq_flag, dt

        for i, (m_p, m_c) in enumerate(zip(prev_metas, cur_metas)):
            clip_p = m_p["clip_id"]
            clip_c = m_c["clip_id"]
            ts_p = float(m_p["timestamp"])
            ts_c = float(m_c["timestamp"])

            flag = ((ts_c - ts_p) > 0.0) and ((ts_c - ts_p) < tth) and (clip_p == clip_c)
            seq_flag[i] = flag
            if flag:
                dt[i] = (ts_c - ts_p)
        return seq_flag, dt

    def gen_shift_feature_grid(self, grid, cur2prev, prev_feat, bev_w_resolution, bev_h_resolution):
        _, _, h, w = prev_feat.shape
        bs = grid.shape[0]  # use actual batch size, not prev_feat batch (may differ at last incomplete batch)
        grid = grid.view(bs, h, w, 3, 1)

        for idx in range(bs):
            grid[idx] = cur2prev[idx].matmul(grid[idx])
        
        # bev2feat
        grid_x = (grid[..., 0, 0].clone() - self.point_cloud_range[0]) / bev_w_resolution
        grid_y = (grid[..., 1, 0].clone() - self.point_cloud_range[1]) / bev_h_resolution
        grid[..., 0, 0] = grid_x.clone()
        grid[..., 1, 0] = grid_y.clone()

        # grid = torch.cat([grid_x.clone(), grid_y.clone()], dim = -1).unsqueeze(-1)

        normalize_factor = torch.tensor([w, h],
                                        dtype=prev_feat.dtype,
                                        device=prev_feat.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

        return grid

    def forward(self, x: torch.Tensor, calib=None, metadata=None,point_feature=None) -> torch.Tensor:
        # print(ShowDataStruct("X",x))
        if self.feature_fuser_config is not None and point_feature is not None:
            fuser_feature = torch.cat([x, point_feature[0]], dim = 1)
            fuser_feature = self.head["feature_fuser"](fuser_feature)
        else:
            fuser_feature = x


        if (self.feature_bank == None):
            self.feature_bank = torch.zeros_like(fuser_feature).detach()
        
        B = len(metadata)
        
        if torch.onnx.is_in_onnx_export():
            seq_flag = torch.ones(B, device = "cpu", dtype = torch.bool)
            B = 1
            prev_feats = F.grid_sample(metadata["prev_feats"], metadata["prev_feats_grid"], align_corners=False)
        else:
            seq_flag, dts = self.SeqCheck(self.prev_metas, metadata)
            rts = self.GetCur2Prev(metadata, dts)
            feats_shifted_grid = self.gen_shift_feature_grid(self.xyz_camA.repeat(B, 1, 1, 1).to(fuser_feature.device).clone(), rts.to(fuser_feature.device), self.feature_bank.clone(), self.voxel_size[0], self.voxel_size[1])
            seq_flag = seq_flag.to(fuser_feature.device).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)#.unsqueeze(-1)
            # slice feature_bank to match actual batch size (last incomplete batch may have fewer frames)
            prev_feat_sliced = self.feature_bank[:B].clone()
            feats_shifted = F.grid_sample(prev_feat_sliced,
                                          feats_shifted_grid.to(self.feature_bank.dtype),
                                          align_corners=False)
            prev_feats = feats_shifted.clone() * seq_flag
            
        # self.feature_bank = fuser_feature.detach().clone()
        self.prev_metas = copy.deepcopy(metadata)

        cur2prev = torch.from_numpy(np.eye(3)).to(fuser_feature.device).unsqueeze(0).repeat(fuser_feature.shape[0], 1, 1)
        x_fuser = self.head["seq_fuser"](prev_feats, fuser_feature, cur2prev)
        self.feature_bank = x_fuser.detach().clone()
        x_decode = self.head["center_head"](x_fuser)
        _set = ["reg", "height","dim", "rot", "vel"]
        head_conv = torch.cat([x_decode[k] for k in _set], dim=1)
        batch_dict = {
            'head_conv': head_conv, 
            "hm_cen": x_decode["hm"], 
            "cur_feats": x_fuser.detach().clone()
        }
        
        return [batch_dict]
