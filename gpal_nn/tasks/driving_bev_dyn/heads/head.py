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
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf
import cv2
from gpal_nn.models.transformers.od_view_transform import gridcloud3d
from torch import distributed

class SeqFeatureFuser(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers_config = layers_config

        self.conv_fuser = nn.Sequential(
            nn.Conv2d(
            self.layers_config["in_channels"], self.layers_config["out_channels"], kernel_size=3, stride=1, padding=1, bias=False
        ), 
        nn.BatchNorm2d(self.layers_config["out_channels"]), nn.ReLU(True))

    def forward(self, prev_feats, cur_feats, cur2prev):
        x = torch.cat([cur_feats, cur_feats], dim = 1)
        return self.conv_fuser(x)


@HEADS.register_module()
class DRIVING_BEV_DYNHead(BaseHead):
    def __init__(self, global_config, task_config, loss_func=DRIVING_BEV_DYNLoss):
        self.task_config = task_config
        self.is_track_task = True  # 区分当前是否是Track任务
        self.head_conv = 64

        self.fuser_config = {"in_channels": 1024, "out_channels": 512}
        self.head_config = {"in_channels": 512,
                            "num_stages": 6, "out_channels": 21, "upsample": 4}

        self.feature_bank = None
        self.feature_zeros = None
        self.prev_metas = None
        self.cnt = 0
        
        transformer_config = global_config.Transformer["transformer_config"]
        self.point_cloud_range = transformer_config["bev_map_range"]
        self.voxel_size = transformer_config["bev_map_voxel_size"]
        
        self.grid_size = [int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.voxel_size[0]),
                          int((
                              self.point_cloud_range[4]-self.point_cloud_range[1])/self.voxel_size[1])]
        
        xyz_camA = gridcloud3d(
            1, 1, self.grid_size[1], self.grid_size[0], norm=False, device='cpu')
        xyz_camA[:, :, 0] = xyz_camA[:, :, 0] * self.voxel_size[0] + \
            self.voxel_size[0]/2 + self.point_cloud_range[0]
        xyz_camA[:, :, 1] = xyz_camA[:, :, 1] * self.voxel_size[1] + \
            self.voxel_size[1]/2 + self.point_cloud_range[1]
        xyz_camA[:, :, 2] = xyz_camA[:, :, 2] * self.voxel_size[2] + \
            self.voxel_size[2]/2 + self.point_cloud_range[2]
        self.xyz_camA = xyz_camA[:,:,[0,1,3],:]
        super(DRIVING_BEV_DYNHead, self).__init__(
            global_config, task_config, loss_func)

    def _setup(self):
        self.head = nn.ModuleDict()
        self.head["fuser"] = SeqFeatureFuser(self.fuser_config)
        self.head["center_head"] = FastDecoderHead(self.head_config)
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

    def shift_feature(self, grid, cur2prev, prev_feat, bev_h_resolution, bev_w_resolution):
        bs, _, h, w = prev_feat.shape
        grid = grid.view(bs, h, w, 3, 1)
        if torch.onnx.is_in_onnx_export():
            grid = cur2prev.matmul(grid)
        else:
            for idx in range(bs):
                grid[idx] = cur2prev[idx].matmul(grid[idx])
        # bev2feat
        grid_x = (grid[..., 0, 0].clone() - self.point_cloud_range[0]) / bev_w_resolution
        grid_y = (grid[..., 1, 0].clone() - self.point_cloud_range[1]) / bev_h_resolution
        grid[..., 0, 0] = grid_x.clone()
        grid[..., 1, 0] = grid_y.clone()
        # todo 需要仔细分辨一下应该用哪个
        if True:
            normalize_factor = torch.tensor([w, h],
                                            dtype=prev_feat.dtype,
                                            device=prev_feat.device)
            grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
            output = F.grid_sample(prev_feat, grid.to(
                prev_feat.dtype), align_corners=False)
        else:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                            dtype=prev_feat.dtype,
                                            device=prev_feat.device)
            grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
            output = F.grid_sample(prev_feat, grid.to(
                prev_feat.dtype), align_corners=True)
        return output


    def forward(self, x: torch.Tensor, calib=None, metadata=None) -> torch.Tensor:
        if (self.feature_zeros == None) or (self.feature_bank == None):
            self.feature_zeros = torch.zeros_like(x).detach().clone()
            self.feature_bank = torch.zeros_like(x).detach().clone()

        seq_flag, dts = self.SeqCheck(self.prev_metas, metadata)
        rts = self.GetCur2Prev(metadata, dts)
        feats_shifted = self.shift_feature(self.xyz_camA.repeat(8, 1, 1, 1).to(x.device).clone(), rts.to(x.device), self.feature_bank.clone(), self.voxel_size[0], self.voxel_size[1])
        seq_flag = seq_flag.to(x.device).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        prev_feats = feats_shifted.clone() * seq_flag + (1-seq_flag) * self.feature_zeros
        self.feature_bank = x.detach().clone()
        self.prev_metas = copy.deepcopy(metadata)

        cur2prev = torch.from_numpy(np.eye(3)).to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x_fuser = self.head["fuser"](prev_feats, x, cur2prev)
        x_decode = self.head["center_head"](x_fuser)
        batch_dict = {'head_conv': x_decode[:, 6:], "hm_cen": x_decode[:, :6]}
        return [batch_dict]
