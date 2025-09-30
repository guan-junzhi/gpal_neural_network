# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from gpal_lightning.neural_network.network_modules.transformers.builder import TRANSFORMERS
from gpal_lightning.neural_network.network_modules.base_module import BaseModule
from tools_scripts.data_format_cvt import ShowDataStruct


def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z, torch.ones_like(z)], dim=2).unsqueeze(-1)
    # this is B x N x 3
    return xyz

def project_ego_pts_to_image(pts, ego2img):
    B, N, _, _ = pts.shape
    reference_points = pts
    ego2img = ego2img.unsqueeze(1).repeat(1, N, 1, 1)  # b,n,4，4
    eps = 1e-5  # 创建体素遮罩

    # 坐标变换
    reference_points_img = torch.matmul(ego2img.to(torch.float32),
                                        reference_points.to(torch.float32)).permute(0, 1, 3, 2)  # B,N,1,4
    # 透视投影
    reference_points_img_uv = reference_points_img[..., 0:2] / (
        eps+reference_points_img[..., 2:3])  # B,N,1,2

    return reference_points_img_uv, reference_points_img[..., 2].clone()


def GetProjectGridByEgo2Imgs(ego2imgs, H, W, div, Z, Y, X, sample_pts_3d):
    sample_pts_3d = sample_pts_3d.repeat(ego2imgs.shape[0], 1, 1, 1)
    uvs, z = project_ego_pts_to_image(sample_pts_3d, ego2imgs)
    WH = torch.tensor([[[[(W-1) * div, (H-1) * div]]]], device=ego2imgs.device,
                      dtype=ego2imgs.dtype)
    uv_norm = (2.0 * (uvs / WH) - 1.0)
    valid_mem = (z[:, :, 0] > 0).reshape(ego2imgs.shape[0], Z, Y, X).float()
    uv_norm = uv_norm.reshape(ego2imgs.shape[0], -1, X, 2)

    return uv_norm, valid_mem

def unproject_image_to_mem(rgb_camBX, Z, Y, X, BB, scale_tensor=None, xyz_camAX=None, mask=None, batch_dict=None, image_crop_config=None):
    div = 8

    bev_feature_batch = []
    for i in range(rgb_camBX.shape[0]):
        rgb_camB = rgb_camBX[i]
        V, C, H, W = list(rgb_camB.shape)
        if torch.onnx.is_in_onnx_export():
            vt_grid, vt_grid_valid = batch_dict["vt_grid"], batch_dict["vt_grid_valid"]
        else:
            vt_grid, vt_grid_valid = GetProjectGridByEgo2Imgs(
                batch_dict["ego2imgs"][i], H, W, div, Z, Y, X, xyz_camAX.to(rgb_camBX.device).clone())
        # print(ShowDataStruct("vt_grid", vt_grid))
        # print(ShowDataStruct("vt_grid_valid", vt_grid_valid))
        
        vt_grid_valid = vt_grid_valid.unsqueeze(1)
        values = F.grid_sample(
            rgb_camB, vt_grid.float(), align_corners=False, padding_mode='zeros')
        values = values.view(V, C, Z, Y, X)
        bev_feature_batch.append((values * vt_grid_valid).sum(0))
    a = torch.stack(bev_feature_batch, dim = 0)
    return a


@TRANSFORMERS.register_module()
class ODViewTransformer(BaseModule):
    """The single frame pattern of BevFormerViewTransformer."""

    def __init__(self, global_config, transformer_config, freeze_module: bool = False):
        super(ODViewTransformer, self).__init__(global_config)
        self.input_source = transformer_config["input_source"]
        self.point_cloud_range = transformer_config["bev_map_range"]
        self.voxel_size = transformer_config["bev_map_voxel_size"]
        self.voxel_size[2] = (self.point_cloud_range[5] -
                              self.point_cloud_range[2])/4

        self.grid_size = [int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.voxel_size[0]),
                          int((
                              self.point_cloud_range[4]-self.point_cloud_range[1])/self.voxel_size[1]),
                          int((self.point_cloud_range[5]-self.point_cloud_range[2])/self.voxel_size[2])]

        xyz_camA = gridcloud3d(
            1, self.grid_size[2], self.grid_size[1], self.grid_size[0], norm=False, device='cpu')
        xyz_camA[:, :, 0] = xyz_camA[:, :, 0] * self.voxel_size[0] + \
            self.voxel_size[0]/2 + self.point_cloud_range[0]
        xyz_camA[:, :, 1] = xyz_camA[:, :, 1] * self.voxel_size[1] + \
            self.voxel_size[1]/2 + self.point_cloud_range[1]
        xyz_camA[:, :, 2] = xyz_camA[:, :, 2] * self.voxel_size[2] + \
            self.voxel_size[2]/2 + self.point_cloud_range[2]

        self.xyz_camA = xyz_camA
        self.image_crop_config = global_config.Tasks['DRIVING_BEV_DYN']['image_crop_config']

    def forward(
        self,
        feats: List[Tensor],
        data: Dict,
    ) -> Tensor:
        """Forward bevformer viewtransformer."""

        if torch.onnx.is_in_onnx_export():
            feats = feats[0].unsqueeze(0)
            B = 1
        else:
            image_feats_stack = []
            for k in self.input_source:
                for fea in feats[k]:
                    image_feats_stack.append(fea)
                    B, C, H, W = fea.shape
            feats = torch.stack(image_feats_stack, dim=1)
            feats = feats.reshape(B, -1, C, H, W)
        xyz_camA = self.xyz_camA.clone()
        
        feat_bev = unproject_image_to_mem(
            feats,
            self.grid_size[2],
            self.grid_size[1],
            self.grid_size[0],
            B,
            # scale_tensor=[input_1, input_2, input_3,
            #               input_4, input_5, input_6, input_7],
            xyz_camAX=xyz_camA,
            mask=None,
            batch_dict=data,
            image_crop_config=self.image_crop_config
        )
        B, C, Z, H, W = feat_bev.shape
        feat_bev = feat_bev.view(
            B, -1, H, W)

        # exit(1)
        return feat_bev


if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("ODViewTransformer.pkl", 'rb'))
    vt = ODViewTransformer(*inputs)
    inputs = pkl.load(open("ODViewTransformer_inputs.pkl", 'rb'))

    print(ShowDataStruct("inputs", inputs))

    time_dp = DetailProf()
    time_dp.Tic("begin")
            
    y = vt(*inputs)
    time_dp.Duration("vt", "begin")

    time_dp.Print()


    print(ShowDataStruct("y", y))
    
    