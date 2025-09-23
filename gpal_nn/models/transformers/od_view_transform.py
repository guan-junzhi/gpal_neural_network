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
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz


def project_radar_to_image_now(radar, distort_coeffa, rmata, tveca, intrinsica):
    # radar shape: (B, N, 3)
    # distort_coeffa shape: (B, 1, 5)
    # rmata shape: (B, 4, 4)
    # intrinsica shape: (B, 3, 3)
    B, N, _ = radar.shape
    reference_points = torch.cat(
        (radar, torch.ones_like(radar[..., :1])), -1).unsqueeze(-1)  # b,n,4，1
    lidar2cam = rmata.unsqueeze(1).repeat(1, N, 1, 1)  # b,n,4，4
    cam_intrinsics = intrinsica.unsqueeze(1).repeat(1, N, 1, 1)  # b,n,3,3
    eps = 1e-5  # 创建体素遮罩

    # 坐标变换
    reference_points_cam = torch.matmul(lidar2cam.to(torch.float32),
                                        reference_points.to(torch.float32)).permute(0, 1, 3, 2)  # B,N,1,4
    # 透视投影
    # reference_points_cam_t2 = reference_points_cam[..., 0:2] / torch.maximum(
    # reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps) #B,N,1,2
    reference_points_cam_t2 = reference_points_cam[..., 0:2] / (
        eps+reference_points_cam[..., 2:3])  # B,N,1,2
    z = reference_points_cam[..., 2:3].clone()
    x = reference_points_cam_t2[..., [0]]  # B,N,1,1
    y = reference_points_cam_t2[..., [1]]  # B,N,1,1

    # x = x * cam_intrinsics[..., 0:1, 0:1] + y * cam_intrinsics[..., 0:1, 1:2] + cam_intrinsics[..., 0:1, 2:3]
    # y = x * cam_intrinsics[..., 1:2, 0:1] + y * cam_intrinsics[..., 1:2, 1:2] + cam_intrinsics[..., 1:2, 2:3]
    x_new = x * cam_intrinsics[..., 0:1, 0:1] + y * \
        cam_intrinsics[..., 0:1, 1:2] + cam_intrinsics[..., 0:1, 2:3]
    y_new = x * cam_intrinsics[..., 1:2, 0:1] + y * \
        cam_intrinsics[..., 1:2, 1:2] + cam_intrinsics[..., 1:2, 2:3]
    x, y = x_new, y_new

    reference_points_cam_t2_new = torch.cat(
        [x, y], dim=-1).squeeze(-2)  # [1, 2, 6, 80000, 2]

    return reference_points_cam_t2_new, z.squeeze(-1)


def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0

    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_y, grid_x


def unproject_image_to_mem(rgb_camBX, Z, Y, X, BB, scale_tensor=None, xyz_camAX=None, mask=None, batch_dict=None):
    image_crop_config = batch_dict['img_crop_dict']

    B, C, H, W = rgb_camBX[:, 0].shape
    view_num = batch_dict['intrinsic'].shape[1]

    intrinsics = batch_dict['intrinsic'].view(
        BB, 1, view_num, 3, 3).repeat(1, 1, 1, 1, 1).view(B, view_num, 3, 3)
    extrinsics = batch_dict['extrinsic'].view(
        BB, 1, view_num, 4, 4).repeat(1, 1, 1, 1, 1).view(B, view_num, 4, 4)
    cam_distorts = batch_dict['cam_dist'].view(
        BB, 1, view_num, 1, 5).repeat(1, 1, 1, 1, 1).view(B, view_num, 1, 5)

    offset_pixel = image_crop_config['CROP_HeSai_ID4']['CROP_START']
    scale = image_crop_config['CROP_HeSai_ID4']['SCALE']
    div = 8
    a = torch.zeros([B, C, Z, Y, X], dtype = torch.float, device = rgb_camBX.device)

    for i in range(intrinsics.shape[1]):

        intrinsic = intrinsics[:, i]
        extrinsic = extrinsics[:, i]
        cam_distort = cam_distorts[:, i]
        xyz_camA = xyz_camAX.clone()
        rgb_camB = rgb_camBX[:, i]
        B, C, H, W = list(rgb_camB.shape)
        tvec = extrinsic[:, :3, 3].view(B, 1, 1, 3)
        xy_pixB1, z = project_radar_to_image_now(
            xyz_camA, cam_distort, extrinsic, tvec, intrinsic)

        x, y = xy_pixB1[:, :, 0]/scale[i][0]/div, xy_pixB1[:,
                                                           :, 1]/scale[i][0]/div - offset_pixel[i][0]/div

        x_valid = (x > -0.5).bool() & (x < float(W-0.5)).bool()
        y_valid = (y > -0.5).bool() & (y < float(H-0.5)).bool()
        z_valid = (z[:, :, 0] > 0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(
            B, 1, Z, Y, X).float()
        y_pixB, x_pixB = normalize_grid2d(y, x, H, W)
        xyz_pixB = torch.stack([x_pixB, y_pixB], axis=2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z*Y, X, 2])
        values = F.grid_sample(
            rgb_camB, xyz_pixB.float(), align_corners=False)
        # values = values.view(B, C, -1) + scale_tensor[i][...,0]
        values = values.view(B, C, Z, Y, X)
        a += values * valid_mem

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

    def forward(
        self,
        feats: List[Tensor],
        data: Dict,
    ) -> Tensor:
        """Forward bevformer viewtransformer."""

        image_feats_stack = []
        for k in self.input_source:
            for fea in feats[k]:
                image_feats_stack.append(fea)
                B, C, H, W = fea.shape
        feats = torch.stack(image_feats_stack, dim=1)
        feats = feats.reshape(B, -1, C, H, W)
        xyz_camA = self.xyz_camA.clone()

        xyz_camA = xyz_camA.to(feats.device).repeat(feats.shape[0], 1, 1)
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
    
    