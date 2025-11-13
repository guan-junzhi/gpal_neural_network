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
    V, N, _, _ = pts.shape
    reference_points = pts
    ego2img = ego2img.unsqueeze(1).repeat(1, N, 1, 1)  # b,n,4，4
    eps = 1e-5  # 创建体素遮罩

    # 坐标变换
    reference_points_img = torch.matmul(ego2img.to(torch.float32),
                                        reference_points.to(torch.float32)).permute(0, 1, 3, 2)  # B,N,1,4
    # 透视投影
    reference_points_img_uv = reference_points_img[..., 0:2] / (eps+reference_points_img[..., 2:3])  # B,N,1,2

    return reference_points_img_uv, reference_points_img[..., 2].clone()


def GetProjectGridByEgo2Imgs_Fisheye(extrin, distor, intrin, div, sample_pts_3d, image_crop_config):
    V, N, _, _ = sample_pts_3d.shape
    points_homo = sample_pts_3d  # V N 4 1
    
    extrin = extrin.unsqueeze(1).repeat(1, N, 1, 1)  # V N 4 4
    
    points_camera_homo = torch.matmul(extrin.to(torch.float32),
                                      points_homo.to(torch.float32)).permute(0, 1, 3, 2)  # B,N,1,4
    
    points_camera = points_camera_homo[..., :3]
    
    # valid_mask = points_camera[..., 2] > 0
    # if not np.any(valid_mask):
    #     return np.zeros((N, 2)), valid_mask
    
    Xs = points_camera[..., [0]]  # V N 1 1
    Ys = points_camera[..., [1]]
    Zs = points_camera[..., [2]]
    
    Zs = torch.where(Zs > 1e-6, Zs, torch.ones_like(Zs) * 1e-6)
    
    x = Xs / Zs
    y = Ys / Zs
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan(r)  # V N 1 1
    
    # 鱼眼畸变模型 (等距投影 + 畸变)
    # r_distorted = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8 + k5*theta^10)
    
    k1 = distor[..., [0]].unsqueeze(-1).repeat(1, N, 1, 1)  # V N 1 1
    k2 = distor[..., [1]].unsqueeze(-1).repeat(1, N, 1, 1) 
    k3 = distor[..., [2]].unsqueeze(-1).repeat(1, N, 1, 1) 
    k4 = distor[..., [3]].unsqueeze(-1).repeat(1, N, 1, 1) 
    k5 = distor[..., [4]].unsqueeze(-1).repeat(1, N, 1, 1) 
    
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta10 = theta8 * theta2  # V N 1 1
    
    # 畸变后的径向距离 (5项式)
    r_distorted = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8 + k5*theta10)
    
    # 避免除零
    r = torch.clamp(r, min=1e-6)
    
    # 计算畸变后的归一化坐标
    x_distorted = x * (r_distorted / r)
    y_distorted = y * (r_distorted / r)
    
    # 归一化平面 -> 像素坐标
    # 提取内参
    fx = intrin[:, [0], [[0]]].unsqueeze(-1).repeat(1, N, 1, 1)
    fy = intrin[:, [1], [[1]]].unsqueeze(-1).repeat(1, N, 1, 1)
    cx = intrin[:, [0], [[2]]].unsqueeze(-1).repeat(1, N, 1, 1)
    cy = intrin[:, [1], [[2]]].unsqueeze(-1).repeat(1, N, 1, 1)
    
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    # 组合结果
    points_image = torch.cat([u, v], dim=-1)
    # 标记无效点
    # points_image[~valid_mask] = 0
    
    Scale = torch.tensor(image_crop_config['CROP_HeSai_ID4']['SCALE'], device=extrin.device, dtype=extrin.dtype)
    Crop_start = torch.tensor(image_crop_config['CROP_HeSai_ID4']['CROP_START'], device=extrin.device, dtype=extrin.dtype)
    
    Scale = Scale.unsqueeze(-1).unsqueeze(-1).repeat(1, N, 1)
    Crop_start = Crop_start.unsqueeze(-1).unsqueeze(-1).repeat(1, N, 1)
    
    points_image[..., 0] = (points_image[..., 0] / Scale) / div
    points_image[..., 1] = (points_image[..., 1] / Scale - Crop_start ) / div
    
    uvs = points_image
    z = points_camera[..., 2]
    
    return uvs, z

def GetProjectGridByEgo2Imgs(ego2imgs, H, W, div, Z, Y, X, sample_pts_3d):
    sample_pts_3d = sample_pts_3d.repeat(ego2imgs.shape[0], 1, 1, 1)
    uvs, z = project_ego_pts_to_image(sample_pts_3d, ego2imgs)
    WH = torch.tensor(
        [[[[(W-1) * div, (H-1) * div]]]], 
        device=ego2imgs.device,
        dtype=ego2imgs.dtype
    )
    uv_norm = (2.0 * (uvs / WH) - 1.0)
    mask = (z <= 0).unsqueeze(-1).expand_as(uv_norm)
    # 将mask对应位置的uv_norm值设置为-2，无效点设置，取消gridsample后面的乘法
    uv_norm[mask] = -2.0
    valid_mem = (z[:, :, 0] > 0).reshape(ego2imgs.shape[0], Z, Y, X).float()
    uv_norm = uv_norm.reshape(ego2imgs.shape[0], -1, X, 2)  # Z*Y

    return uv_norm, valid_mem

def GetProjectGridByEgo2ImgsFisheye(extrin, distor, intrin, H, W, div, Z, Y, X, sample_pts_3d, image_crop_config):
    V = extrin.shape[0]
    sample_pts_3d = sample_pts_3d.repeat(V, 1, 1, 1)
    uvs, z = GetProjectGridByEgo2Imgs_Fisheye(extrin, distor, intrin, div, sample_pts_3d, image_crop_config)
    WH = torch.tensor(
        [[[[(W-1) * div, (H-1) * div]]]], 
        device=extrin.device,
        dtype=extrin.dtype
    )
    uv_norm = (2.0 * (uvs / WH) - 1.0)
    mask = (z <= 0).unsqueeze(-1).expand_as(uv_norm)
    # 将mask对应位置的uv_norm值设置为-2，无效点设置，取消gridsample后面的乘法
    uv_norm[mask] = -2.0
    valid_mem = (z[:, :, 0] > 0).reshape(V, Z, Y, X).float()
    uv_norm = uv_norm.reshape(V, -1, X, 2)  # Z*Y

    return uv_norm, valid_mem

def unproject_image_to_mem(rgb_camBX, Z, Y, X, BB, image_down_div=None, xyz_camAX=None, subtask_name=None, mask=None, batch_dict=None, image_crop_config=None):
    div = image_down_div

    bev_feature_batch = []
    for i in range(rgb_camBX.shape[0]):
        rgb_camB = rgb_camBX[i]
        V, C, H, W = list(rgb_camB.shape)
        
        if torch.onnx.is_in_onnx_export():
            vt_grid= batch_dict["vt_grid"]
        else:
            
            if subtask_name in ["DRIVING_BEV_DYN_FISHEYE"]:
                extrinsic_matrix = batch_dict['extrinsic'][i]
                distortion_coeffs= batch_dict['cam_dist'][i]
                intrinsic_matrix = batch_dict['intrinsic'][i]
                vt_grid, vt_grid_valid = GetProjectGridByEgo2ImgsFisheye(
                    extrinsic_matrix,
                    distortion_coeffs,
                    intrinsic_matrix,
                    H, W, div, Z, Y, X,
                    xyz_camAX.to(rgb_camBX.device).clone(),
                    image_crop_config=image_crop_config,
                )
            elif subtask_name in ["DRIVING_BEV_DYN"]:
                vt_grid, vt_grid_valid = GetProjectGridByEgo2Imgs(
                    batch_dict["ego2imgs"][i],
                    H, W, div, Z, Y, X,
                    xyz_camAX.to(rgb_camBX.device).clone()
                )
            else:
                raise NotImplementedError(f"subtask_name {subtask_name} is not supported")
        
        values = F.grid_sample(rgb_camB, vt_grid.float(), align_corners=False, padding_mode='zeros')
        bev_feature_batch.append((values).sum(0).view(C*Z, Y, X))
    
    a = torch.stack(bev_feature_batch, dim = 0)
    
    return a


@TRANSFORMERS.register_module()
class ODViewTransformer(BaseModule):
    """The single frame pattern of BevFormerViewTransformer."""

    def __init__(self, global_config, transformer_config, freeze_module: bool = False):
        super(ODViewTransformer, self).__init__(global_config)
        # z_layer_num = 4
        self.input_source = transformer_config["input_source"]
        self.point_cloud_range = transformer_config["bev_map_range"]
        self.voxel_size = transformer_config["bev_map_voxel_size"]
        self.image_down_div = transformer_config["image_down_div"]
        # self.voxel_size[2] = (self.point_cloud_range[5] - self.point_cloud_range[2]) / z_layer_num

        self.grid_size = [int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.voxel_size[0]),
                          int((self.point_cloud_range[4]-self.point_cloud_range[1])/self.voxel_size[1]),
                          int((self.point_cloud_range[5]-self.point_cloud_range[2])/self.voxel_size[2])]

        z_layer_num = self.grid_size[2]
        
        xyz_camA = gridcloud3d(1, self.grid_size[2], self.grid_size[1], self.grid_size[0], norm=False, device='cpu')
        xyz_camA[:, :, 0] = xyz_camA[:, :, 0] * self.voxel_size[0] + self.voxel_size[0]/2 + self.point_cloud_range[0]
        xyz_camA[:, :, 1] = xyz_camA[:, :, 1] * self.voxel_size[1] + self.voxel_size[1]/2 + self.point_cloud_range[1]
        xyz_camA[:, :, 2] = xyz_camA[:, :, 2] * self.voxel_size[2] + self.voxel_size[2]/2 + self.point_cloud_range[2]
        self.xyz_camA = xyz_camA
        
        self.image_crop_config = global_config.Tasks['DRIVING_BEV_DYN']['image_crop_config']
        
        self.subtask_name = global_config.Tasks['DRIVING_BEV_DYN']['SWITCH_SUBTASK']
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(transformer_config["in_channels"] * z_layer_num,
                      transformer_config["out_channels"], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(transformer_config["out_channels"]),
            nn.ReLU(True)
        )


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
            image_down_div=self.image_down_div,
            xyz_camAX=xyz_camA,
            subtask_name=self.subtask_name,
            batch_dict=data,
            image_crop_config=self.image_crop_config
        )
        # B, C*Z, H, W = feat_bev.shape
        # feat_bev = feat_bev.view(
        #     B, -1, H, W)

        feat_bev = self.conv_out(feat_bev)
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