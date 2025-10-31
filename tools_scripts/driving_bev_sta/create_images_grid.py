import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PreproModule(nn.Module):
    def __init__(self):
        super(PreproModule, self).__init__()
        
    def compute_grid(self, K, dist, ori_h, ori_w, dst_h, dst_w, cut_start_h):
        scale = min(dst_h / ori_h, dst_w / ori_w)
        
        # 构建新内参矩阵
        K_new = K.copy()
        
        # 1. 缩放变换
        K_new[:2, :] *= scale
        
        # 2. 裁剪变换
        K_new[1, 2] = K_new[1, 2] - cut_start_h
        
        # 目标最终尺寸
        target_size = (dst_w, dst_h - cut_start_h)  # (宽, 高)
        
        # 计算映射网格
        x, y = np.meshgrid(np.arange(target_size[0]), np.arange(target_size[1]))
        dst_coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3).T
        
        # 反投影到归一化相机坐标
        K_new_inv = np.linalg.inv(K_new.astype(np.float64))
        normalized_coords = K_new_inv[:2, :] @ dst_coords
        
        # 应用畸变校正
        k1, k2, p1, p2, k3 = dist.astype(np.float64)
        
        # 提取归一化坐标
        xn = normalized_coords[0, :]
        yn = normalized_coords[1, :]
        
        # 计算径向距离
        r2 = xn**2 + yn**2
        
        # 计算径向畸变系数
        radial_dist = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
        
        # 计算切向畸变
        x_tangential = 2*p1*xn*yn + p2*(r2 + 2*xn**2)
        y_tangential = p1*(r2 + 2*yn**2) + 2*p2*xn*yn
        
        # 应用畸变
        xd = xn * radial_dist + x_tangential
        yd = yn * radial_dist + y_tangential
        
        # 投影回原始图像像素坐标
        distorted_coords = K[:2, :2] @ np.vstack([xd, yd]) + K[:2, 2:]
        x_distorted = distorted_coords[0, :].reshape(target_size[1], target_size[0])
        y_distorted = distorted_coords[1, :].reshape(target_size[1], target_size[0])
        
        # 创建网格 (归一化到[-1, 1])
        grid_x = (2.0 * x_distorted / (ori_w - 1) - 1).astype(np.float32)
        grid_y = (2.0 * y_distorted / (ori_h - 1) - 1).astype(np.float32)
        
        # 创建网格张量
        grid = torch.stack([
            torch.from_numpy(grid_x),
            torch.from_numpy(grid_y)
        ], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        return grid
    
    def export_input(self, batch_size, K1, dists1, K2, dists2, ori_shape, dst_h, dst_w, cut_start_h):
        ori_h, ori_w = ori_shape
        self.grid_30 = self.compute_grid(K1, dists1, ori_h, ori_w, dst_h, dst_w, cut_start_h)
        grid_30 = self.grid_30.repeat(batch_size, 1, 1, 1) 
        self.grid_120 = self.compute_grid(K2, dists2, ori_h, ori_w, dst_h, dst_w, cut_start_h)
        grid_120 = self.grid_120.repeat(batch_size, 1, 1, 1) 
        return torch.cat([grid_30, grid_120], dim=0)
        
    def forward(self, img, grid):
        # 使用grid_sample进行采样
        result_tensor = F.grid_sample(
            img,
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        )
        result_tensor = result_tensor / 255.0
        
        return result_tensor


if __name__ == "__main__":
    # 测试PreproModule
    import pickle; import cv2; import os
    ds = pickle.load(open('/data/ai_group/datasets/multiview_lane_det/lane_pkl/1f_from_20250906_split_merge_val.pkl', 'rb'))
    sample = ds[0]
    k1, d1 = sample['sensor']['img_front_30']['intr']['K'], sample['sensor']['img_front_30']['intr']['dist']
    k2, d2 = sample['sensor']['img_front_120']['intr']['K'], sample['sensor']['img_front_120']['intr']['dist']
    im_folder = '/data/dp_group/process-prod-bucket/data_collect/'
    im1_path = sample['sensor']['img_front_30']['img_path']
    im2_path = sample['sensor']['img_front_120']['img_path']

    batch_size = 1
    ori_h, ori_w = 2160, 3840
    dst_h, dst_w = 540, 960
    cut_start_h = 28

    img = [cv2.imread(os.path.join(im_folder, path)) for path in [im1_path, im2_path]]
    img = torch.from_numpy(np.stack(img, axis=0)).permute(0, 3, 1, 2).float()  # [2, 3, H, W]

    prepro_module = PreproModule()
    grid = prepro_module.export_input(batch_size, k1, d1, k2, d2, (ori_h, ori_w), dst_h, dst_w, cut_start_h)
    # output = prepro_module(img, grid)
    # np.save("abel/npy/0_images_grid.npy", grid.cpu().numpy())
    print("grid shape:", grid.shape)  #  [batch_size*2, 3, dst_h - cut_start_h, dst_w]