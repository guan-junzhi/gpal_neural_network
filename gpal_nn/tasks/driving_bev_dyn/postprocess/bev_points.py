import math
import numpy as np
import torch
import torch.nn as nn
import random
# from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
import time
from tools_scripts.data_format_cvt import ShowDataStruct


def get_feature_of_key_points_indice(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    ans = im[y0, x0].view(-1)
    return ans


class Bev_To_Points(nn.Module):
    def __init__(self,
                 model_cfg,
                 **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = self.model_cfg['VOXEL_SIZE']
        self.point_cloud_range = self.model_cfg['OD_RANGE']
        self.num_key_points = self.model_cfg['NUM_KEYPOINTS']

    def get_sampled_points_and_gather_matched_features(self, batch_dict, batch_size, mode_gt, mode_pred, features):

        B, C, H, W = batch_dict['hm_cen'].shape

        if "curr" in mode_pred:
            hm_pred = batch_dict['hm_cen'].view(batch_size, C, -1)
        else:
            raise NotImplementedError

        score, _ = hm_pred.max(dim=1)  # 帧预测的热力图的通道最大值

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W) * self.voxel_size[1] + self.point_cloud_range[1]
        xs = xs.view(1, H, W) * self.voxel_size[0] + self.point_cloud_range[0]
        xys = torch.cat([ys, xs], dim=0).view(1, 2, -1).to(hm_pred).repeat(B, 1, 1)

        _, indice_topk = torch.topk(score, k=self.num_key_points, dim=-1)
        indice_topk = indice_topk.view(B, -1)

        features_topk = features.view(B, features.shape[1], -1)[
            torch.arange(B)[:, None, None],
            torch.arange(features.shape[1])[None, :, None],
            indice_topk.reshape(B, 1, -1).repeat(1, features.shape[1], 1)]

        xys_topk = xys[torch.arange(B)[:, None, None],
                       torch.arange(xys.shape[1])[None, :, None],
                       indice_topk.reshape(B, 1, -1).repeat(1, xys.shape[1], 1)]

        score_topk = score[torch.arange(B)[:, None], indice_topk]

        # 置信度得分
        if "curr" in mode_pred:
            batch_dict['pred_curr_track_score'] = score_topk.reshape(batch_size, self.num_key_points, 1)
            batch_dict['pred_curr_track_point_features'] = features_topk.permute(0, 2, 1).reshape(batch_size, self.num_key_points, -1)  # -> N, 256, D
            batch_dict['pred_curr_track_point_coords'] = xys_topk.permute(0, 2, 1).reshape(batch_size, self.num_key_points, -1)  # (BxN, 2)
            batch_dict['pred_curr_track_point_idx'] = indice_topk

            hm = batch_dict['hm_cen'].reshape(batch_size, C, -1)
            score_raw_topk = hm[torch.arange(B)[:, None, None],
                                torch.arange(hm.shape[1])[None, :, None],
                                indice_topk.reshape(B, 1, -1).repeat(1, hm.shape[1], 1)
            ]
            batch_dict['score'] = score_raw_topk.reshape(batch_size, -1, 1, self.num_key_points)  # -> [1, C, 1, 256]

        else:
            raise NotImplementedError

        return batch_dict

    def forward(self, batch_dict):

        batch_size = int(batch_dict['head_conv'].shape[0])

        batch_dict = self.get_sampled_points_and_gather_matched_features(batch_dict,
                                                                         batch_size,
                                                                         mode_gt="gt_curr_",
                                                                         mode_pred="pred_curr_",
                                                                         features=batch_dict['head_conv'])

        pred_curr_track_point_features = batch_dict['pred_curr_track_point_features'].permute(0, 2, 1)  # -> B 15 256
        pred_curr_track_score = batch_dict['pred_curr_track_score'].permute(0, 2, 1)

        estimation_cen = pred_curr_track_point_features[:,:2,:] #  -> B C 256
        estimation_z = pred_curr_track_point_features[:,2:3,:]
        estimation_dim = pred_curr_track_point_features[:,3:6,:]
        estimation_dir = pred_curr_track_point_features[:,6:12,:]
        estimation_vel = pred_curr_track_point_features[:,12:14,:]
        # estimation_score = pred_curr_track_point_features[:,10:11,:]

        template_xyz = torch.cat([batch_dict['pred_curr_track_point_coords'][:, :, :2],
                                  batch_dict['pred_curr_track_score']], dim=-1)  # -> [B, 256, 3]
        
        # _, label = batch_dict['score'].max(dim=1)  # 原始的score含通道
        
        # batch_dict['batch_pred_labels'] = label.view(batch_dict['score'].shape[0], -1) + 1
        
        batch_dict['Points_Loss'] = {
            'estimation_cen': estimation_cen + template_xyz.permute(0, 2, 1)[:,0:2,:].flip(1),  # 函数内是ys,xs
            'estimation_z': estimation_z,
            'estimation_dim': estimation_dim,
            'estimation_dir': estimation_dir,
            'estimation_vel': estimation_vel,
            'estimation_score': pred_curr_track_score,
            'estimation_score_cls': batch_dict['score'].squeeze(2),
        }

        if torch.onnx.is_in_onnx_export():
            return {k: batch_dict['Points_Loss'][k].squeeze(0) for k in batch_dict['Points_Loss']}
        else:
            return batch_dict
