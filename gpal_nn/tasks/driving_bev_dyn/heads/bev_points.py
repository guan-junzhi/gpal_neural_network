import math
import numpy as np
import torch
import torch.nn as nn
import random
# from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
import time


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
                 grid_size,
                 voxel_size,
                 point_cloud_range,
                 num_bev_features=None,
                 **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.voxel_size = voxel_size  # 来自于 det3dtemplate
        self.voxel_size = self.model_cfg['VOXEL_SIZE']  # 来自于 yaml
        self.point_cloud_range = point_cloud_range
        self.num_point_features = self.model_cfg['NUM_OUTPUT_FEATURES']
        self.num_point_features_before_fusion = self.model_cfg['NUM_BEV_FEATURES']
        self.num_key_points = self.model_cfg['NUM_KEYPOINTS']
        self.training = self.model_cfg['TRAIN']
        self.score_thresh = self.model_cfg['SCORE_THRESH']
        self.down_ratio = self.model_cfg['DOWN_RATIO']

    def get_sampled_points_and_gather_matched_features(self, batch_dict, batch_size, mode_gt, mode_pred, features):

        B, _, C, H, W = batch_dict['hm_cen_pred'].shape

        # 为了不让hm空
        # if self.training:
        #     # 通过 mode_gt 切换
        #     hm_gt = batch_dict[mode_gt + 'hm_cen'].view(batch_size, C, -1)
        # else:  # 测试不需要gt热力
        #     if "prev" in mode_pred:
        #         hm_gt = batch_dict['hm_cen_pred'][:, 1].view(
        #             batch_size, C, -1)  # 经过 maxpool 和 == [1, 4, 96, 240]
        #     elif "curr" in mode_pred:
        #         hm_gt = batch_dict['hm_cen_pred'][:, 0].view(batch_size, C, -1)
        #     else:
        #         raise NotImplementedError

        if "prev" in mode_pred:
            hm_pred = batch_dict['hm_cen_pred'][:, 1].view(
                batch_size, C, -1)  # 经过 maxpool 和 == [1, 4, 23040]
        elif "curr" in mode_pred:
            hm_pred = batch_dict['hm_cen_pred'][:, 0].view(batch_size, C, -1)
        else:
            raise NotImplementedError

        score, _ = hm_pred.max(dim=1)  # 帧预测的热力图的通道最大值
        # score_gt = hm_gt

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W) * self.voxel_size[1] + self.point_cloud_range[1]
        xs = xs.view(1, H, W) * self.voxel_size[0] + self.point_cloud_range[0]
        xys = torch.cat([ys, xs], dim=0).view(
            1, 2, -1).to(hm_pred).repeat(B, 1, 1)
        # print(self.voxel_size,self.point_cloud_range )

        _, indice_topk = torch.topk(score, k=256, dim=-1)
        indice_topk = indice_topk.view(B, -1)

        features_topk = features.view(B, features.shape[1], -1)[torch.arange(B)[:, None, None],
                                                                torch.arange(features.shape[1])[
            None, :, None],
            indice_topk.reshape(B, 1, -1).repeat(1, features.shape[1], 1)]

        xys_topk = xys[torch.arange(B)[:, None, None],
                       torch.arange(xys.shape[1])[None, :, None],
                       indice_topk.reshape(B, 1, -1).repeat(1, xys.shape[1], 1)]

        score_topk = score[torch.arange(B)[:, None], indice_topk]
        # score_gt_topk = score_gt[torch.arange(B)[:, None, None],
        #                          torch.arange(score_gt.shape[1])[
        #     None, :, None],
        #     indice_topk.reshape(B, 1, -1).repeat(1, score_gt.shape[1], 1)]

        # c_input_topk = torch.cat([xys_topk, score_topk, features_topk], dim=1)

        # 置信度得分
        if "curr" in mode_pred:
            batch_dict['pred_curr_track_score'] = score_topk.view(
                batch_size, self.num_key_points, 1)
            batch_dict['pred_curr_track_point_features'] = features_topk.permute(
                0, 2, 1).view(batch_size, -1, 64)  # -> N, 256, 64
            batch_dict['pred_curr_track_point_coords'] = xys_topk.permute(
                0, 2, 1).view(batch_size, self.num_key_points, -1)  # (BxN, 4)
            # batch_dict['score'] = score_gt_topk.view(
            #     batch_size, -1, 1, self.num_key_points)  # -> [1, 4, 1, 256]  # 只是用当前帧的

        elif "prev" in mode_pred:
            batch_dict['pred_prev_score'] = score_topk.view(
                batch_size, self.num_key_points, 1)
            batch_dict['pred_prev_point_features'] = features_topk.permute(
                0, 2, 1).view(batch_size, self.num_key_points, -1)  # -> N, 256, 64
            batch_dict['pred_prev_point_coords'] = xys_topk.permute(
                0, 2, 1).view(batch_size, self.num_key_points, -1)  # -> N, 256, 2
        else:
            raise NotImplementedError

        # if self.training and mode_gt == "gt_curr_":
        #     for bs_idx in range(batch_size):
        #         indice_topk_single = indice_topk[bs_idx]
        #         gt_mask = batch_dict[mode_gt + 'obj_mask'][bs_idx]
        #         track_ind = batch_dict[mode_gt +
        #                                'indices_center'][bs_idx].clone()
        #         for idx in range(self.num_key_points):
        #             if gt_mask[idx] == 0:   # 当前帧没有真值点，跳过
        #                 continue
        #             # 有真值点，但没有预测匹配上
        #             if (indice_topk_single == track_ind[idx]).sum() < 1:
        #                 batch_dict[mode_gt + 'obj_mask'][bs_idx][idx] = 0  #
        #                 batch_dict[mode_gt + 'indices_center'][bs_idx][idx] = 0
        #             else:  # 有真值点，有预测匹配上
        #                 key_mask = indice_topk_single == track_ind[idx]
        #                 key_range = torch.arange(
        #                     0, self.num_key_points, device=indice_topk_single.device)[key_mask]
        #                 batch_dict[mode_gt +
        #                            'indices_center'][bs_idx][idx] = key_range[0]

        return batch_dict

    def forward(self, batch_dict):

        batch_size = int(batch_dict['head_conv'].shape[0] / 2)

        spatial_features_2d_size = batch_dict['head_conv'].size()
        spatial_features_2d = batch_dict['head_conv'].view(batch_size, 2,
                                                           spatial_features_2d_size[1],
                                                           spatial_features_2d_size[2],
                                                           spatial_features_2d_size[3]
                                                           )

        spatial_features_2d_curr = spatial_features_2d[:, 0]
        spatial_features_2d_prev = spatial_features_2d[:, 1]

        batch_dict = self.get_sampled_points_and_gather_matched_features(batch_dict,
                                                                         batch_size,
                                                                         mode_gt="gt_curr_",
                                                                         mode_pred="pred_curr_",
                                                                         features=spatial_features_2d_curr)

        batch_dict = self.get_sampled_points_and_gather_matched_features(batch_dict,
                                                                         batch_size,
                                                                         mode_gt="gt_prev_",
                                                                         mode_pred="pred_prev_",
                                                                         features=spatial_features_2d_prev)

        return batch_dict
