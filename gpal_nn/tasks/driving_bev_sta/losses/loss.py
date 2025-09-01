import torch
import numpy as np
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.losses.loss import BaseLoss
from gpal_lightning.neural_network.tasks.builder import LOSSES
from gpal_nn.tasks.driving_bev_sta.losses.transform_gt import transform_gt_box, shift_polyline_points, shift_polygen_points
from gpal_nn.tasks.driving_bev_sta.losses.map_loss import BaseMapLossCost
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *
import time
from tools_scripts.data_format_cvt import ShowDataStruct
import pickle as pkl


def pack_polyline_gt_points(data):
    annos = []
    classes = []
    shape_types = []
    if 'points' in data['polylines']:
        annos.append(data['polylines']['points'])
        shape_types.append(data['polylines']['shape_type'])
        classes.append(np.ones(len(data['polylines']['points'])) * main_class_type_map['lane_marking'])
    if 'points' in data['edges']:
        annos.append(data['edges']['points'])
        shape_types.append(np.ones(len(data['edges']['points'])) * (-1))
        classes.append(np.ones(len(data['edges']['points'])) * main_class_type_map['edge'])

    if len(annos) > 0:
        annos = np.concatenate(annos, axis=0)
        classes = np.concatenate(classes, axis=0)
        shape_types = np.concatenate(shape_types, axis=0)
    return annos, classes, shape_types

def lane_loss_computation(preds, data, loss_func):

    # print(data[0])

    # print(ShowDataStruct("preds", preds))
    # print(ShowDataStruct("data", data))
    # preds1, data1 = pkl.load(open("../wangtong_loss.pkl", 'rb'))
    # print(ShowDataStruct("preds", preds))
    # print(ShowDataStruct("data", data))
    # bev_embed, all_cls_scores, all_bbox_pred, all_pts_pred = \
    #     preds['bev_embed'], preds['all_cls_scores'], preds['all_bbox_pred'], preds['all_pts_pred']
    bev_embed, all_cls_scores, all_bbox_pred, all_pts_pred, all_shape_types_pred = \
            None, preds['all_cls_scores'], preds['all_bbox_preds'], preds['all_pts_preds'], preds['all_shape_types_preds']
    # num_iter_layer, bs, num_query, score shape
    num_iter, bs, _, pts_per_vector, _ = all_pts_pred.shape

    # print(data['annot'][0])

    loss_list = list()
    for k in range(num_iter):
        loss_dict = dict()
        for j in range(bs):

            time_dp = DetailProf()
            time_dp.Tic("begin")

            score_pred, bbox_pred, pts_pred, shape_type_pred = all_cls_scores[k, j], all_bbox_pred[k, j], all_pts_pred[k, j], all_shape_types_pred[k, j]
            #  [n, 2], [n, 4], [n, 20, 2]
            subdata = data[j]
            # subdata = data['annot'][j]
            annos, classes, shape_types = pack_polyline_gt_points(subdata)
            time_dp.Duration("lane_loss_computation_all_1", "begin")

            start_x = 120
            start_y = 16
            # gt ploylines to gt bboxes  [n, 4], [n, 20, 2]
            bboxes_gt, points_gt = transform_gt_box(annos, start_x, start_y,
                                                    num_pts_per_vec=pts_per_vector, y_first=False, device=pts_pred.device)

            time_dp.Duration("lane_loss_computation_all_2",
                             "lane_loss_computation_all_1")

            # [n,20, 2]->[n, 2, 20, 2]  矢量线翻转建模
            points_gt = shift_polyline_points(points_gt, pts_per_vector)

            time_dp.Duration("lane_loss_computation_all_3",
                             "lane_loss_computation_all_2")
            single_loss_dict = loss_func(
                (score_pred, bbox_pred, pts_pred, shape_type_pred), (classes, bboxes_gt, points_gt, shape_types))

            time_dp.Duration("lane_loss_computation_all_4",
                             "lane_loss_computation_all_3")
            for key in single_loss_dict.keys():
                if key in loss_dict.keys():
                    loss_dict[key] += single_loss_dict[key]
                else:
                    loss_dict[key] = single_loss_dict[key]
            time_dp.Duration("lane_loss_computation_all_5",
                             "lane_loss_computation_all_4")

            time_dp.Duration("lane_loss_computation_all", "begin")
            # time_dp.Print()

        for key in loss_dict.keys():
            loss_dict[key] /= bs

        loss_list.append(loss_dict)
    total_dict = {}
    final_total_loss = 0.0
    for k in range(num_iter):
        d_loss_dict = loss_list[k]
        for key in d_loss_dict.keys():
            total_dict[f"lane_d{k}.{key}"] = d_loss_dict[key]

    for key in total_dict.keys():
        if "loss" in key:
            final_total_loss += total_dict[key]
    total_dict['total_loss'] = final_total_loss
    return total_dict


def lane_loss_computation2(preds, data, loss_func):
    # bev_embed, all_cls_scores, all_bbox_pred, all_pts_pred = \
    #     preds['bev_embed'], preds['all_cls_scores'], preds['all_bbox_pred'], preds['all_pts_pred']
    bev_embed, all_cls_scores, all_bbox_pred, all_pts_pred = \
        None, preds['all_cls_scores'], preds['all_bbox_preds'], preds['all_pts_preds']
    # num_iter_layer, bs, num_query, score shape
    num_iter, bs, _, pts_per_vector, _ = all_pts_pred.shape

    loss_list = [dict() for k in range(num_iter)]
    # loss_dict = dict()
    for j in range(bs):
        annos = pack_polyline_gt_points(data[j])
        start_x = 120
        start_y = 16
        # gt ploylines to gt bboxes  [n, 4], [n, 20, 2]
        bboxes_gt, points_gt = transform_gt_box(
            annos, start_x, start_y, num_pts_per_vec=pts_per_vector, y_first=False, device=all_cls_scores.device)
        # [n,20, 2]->[n, 2, 20, 2]  矢量线翻转建模
        points_gt = shift_polyline_points(points_gt, pts_per_vector)

        for k in range(num_iter):
            loss_dict = loss_list[k]

            time_dp = DetailProf()
            time_dp.Tic("begin")

            score_pred, bbox_pred, pts_pred = all_cls_scores[k,
                                                             j], all_bbox_pred[k, j], all_pts_pred[k, j]

            time_dp.Duration("lane_loss_computation_all_2",
                             "begin")

            # here to loss
            single_loss_dict = loss_func(
                (score_pred, bbox_pred, pts_pred), (bboxes_gt, points_gt))

            time_dp.Duration("lane_loss_computation_all_4",
                             "lane_loss_computation_all_2")
            for key in single_loss_dict.keys():
                if key in loss_dict.keys():
                    loss_dict[key] += single_loss_dict[key] / bs
                else:
                    loss_dict[key] = single_loss_dict[key] / bs
            time_dp.Duration("lane_loss_computation_all_5",
                             "lane_loss_computation_all_4")

            time_dp.Duration("lane_loss_computation_all", "begin")
            # time_dp.Print()

        # for key in loss_dict.keys():
        #     loss_dict[key] /= bs

        # loss_list.append(loss_dict)
    total_dict = {}
    final_total_loss = 0.0
    for k in range(num_iter):
        d_loss_dict = loss_list[k]
        for key in d_loss_dict.keys():
            total_dict[f"lane_d{k}.{key}"] = d_loss_dict[key]

    for key in total_dict.keys():
        if "loss" in key:
            final_total_loss += total_dict[key]
    total_dict['total_loss'] = final_total_loss
    return total_dict


def loss_computation(preds, data, loss_func):

    total_dict = {}
    time_dp = DetailProf()
    time_dp.Tic("begin")
    lane_total_dict1 = lane_loss_computation(preds, data, loss_func)
    time_dp.Duration("lane_loss_computation1", "begin")
    # lane_total_dict2 = lane_loss_computation2(preds, data, loss_func)
    time_dp.Duration("lane_loss_computation2", "lane_loss_computation1")
    # time_dp.Print()
    # print(preds['all_cls_scores'].device)

    total_dict.update(lane_total_dict1)

    # print(lane_total_dict1)
    # print(lane_total_dict2)

    # # for k in lane_total_dict1:
    # for k in ['total_loss']:
    #     print(k, float(lane_total_dict1[k] - lane_total_dict2[k]))
    return total_dict


@LOSSES.register_module()
class DRIVING_BEV_STALoss(BaseLoss):
    def __init__(self, global_config: GlobalConfig, task_config):

        pc_range = task_config.pc_range
        pc_range = [0, 0, 0, 32.0, 120.0, 0]
        super(DRIVING_BEV_STALoss, self).__init__(pc_range, task_config)
        self.polyline_loss = BaseMapLossCost(pc_range, cls_loss_weight=1.0, l1_loss_weight=4.0,
                                             giou_loss_weight=0.01, pts_l1_loss_weight=5.0, pts_dir_loss_weight=0.005)

    def forward(self, preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor) -> dict:
        """

        Args:
            preds: preds tensor from model
            trues: trues tensor from dataloader
            masks: masks tensor from preprocess modules, used for mask invalid labels

        Returns: dictionary contains the loss of current batch.

        """

        loss = loss_computation(preds[0], trues, self.polyline_loss)
        return loss
