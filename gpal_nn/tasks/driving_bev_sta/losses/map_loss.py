import torch
import torch.nn as nn

from gpal_nn.tasks.driving_bev_sta.losses.base_assigner import MapAssigner

from gpal_nn.tasks.driving_bev_sta.losses.ClsLabel import ClassLabelLossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.BBox import BBoxL1LossWithCost, GIoULossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.Points import PointsL1LossWithCost, PointsDirLoss

from gpal_nn.tasks.driving_bev_sta.losses.base_assigner import normalize_2d_bbox, denormalize_2d_bbox, normalize_2d_pts, denormalize_2d_pts
from gpal_nn.tasks.driving_bev_sta.losses.focal_loss import FocalLoss
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *


class BaseMapLossCost(nn.Module):
    def __init__(self, pc_range=(0, 0, 0, 100, 50, 0), cls_loss_weight=2.0, l1_loss_weight=0.0,
                 giou_loss_weight=0.0, pts_l1_loss_weight=5.0, pts_dir_loss_weight=0.005, output_group=None):
        super().__init__()

        self.bbox_loss = BBoxL1LossWithCost()
        self.iou_loss = GIoULossWithCost()
        self.pts_dir_loss = PointsDirLoss()
        self.pts_l1_loss = PointsL1LossWithCost()
        self.keypoint_reg_loss = nn.L1Loss(reduction='none')
        self.cls_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
        )
        self.shape_type_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
        )
        self.centerline_type_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
        )
        self.keypoint_cls_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
        )
        self.cls_loss2 = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
            reduction="none"
        )
        self.lane_marking_type_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight
        )
        self.lane_marking_color_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight
        )
        self.shape_type_loss2 = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
            reduction="none"
        )
        self.lane_marking_type_loss2 = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
            reduction="none"
        )
        self.lane_marking_color_loss2 = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
            reduction="none"
        )
        self.centerline_type_loss2 = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
            reduction="none"
        )
        self.keypoint_cls_loss2 = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_weight,
            reduction="none"
        )

        self.pc_range = pc_range

        self.assigner = MapAssigner(pc_range, self.cls_loss)

        self.l1_loss_weight = l1_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.pts_l1_loss_weight = pts_l1_loss_weight
        self.pts_dir_loss_weight = pts_dir_loss_weight
        self.output_group = []
        if output_group is not None:
            start_vec_idx = 0
            for group in output_group:
                end_vec_idx = start_vec_idx + group[1]
                self.output_group.append([group[0], (start_vec_idx, end_vec_idx)])
                start_vec_idx = end_vec_idx

    def loss_single_group(self, score_pred, bbox_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, \
                          keypoint_cls_pred, keypoint_reg_pred, \
                          cls_gt, bbox_gt, points_gt, lane_marking_types_gt, lane_marking_colors_gt, shape_types_gt, centerline_type_gt, \
                           keypoint_cls_gt, keypoint_reg_gt, valid_masks, valid_lens, center_line_flags, is_centerline):
        loss_list = []
        avg_factor = []
        pred_mask_all = []
        gt_index_all = []
        gt_order_idx_all = []
        pred_to_gt_label_all = []
        for b, l in enumerate(valid_lens):
            if l == 0:
                if (center_line_flags is None) or (center_line_flags[b]):
                    loss_list.append(self.no_gt_loss(score_pred[b], bbox_pred[b], points_pred[b], \
                                                     lane_marking_type_pred[b], lane_marking_color_pred[b], shape_type_pred[b], \
                                                     centerline_type_pred[b], keypoint_cls_pred[b], keypoint_reg_pred[b]))
                
                pred_mask_all.append(torch.zeros_like(score_pred[0,:,0]))
                avg_factor.append(torch.zeros_like(score_pred[0, :, 0]))
                pred_to_gt_label_all.append(
                    torch.zeros_like(score_pred[0, :, 0])+3)
                continue
                
            gt_width = cls_gt.shape[1]
            pred_to_gt_index, pred_to_gt_label, order_index = self.assigner.assign(bbox_pred[b], score_pred[b], points_pred[b],
                                                                                bbox_gt[b,:l], cls_gt[b,:l], points_gt[b, :l])

            pred_mask = pred_to_gt_index > 0
            gt_order = pred_to_gt_index[pred_mask]
            order_index = order_index[pred_mask]

            pred_mask_all.append(pred_mask)
            gt_index_all.append(gt_order + gt_width * b)
            gt_order_idx_all.append(order_index)
            pred_to_gt_label_all.append(pred_to_gt_label)
            avg_factor.append(torch.ones_like(
                pred_to_gt_index).float() / len(gt_order))

        pred_mask_all = torch.cat(pred_mask_all).bool()
        if len(gt_index_all) == 0:
            # 若为空，创建空张量（注意设备要与其他张量一致，这里用 bbox_pred 的设备）
            gt_index_all = torch.tensor([], dtype=torch.long, device=bbox_pred.device) - 1
        else:
            gt_index_all = torch.cat(gt_index_all).long() - 1
        if len(gt_order_idx_all) == 0:
            gt_order_idx_all = torch.tensor([], dtype=torch.long, device=bbox_pred.device)
        else:
            gt_order_idx_all = torch.cat(gt_order_idx_all).long()
        avg_factor = torch.cat(avg_factor).float()
        pred_to_gt_label_all = torch.cat(pred_to_gt_label_all).long()

        _bbox_pred = bbox_pred.flatten(0,1)[pred_mask_all]
        _points_pred = points_pred.flatten(0, 1)[pred_mask_all]
        _lane_marking_type_pred = lane_marking_type_pred.flatten(0,1)[pred_mask_all]
        _lane_marking_color_pred = lane_marking_color_pred.flatten(0,1)[pred_mask_all]
        _shape_type_pred = shape_type_pred.flatten(0,1)[pred_mask_all]
        _centerline_type_pred = centerline_type_pred.flatten(0,1)[pred_mask_all]
        _keypoint_cls_pred = keypoint_cls_pred.flatten(0,1)[pred_mask_all]
        _keypoint_reg_pred = keypoint_reg_pred.flatten(0,1)[pred_mask_all]
        _bbox_gt = bbox_gt.flatten(0,1)[gt_index_all]
        _points_gt = points_gt.flatten(0, 1)[gt_index_all, gt_order_idx_all]
        _lane_marking_type_gt = lane_marking_types_gt.flatten(0,1)[gt_index_all]
        _lane_marking_color_gt = lane_marking_colors_gt.flatten(0,1)[gt_index_all]
        _shape_type_gt = shape_types_gt.flatten(0, 1)[gt_index_all]
        _centerline_type_gt = centerline_type_gt.flatten(0, 1)[gt_index_all]
        _keypoint_cls_gt = keypoint_cls_gt.flatten(0, 1)[gt_index_all]
        _keypoint_reg_gt = keypoint_reg_gt.flatten(
            0, 1)[gt_index_all, gt_order_idx_all]

        loss = self.loss_single(score_pred.flatten(0, 1), _bbox_pred, _points_pred, _lane_marking_type_pred, _lane_marking_color_pred, \
                                _shape_type_pred, _centerline_type_pred, 
                                  _keypoint_cls_pred, _keypoint_reg_pred,
                                pred_to_gt_label_all, _bbox_gt, _points_gt, \
                                    _lane_marking_type_gt, _lane_marking_color_gt, _shape_type_gt, _centerline_type_gt, \
                                _keypoint_cls_gt, _keypoint_reg_gt, is_centerline)


        avg_factor = avg_factor.unsqueeze(-1)
        loss['loss_score'] = (loss['loss_score'] * avg_factor).sum()
        loss['loss_lane_marking_type'] = (
            loss['loss_lane_marking_type'] * avg_factor[pred_mask_all]).sum()
        loss['loss_lane_marking_color'] = (
            loss['loss_lane_marking_color'] * avg_factor[pred_mask_all]).sum()
        loss['loss_shape_type'] = (
            loss['loss_shape_type'] * avg_factor[pred_mask_all]).sum()
        loss['loss_centerline_type'] = (
            loss['loss_centerline_type'] * avg_factor[pred_mask_all]).sum()
        loss['loss_keypoint_cls'] = (
            loss['loss_keypoint_cls'] * avg_factor[pred_mask_all]).sum()
        loss_list.append(loss)
        return loss_list

    def no_gt_loss(self, score_pred, bbox_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, keypoint_cls_pred, keypoint_reg_pred):
        cls_gt = torch.zeros_like(score_pred)
        score_loss = self.cls_loss(score_pred, cls_gt, avg_factor=1)
        box_l1_loss = self.bbox_loss(bbox_pred, bbox_pred).sum() * self.l1_loss_weight
        denormalized_bbox_pred = denormalize_2d_bbox(bbox_pred, self.pc_range)
        box_iou_loss = self.iou_loss(denormalized_bbox_pred, denormalized_bbox_pred).sum() * self.giou_loss_weight
        denormalized_points_pred = denormalize_2d_pts(points_pred, self.pc_range)
        points_l1_loss = self.pts_l1_loss(points_pred, points_pred).sum() * self.pts_l1_loss_weight
        points_dir_loss = self.pts_dir_loss(denormalized_points_pred,
                                            denormalized_points_pred).sum() * self.pts_dir_loss_weight
        lane_marking_type_loss = self.lane_marking_type_loss(lane_marking_type_pred, lane_marking_type_pred,
                                                           weight=torch.zeros_like(lane_marking_type_pred))
        lane_marking_color_loss = self.lane_marking_color_loss(lane_marking_color_pred, lane_marking_color_pred,
                                                             weight=torch.zeros_like(lane_marking_color_pred))
        shape_type_loss = self.shape_type_loss(shape_type_pred, shape_type_pred, weight=torch.zeros_like(shape_type_pred))
        centerline_type_loss = self.centerline_type_loss(centerline_type_pred, centerline_type_pred, weight=torch.zeros_like(centerline_type_pred))
        keypoint_cls_loss = self.keypoint_cls_loss(keypoint_cls_pred, keypoint_cls_pred, weight=torch.zeros_like(keypoint_cls_pred))
        keypoint_reg_loss = self.keypoint_reg_loss(keypoint_reg_pred, keypoint_reg_pred).sum() * self.pts_l1_loss_weight

        return {
            "loss_score": score_loss,
            "loss_box_l1": box_l1_loss,
            "loss_box_iou": box_iou_loss,
            "loss_points_l1": points_l1_loss,
            "loss_points_dir": points_dir_loss,
            "loss_lane_marking_type": lane_marking_type_loss,
            "loss_lane_marking_color": lane_marking_color_loss,
            "loss_shape_type": shape_type_loss,
            "loss_centerline_type": centerline_type_loss,
            "loss_keypoint_cls": keypoint_cls_loss,
            "loss_keypoint_reg": keypoint_reg_loss,
        }
    def forward(self, pred_items, gt_items):
        score_pred, bbox_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, keypoint_cls_pred, keypoint_reg_pred = pred_items

        total_loss_dict = {
            "loss_score": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_box_l1": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_box_iou": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_points_l1": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_points_dir": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_lane_marking_type": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_lane_marking_color": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_shape_type": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_centerline_type": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_keypoint_cls": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
            "loss_keypoint_reg": torch.tensor(0, dtype=torch.float32, device=bbox_pred.device),
        }

        for group_idx, group in enumerate(self.output_group):
            group_flag = f"group_{group_idx}"
            cls_gt = gt_items[group_flag]["classes"]
            bbox_gt = gt_items[group_flag]["bboxes"]
            points_gt = gt_items[group_flag]["points"]
            lane_marking_types_gt = gt_items[group_flag]["lane_marking_types"]
            lane_marking_colors_gt = gt_items[group_flag]["lane_marking_colors"]
            shape_types_gt = gt_items[group_flag]["types"]
            centerline_types_gt = gt_items[group_flag]["centerline_types"]
            keypoint_cls_gt = gt_items[group_flag]["keyp_cls"]
            keypoint_reg_gt = gt_items[group_flag]["keyp_reg"]
            valid_mask = gt_items[group_flag]["valid_mask"]
            valid_len = gt_items[group_flag]["valid_len"]
            center_line_flags = gt_items[group_flag]["center_line_flag"]

            is_centerline = main_class_type_map["centerline"] in group[0]
            start_vec_idx = group[1][0]
            end_vec_idx = group[1][1]
            cur_score_pred = score_pred[:, start_vec_idx:end_vec_idx]
            cur_bbox_pred = bbox_pred[:, start_vec_idx:end_vec_idx]
            cur_points_pred = points_pred[:, start_vec_idx:end_vec_idx]
            cur_lane_marking_type_pred = lane_marking_type_pred[:, start_vec_idx:end_vec_idx]
            cur_lane_marking_color_pred = lane_marking_color_pred[:, start_vec_idx:end_vec_idx]
            cur_shape_type_pred = shape_type_pred[:, start_vec_idx:end_vec_idx]
            cur_centerline_type_pred = centerline_type_pred[:, start_vec_idx:end_vec_idx]
            cur_keypoint_cls_pred = keypoint_cls_pred[:, start_vec_idx:end_vec_idx]
            cur_keypoint_reg_pred = keypoint_reg_pred[:, start_vec_idx:end_vec_idx]

            center_line_flags = center_line_flags if main_class_type_map["centerline"] in group[0] else None
            loss_dict_list = self.loss_single_group(cur_score_pred, cur_bbox_pred, cur_points_pred, \
                                                    cur_lane_marking_type_pred, cur_lane_marking_color_pred, cur_shape_type_pred, cur_centerline_type_pred, \
                                               cur_keypoint_cls_pred, cur_keypoint_reg_pred, \
                                               cls_gt, bbox_gt, points_gt, \
                                                lane_marking_types_gt, lane_marking_colors_gt, shape_types_gt, centerline_types_gt, \
                                               keypoint_cls_gt, keypoint_reg_gt, valid_mask, valid_len, center_line_flags, is_centerline)
            for loss_dict in loss_dict_list:
                for key in loss_dict:
                    total_loss_dict[key] += loss_dict[key]
        return total_loss_dict

    def loss_single(self, score_pred, bbox_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, keypoint_cls_pred, keypoint_reg_pred,
                    cls_gt, bbox_gt, points_gt, lane_marking_type_gt, lane_marking_color_gt, shape_type_gt, centerline_type_gt, keypoint_cls_gt, keypoint_reg_gt, is_centerline):

        cls_weight = torch.ones_like(score_pred)
        cls_valid_mask = cls_gt >= 0
        cls_gt_valid = torch.where(cls_valid_mask, cls_gt, torch.zeros_like(cls_gt))
        cls_weight = cls_weight * cls_valid_mask[:, None]
        score_loss = self.cls_loss2(score_pred, cls_gt_valid, weight=cls_weight, avg_factor=1)

        lane_marking_type_weight = torch.ones_like(lane_marking_type_pred)
        lane_marking_type_valid_mask = lane_marking_type_gt >= 0
        lane_marking_type_gt_valid = torch.where(lane_marking_type_valid_mask, lane_marking_type_gt, torch.zeros_like(lane_marking_type_gt))
        lane_marking_type_weight = lane_marking_type_weight * lane_marking_type_valid_mask[:, None]
        lane_marking_type_loss = self.lane_marking_type_loss2(lane_marking_type_pred, lane_marking_type_gt_valid, weight=lane_marking_type_weight, avg_factor=1)

        lane_marking_color_weight = torch.ones_like(lane_marking_color_pred)
        lane_marking_color_valid_mask = lane_marking_color_gt >= 0
        lane_marking_color_gt_valid = torch.where(lane_marking_color_valid_mask, lane_marking_color_gt, torch.zeros_like(lane_marking_color_gt))
        lane_marking_color_weight = lane_marking_color_weight * lane_marking_color_valid_mask[:, None]
        lane_marking_color_loss = self.lane_marking_color_loss2(lane_marking_color_pred, lane_marking_color_gt_valid, weight=lane_marking_color_weight, avg_factor=1)

        shape_type_weight = torch.ones_like(shape_type_pred)
        shape_type_valid_mask = shape_type_gt >= 0
        shape_type_gt_valid = torch.where(shape_type_valid_mask, shape_type_gt, torch.zeros_like(shape_type_gt))
        shape_type_weight = shape_type_weight * shape_type_valid_mask[:, None]
        shape_type_loss = self.shape_type_loss2(shape_type_pred, shape_type_gt_valid, weight=shape_type_weight, avg_factor=1)

        normalized_bbox_gt = normalize_2d_bbox(bbox_gt, self.pc_range)
        denormalized_bbox_pred = denormalize_2d_bbox(bbox_pred, self.pc_range)
        box_l1_loss = self.bbox_loss(
            bbox_pred, normalized_bbox_gt).sum() * self.l1_loss_weight
        box_iou_loss = self.iou_loss(
            denormalized_bbox_pred, bbox_gt).sum() * self.giou_loss_weight

        normalized_points_gt = normalize_2d_pts(points_gt, self.pc_range)
        denormalized_points_pred = denormalize_2d_pts(
            points_pred, self.pc_range)
        points_l1_loss = self.pts_l1_loss(points_pred,
                                        normalized_points_gt).sum() * self.pts_l1_loss_weight
        points_dir_loss = self.pts_dir_loss(
            denormalized_points_pred, points_gt).sum() * self.pts_dir_loss_weight
        centerline_type_weight = torch.ones_like(centerline_type_pred)
        centerline_type_valid_mask = centerline_type_gt >= 0
        centerline_type_gt_valid = torch.where(centerline_type_valid_mask, centerline_type_gt, torch.zeros_like(centerline_type_gt))
        centerline_type_weight = centerline_type_weight * centerline_type_valid_mask[:, None]
        centerline_type_loss = self.centerline_type_loss2(centerline_type_pred, centerline_type_gt_valid, weight=centerline_type_weight, avg_factor=1)

        if is_centerline:
            # keypoint_cls_gt为0，表示不是关键点，要把值改为1才能适用focal_loss
            keypoint_cls_loss = self.keypoint_cls_loss2(
                keypoint_cls_pred, (1 - keypoint_cls_gt), avg_factor=1)
            keypoint_reg_loss = (self.keypoint_reg_loss(
                keypoint_reg_pred, keypoint_reg_gt[:, None]) * keypoint_cls_gt[:, None]).sum() * self.pts_l1_loss_weight
        else:
            keypoint_cls_loss = torch.zeros_like(keypoint_cls_pred)
            keypoint_reg_loss = torch.zeros_like(keypoint_reg_pred).sum()

        return {
            "loss_score": score_loss,
            "loss_box_l1": box_l1_loss,
            "loss_box_iou": box_iou_loss,
            "loss_points_l1": points_l1_loss,
            "loss_points_dir": points_dir_loss,
            "loss_lane_marking_type": lane_marking_type_loss,
            "loss_lane_marking_color": lane_marking_color_loss,
            "loss_shape_type": shape_type_loss,
            "loss_centerline_type": centerline_type_loss,
            "loss_keypoint_cls": keypoint_cls_loss,
            "loss_keypoint_reg": keypoint_reg_loss,
        }
