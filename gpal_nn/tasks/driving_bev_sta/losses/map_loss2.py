import torch
import torch.nn as nn


from gpal_nn.tasks.driving_bev_sta.losses.ClsLabel import ClassLabelLossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.BBox import BBoxL1LossWithCost, GIoULossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.Points import PointsL1LossWithCost, PointsDirLoss

from gpal_nn.tasks.driving_bev_sta.losses.base_assigner import normalize_2d_bbox, denormalize_2d_bbox, normalize_2d_pts, denormalize_2d_pts
from gpal_nn.tasks.driving_bev_sta.losses.focal_loss import FocalLoss
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *
from scipy.optimize import linear_sum_assignment


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
            reduction="none"
        )

        self.pc_range = pc_range

        self.l1_loss_weight = l1_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.pts_l1_loss_weight = pts_l1_loss_weight
        self.pts_dir_loss_weight = pts_dir_loss_weight

        # 匹配相关
        self.cls_cost = self.cls_loss
        self.pts_cost = PointsL1LossWithCost()
        self.pts_weight = 5.0

    def get_type_loss(self, pred, target, weight=None, avg_factor=1):
        valid_mask = target >= 0
        target_valid = torch.where(valid_mask, target, torch.zeros_like(target))
        weight_final = torch.ones_like(pred) * valid_mask[:, None]
        if weight is not None:
            weight_final = weight_final * weight[:, None]
        loss = self.cls_loss(pred, target_valid, weight=weight_final, avg_factor=1).sum() / avg_factor
        return loss
    
    def forward(self, pred_items, gt_items):
        score_pred, bbox_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, \
        centerline_direction_pred, keypoint_cls_pred, keypoint_reg_pred, polygon_class_pred, arrow_class_pred = pred_items
        total_loss_dict = {}
        bs = score_pred.shape[0]
        score_target = score_pred.new_ones(score_pred.shape[:2], dtype=torch.long) * score_pred.shape[-1]
        bbox_target = torch.zeros_like(bbox_pred)
        points_target = torch.zeros_like(points_pred)
        lane_marking_type_target = lane_marking_type_pred.new_ones(lane_marking_type_pred.shape[:2], dtype=torch.long) * lane_marking_type_pred.shape[-1]
        lane_marking_color_target = lane_marking_color_pred.new_ones(lane_marking_color_pred.shape[:2], dtype=torch.long) * lane_marking_color_pred.shape[-1]
        shape_type_target = shape_type_pred.new_ones(shape_type_pred.shape[:2], dtype=torch.long) * shape_type_pred.shape[-1]
        centerline_type_target = centerline_type_pred.new_ones(centerline_type_pred.shape[:2], dtype=torch.long) * centerline_type_pred.shape[-1]
        centerline_direction_target = centerline_direction_pred.new_ones(centerline_direction_pred.shape[:2], dtype=torch.long) * centerline_direction_pred.shape[-1]
        keypoint_cls_target = keypoint_cls_pred.new_ones(keypoint_cls_pred.shape[:2], dtype=torch.long) * keypoint_cls_pred.shape[-1]
        keypoint_reg_target = torch.zeros_like(keypoint_reg_pred)
        polygon_class_target = polygon_class_pred.new_ones(polygon_class_pred.shape[:2], dtype=torch.long) * polygon_class_pred.shape[-1]
        arrow_class_target = arrow_class_pred.new_ones(arrow_class_pred.shape[:2], dtype=torch.long) * arrow_class_pred.shape[-1]
        target_weight = points_pred.new_zeros(points_pred.shape[:2], dtype=torch.float32)  # 标记是否匹配了gt

        for bs_idx in range(bs):
            cur_score_pred = score_pred[bs_idx]
            cur_points_pred = points_pred[bs_idx]

            cur_valid_mask = gt_items["valid_mask"][bs_idx]
            cur_class_gt = gt_items["classes"][bs_idx]
            cur_bbox_gt = gt_items["bboxes"][bs_idx]
            cur_points_gt = gt_items["points"][bs_idx]
            cur_lane_marking_type_gt = gt_items["lane_marking_types"][bs_idx]
            cur_lane_marking_color_gt = gt_items["lane_marking_colors"][bs_idx]
            cur_shape_type_gt = gt_items["types"][bs_idx]
            cur_centerline_type_gt = gt_items["centerline_types"][bs_idx]
            cur_keypoint_cls_gt = gt_items["keyp_cls"][bs_idx]
            cur_keypoint_reg_gt = gt_items["keyp_reg"][bs_idx]
            cur_polygon_class_gt = gt_items["polygon_classes"][bs_idx]
            cur_arrow_class_gt = gt_items["arrow_classes"][bs_idx]

            is_polyline = (cur_points_gt[:,2:,:,:] - cur_points_gt[:,0:1,:,:]).flatten(1,3).sum(-1) < 1e-3
            num_gts, num_preds = cur_points_gt.shape[0], cur_points_pred.shape[0]

            cls_cost = self.cls_cost.cost(cur_score_pred, cur_class_gt.clone())

            _, num_orders, num_pts_per_gtline, num_coords = cur_points_gt.shape
            normalized_gt_pts = normalize_2d_pts(cur_points_gt, self.pc_range)
            normalized_gt_pts[is_polyline, 2:, :, :] = 1e9
            pts_cost_ordered = self.pts_cost.cost(
                cur_points_pred, normalized_gt_pts) * self.pts_weight

            pts_cost_ordered = pts_cost_ordered.view(
                num_preds, num_gts, num_orders)
            pts_cost, order_index = torch.min(pts_cost_ordered, 2)

            cost = cls_cost + pts_cost

            valid_query_mask = cur_valid_mask.any(-1)
            if not valid_query_mask.any():
                continue
            valid_query_indice = torch.argwhere(valid_query_mask).squeeze(-1)

            cost[~cur_valid_mask] = 1e9
            cost = cost[valid_query_indice].detach().cpu().numpy()

            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

            match_valid = cost[matched_row_inds, matched_col_inds] < 1e8  # 防止gt超过本组query，而匹配到其他query
            if not match_valid.all():
                print("match_valid not all True: ", cur_class_gt)
            matched_row_inds = torch.from_numpy(matched_row_inds[match_valid]).to(bbox_pred.device)
            matched_col_inds = torch.from_numpy(matched_col_inds[match_valid]).to(bbox_pred.device)
            matched_row_inds = valid_query_indice[matched_row_inds]

            score_target[bs_idx, matched_row_inds] = cur_class_gt[matched_col_inds]
            bbox_target[bs_idx, matched_row_inds] = cur_bbox_gt[matched_col_inds]
            points_target[bs_idx, matched_row_inds] = cur_points_gt[matched_col_inds, order_index[matched_row_inds, matched_col_inds]]
            lane_marking_type_target[bs_idx, matched_row_inds] = cur_lane_marking_type_gt[matched_col_inds]
            lane_marking_color_target[bs_idx, matched_row_inds] = cur_lane_marking_color_gt[matched_col_inds]
            shape_type_target[bs_idx, matched_row_inds] = cur_shape_type_gt[matched_col_inds]
            centerline_type_target[bs_idx, matched_row_inds] = cur_centerline_type_gt[matched_col_inds]
            centerline_direction_target[bs_idx, matched_row_inds] = torch.clamp(order_index[matched_row_inds, matched_col_inds], 0, 1)
            keypoint_cls_target[bs_idx, matched_row_inds] = 1 - cur_keypoint_cls_gt[matched_col_inds]
            keypoint_reg_target[bs_idx, matched_row_inds] = cur_keypoint_reg_gt[matched_col_inds, 
                    torch.clamp(order_index[matched_row_inds, matched_col_inds], 0, 1)][:,None]
            polygon_class_target[bs_idx, matched_row_inds] = cur_polygon_class_gt[matched_col_inds]
            arrow_class_target[bs_idx, matched_row_inds] = cur_arrow_class_gt[matched_col_inds]
            target_weight[bs_idx, matched_row_inds] = 1.0

        centerline_mask = (score_target == main_class_type_map["centerline"])
        pos_num = max(target_weight.sum(), 1)
        total_loss_dict["loss_score"] = self.get_type_loss(
            score_pred.flatten(0, 1), score_target.flatten(0, 1), avg_factor=pos_num)
        total_loss_dict["loss_lane_marking_type"] = self.get_type_loss(
            lane_marking_type_pred.flatten(0, 1), lane_marking_type_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)
        total_loss_dict["loss_lane_marking_color"] = self.get_type_loss(
            lane_marking_color_pred.flatten(0, 1), lane_marking_color_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)
        total_loss_dict["loss_shape_type"] = self.get_type_loss(
            shape_type_pred.flatten(0, 1), shape_type_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)
        total_loss_dict["loss_centerline_type"] = self.get_type_loss(
            centerline_type_pred.flatten(0, 1), centerline_type_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)
        total_loss_dict["loss_centerline_direction"] = self.get_type_loss(
            centerline_direction_pred.flatten(0, 1), centerline_direction_target.flatten(0, 1), (target_weight * centerline_mask).flatten(0, 1), pos_num)
        total_loss_dict["loss_keypoint_cls"] = self.get_type_loss(
            keypoint_cls_pred.flatten(0, 1), keypoint_cls_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)
        total_loss_dict["loss_polygon_class"] = self.get_type_loss(
            polygon_class_pred.flatten(0, 1), polygon_class_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)
        total_loss_dict["loss_arrow_class"] = self.get_type_loss(
            arrow_class_pred.flatten(0, 1), arrow_class_target.flatten(0, 1), target_weight.flatten(0, 1), pos_num)

        normalized_bbox_gt = normalize_2d_bbox(bbox_target, self.pc_range)
        denormalized_bbox_pred = denormalize_2d_bbox(bbox_pred, self.pc_range)
        normalized_points_gt = normalize_2d_pts(points_target, self.pc_range)
        denormalized_points_pred = denormalize_2d_pts(
            points_pred, self.pc_range)
        
        x_valid = (torch.abs(points_target[:, :, :, 0] - points_target[:, :, 0:1, 0]) > 1e-3).any(-1)
        y_valid = (torch.abs(points_target[:, :, :, 1] - points_target[:, :, 0:1, 1]) > 1e-3).any(-1)
        reg_weight = ((x_valid & y_valid).float() * target_weight).flatten(0, 1)

        total_loss_dict["loss_box_l1"] = self.bbox_loss(
            bbox_pred.flatten(0, 1), normalized_bbox_gt.flatten(0, 1), weight=reg_weight[:, None]).sum() * self.l1_loss_weight / bs
        total_loss_dict["loss_box_iou"] = (self.iou_loss(
            denormalized_bbox_pred.flatten(0, 1), bbox_target.flatten(0, 1)) * reg_weight).sum() * self.giou_loss_weight / bs
        total_loss_dict["loss_points_l1"] = (self.pts_l1_loss(
            points_pred.flatten(0, 1), normalized_points_gt.flatten(0, 1)) * reg_weight).sum() * self.pts_l1_loss_weight / bs
        total_loss_dict["loss_points_dir"] = (self.pts_dir_loss(
            denormalized_points_pred.flatten(0, 1), points_target.flatten(0, 1)) * reg_weight).sum() * self.pts_dir_loss_weight / bs
        total_loss_dict["loss_keypoint_reg"] = (self.keypoint_reg_loss(
            keypoint_reg_pred, keypoint_reg_target) * (1 - keypoint_cls_target)[:,:,None]).sum() * self.pts_l1_loss_weight / bs
        
        return total_loss_dict
