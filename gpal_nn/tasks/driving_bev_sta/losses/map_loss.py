import torch
import torch.nn as nn

from gpal_nn.tasks.driving_bev_sta.losses.base_assigner import MapAssigner

from gpal_nn.tasks.driving_bev_sta.losses.ClsLabel import ClassLabelLossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.BBox import BBoxL1LossWithCost, GIoULossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.Points import PointsL1LossWithCost, PointsDirLoss

from gpal_nn.tasks.driving_bev_sta.losses.base_assigner import normalize_2d_bbox, denormalize_2d_bbox, normalize_2d_pts, denormalize_2d_pts


class BaseMapLossCost(nn.Module):
    def __init__(self, num_label=2, pc_range=(0, 0, 0, 100, 50, 0), cls_loss_weight=2.0, l1_loss_weight=0.0,
                 giou_loss_weight=0.0, pts_l1_loss_weight=5.0, pts_dir_loss_weight=0.005):
        super().__init__()

        self.bbox_loss = BBoxL1LossWithCost()
        self.iou_loss = GIoULossWithCost()
        self.pts_dir_loss = PointsDirLoss()
        self.pts_l1_loss = PointsL1LossWithCost()
        self.cls_loss = ClassLabelLossWithCost(num_label)
        self.num_label = num_label
        self.pc_range = pc_range

        self.assigner = MapAssigner(pc_range, num_label)

        self.cls_loss_weight = cls_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.pts_l1_loss_weight = pts_l1_loss_weight
        self.pts_dir_loss_weight = pts_dir_loss_weight

    def forward(self, pred_items, gt_items):
        score_pred, bbox_pred, points_pred = pred_items
        bbox_gt, points_gt = gt_items

        if bbox_gt.shape[0] == 0 or points_gt.shape[0] == 0:
            cls_gt = nn.functional.one_hot(
                torch.zeros((score_pred.shape[0]), dtype=torch.long, device=score_pred.device), self.num_label).float()
            score_loss = self.cls_loss(score_pred, cls_gt).sum() * self.cls_loss_weight
            box_l1_loss = self.bbox_loss(bbox_pred, bbox_pred).sum() * self.l1_loss_weight
            denormalized_bbox_pred = denormalize_2d_bbox(bbox_pred, self.pc_range)
            box_iou_loss = self.iou_loss(denormalized_bbox_pred, denormalized_bbox_pred).sum() * self.giou_loss_weight
            denormalized_points_pred = denormalize_2d_pts(points_pred, self.pc_range)
            points_l1_loss = self.pts_l1_loss(points_pred, points_pred).sum() * self.pts_l1_loss_weight
            points_dir_loss = self.pts_dir_loss(denormalized_points_pred,
                                                denormalized_points_pred).sum() * self.pts_dir_loss_weight

            # score_loss = torch.nan_to_num(score_loss)
            # box_l1_loss = torch.nan_to_num(box_l1_loss)
            # box_iou_loss = torch.nan_to_num(box_iou_loss)
            # points_l1_loss = torch.nan_to_num(points_l1_loss)
            # points_dir_loss = torch.nan_to_num(points_dir_loss)

            return {
                "loss_score": score_loss,
                "loss_box_l1": box_l1_loss,
                "loss_box_iou": box_iou_loss,
                "loss_points_l1": points_l1_loss,
                "loss_points_dir": points_dir_loss,
            }

        cls_gt = nn.functional.one_hot(torch.ones(bbox_gt.shape[0]).long(), self.num_label).float()
        bbox_gt, points_gt, cls_gt = \
            bbox_gt.to(bbox_pred.device), points_gt.to(points_pred.device), cls_gt.to(score_pred.device),

        pred_to_gt_index, pred_to_gt_label, order_index = self.assigner.assign(bbox_pred, score_pred, points_pred,
                                                                               bbox_gt, cls_gt, points_gt)

        pred_mask = pred_to_gt_index > 0

        gt_order = pred_to_gt_index[pred_mask]
        order_index = order_index[pred_mask]

        _bbox_pred = bbox_pred[pred_mask]
        _points_pred = points_pred[pred_mask]

        _bbox_gt = bbox_gt[gt_order - 1]
        # loguru.logger.info(f"{points_gt.shape}, {gt_order.shape}, {order_index.shape}")

        _points_gt = points_gt[gt_order - 1, order_index]
        # loguru.logger.info(f"{_points_gt.shape}, {gt_order.shape}, {order_index.shape}")

        # print(f"{_score_pred.shape}, {_bbox_pred.shape}, {_points_pred.shape}")
        # print(f"{_cls_gt.shape}, {_bbox_gt.shape}, {_points_gt.shape}")
        # print(score_pred, pred_to_gt_label)
        return self.loss_single(score_pred, _bbox_pred, _points_pred, pred_to_gt_label, _bbox_gt, _points_gt)

    def loss_single(self, score_pred, bbox_pred, points_pred, cls_gt, bbox_gt, points_gt):
        score_loss = self.cls_loss(score_pred, cls_gt).sum() * self.cls_loss_weight

        normalized_bbox_gt = normalize_2d_bbox(bbox_gt, self.pc_range)
        denormalized_bbox_pred = denormalize_2d_bbox(bbox_pred, self.pc_range)
        box_l1_loss = self.bbox_loss(bbox_pred, normalized_bbox_gt).sum() * self.l1_loss_weight
        box_iou_loss = self.iou_loss(denormalized_bbox_pred, bbox_gt).sum() * self.giou_loss_weight

        
        normalized_points_gt = normalize_2d_pts(points_gt, self.pc_range)
        denormalized_points_pred = denormalize_2d_pts(points_pred, self.pc_range)
        points_l1_loss = self.pts_l1_loss(points_pred,
                                          normalized_points_gt).sum() * self.pts_l1_loss_weight
        # import torch.nn.functional as F

        # loss_matrix = F.mse_loss(points_pred, normalized_points_gt, reduction='none')


        # print(f"points_pred = {points_pred[0]}")
        # print(f"normalized_points_gt = {normalized_points_gt[0]}")
        # print(f"points_l1_loss = {points_l1_loss}")
        # print(f"self.pts_l1_loss_weight = {self.pts_l1_loss_weight}")
        # print(f"points_l1_loss = {loss_matrix.abs().sum() * self.pts_l1_loss_weight}")
        # exit(1)

        # points_l1_loss = self.pts_l1_loss(
        #     denormalized_points_pred, points_gt).sum() * self.pts_l1_loss_weight
        points_dir_loss = self.pts_dir_loss(denormalized_points_pred, points_gt).sum() * self.pts_dir_loss_weight

        # score_loss = torch.nan_to_num(score_loss)
        # box_l1_loss = torch.nan_to_num(box_l1_loss)
        # box_iou_loss = torch.nan_to_num(box_iou_loss)
        # points_l1_loss = torch.nan_to_num(points_l1_loss)
        # points_dir_loss = torch.nan_to_num(points_dir_loss)

        return {
            "loss_score": score_loss,
            "loss_box_l1": box_l1_loss,
            "loss_box_iou": box_iou_loss,
            "loss_points_l1": points_l1_loss,
            "loss_points_dir": points_dir_loss,
        }
