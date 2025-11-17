import torch
import torch.nn as nn

from scipy.optimize import linear_sum_assignment

from gpal_nn.tasks.driving_bev_sta.losses.lanes_to_box import xyxy_to_cxcywh, cxcywh_to_xyxy
from gpal_nn.tasks.driving_bev_sta.losses.ClsLabel import ClassLabelLossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.BBox import BBoxL1LossWithCost, GIoULossWithCost
from gpal_nn.tasks.driving_bev_sta.losses.Points import PointsL1LossWithCost


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] *
                         (pc_range[3] - pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] *
                         (pc_range[4] - pc_range[1]) + pc_range[1])

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]

    new_pts[..., 0:1] = pts[..., 0:1] * patch_w + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * patch_h + pc_range[1]

    return new_pts


class MapAssigner(nn.Module):
    def __init__(self, pc_range, cls_cost, reg_weight=0.0, iou_weight=0.0, pts_weight=5.0):
        super().__init__()
        self.cls_cost = cls_cost

        self.reg_cost = BBoxL1LossWithCost()
        self.reg_weight = reg_weight

        self.iou_cost = GIoULossWithCost()
        self.iou_weight = iou_weight

        self.pts_cost = PointsL1LossWithCost()
        self.pts_weight = pts_weight

        self.pc_range = pc_range  # [x0, y0, z0, x1, y1, z1]

    @torch.no_grad()
    def assign(self, bbox_pred, cls_pred, pts_pred, gt_bboxes, gt_label, gt_pts_in):
        assert bbox_pred.shape[-1] == 4, f"bbox_pred shape is wrong {bbox_pred.shape}"
        assert pts_pred.shape[-1] == 2, f"pts_pred shape is wrong {pts_pred.shape}"
        assert bbox_pred.shape[0] == pts_pred.shape[0], \
            f"bbox query {bbox_pred.shape} is not equal pts query {pts_pred.shape}"
        is_polyline = gt_pts_in[:,2:,:,:].sum() < 1e-3
        if is_polyline:
            gt_pts = gt_pts_in[:,:2,:,:]
        else:
            gt_pts = gt_pts_in
        num_gts, num_preds = gt_bboxes.shape[0], bbox_pred.shape[0]
        num_cls = cls_pred.shape[-1]

        assigned_gt_inds = bbox_pred.new_full(
            (num_preds,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_preds,), num_cls, dtype=torch.long)
        assigned_index = bbox_pred.new_full((num_preds,), -1, dtype=torch.long)

        # if num_gts == 0 or num_preds == 0:
        #     # No ground truth or boxes, return empty assignment
        #     if num_gts == 0:
        #         # No ground truth, assign all to background
        #         assigned_gt_inds[:] = 0
        #     return None

        cls_cost = self.cls_cost.cost(cls_pred, gt_label.clone())

        normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost.cost(
            bbox_pred[:, :4], normalized_gt_bboxes[:, :4]) * self.reg_weight

        # TODO: add points order next version
        # [num_query, num_order, num_pts_per_vec, 2] the order num default is 1
        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        denormalize_pts = denormalize_2d_pts(pts_pred, self.pc_range)
        pts_cost_ordered = self.pts_cost.cost(
            pts_pred, normalized_gt_pts) * self.pts_weight
        # pts_cost_ordered = self.pts_cost.cost(denormalize_pts, gt_pts) * self.pts_weight
        pts_cost_ordered = pts_cost_ordered.view(
            num_preds, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)
        bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)

        iou_cost = self.iou_cost.cost(bboxes, gt_bboxes) * self.iou_weight
        # print(f"cost matrix {cls_cost}, {reg_cost}, {iou_cost}, {pts_cost}")
        # print(iou_cost.shape)
        # invalid_iou = ((iou_cost > 2).float() + (iou_cost < 0).float()).sum()
        # print(invalid_iou)
        # print(cls_cost.shape, reg_cost.shape, iou_cost.shape, pts_cost.shape)
        # cls_cost = np.nan_to_num(cls_cost.detach().cpu().numpy())
        # reg_cost = np.nan_to_num(reg_cost.detach().cpu().numpy())
        # iou_cost = np.nan_to_num(iou_cost.detach().cpu().numpy())
        # pts_cost = np.nan_to_num(pts_cost.detach().cpu().numpy())
        cost = cls_cost + reg_cost + iou_cost + pts_cost

        cost = cost.detach().cpu().numpy()
        # cost = np.nan_to_num(cost)
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(
            matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(
            matched_col_inds).to(bbox_pred.device)
        order_index = order_index.to(bbox_pred.device)
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_label[matched_col_inds]
        assigned_index[matched_row_inds] = order_index[matched_row_inds,
                                                       matched_col_inds]
        return assigned_gt_inds, assigned_labels, assigned_index


if __name__ == "__main__":
    num_proposal = 10
    num_target = 5
    boxes_loss = torch.randn((num_proposal, num_target)).cpu()

    matched_row_inds, matched_col_inds = linear_sum_assignment(boxes_loss)
    matched_row_inds = torch.from_numpy(matched_row_inds)
    matched_col_inds = torch.from_numpy(matched_col_inds)
    print(matched_row_inds, matched_col_inds, boxes_loss)
