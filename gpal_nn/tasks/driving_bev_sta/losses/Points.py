import torch
import torch.nn as nn
import torch.nn.functional as F


class PointsL1LossWithCost(nn.Module):
    def __init__(self, x_weight=1.0, y_weight=1.0):
        super().__init__()
        self.x_weight = x_weight
        self.y_weight = y_weight

    def forward(self, pts_pred, pts_gt):
        num_pred = pts_pred.shape[0]
        num_gt = pts_gt.shape[0]
        assert num_pred == num_gt
        loss_matrix = F.mse_loss(pts_pred, pts_gt, reduction='none')
        loss_matrix[..., 0] *= self.x_weight
        loss_matrix[..., 1] *= self.y_weight
        loss_matrix = loss_matrix.sum(-1).sum(-1)

        return loss_matrix

    @torch.no_grad()
    def cost(self, pts_pred, pts_gt):
        num_preds = pts_pred.shape[0]

        num_gts, num_orders, num_pts, num_coords = pts_gt.shape

        pts_gt = pts_gt.contiguous().flatten(0, 1)

        pred = pts_pred[:, None, :, :].repeat(1, num_gts * num_orders, 1, 1)
        gt = pts_gt[None, :, :, :].repeat(num_preds, 1, 1, 1)
        cost_matrix = F.mse_loss(pred, gt, reduction='none')
        cost_matrix[..., 0] *= self.x_weight
        cost_matrix[..., 1] *= self.y_weight
        cost_matrix = torch.sqrt(cost_matrix.sum(-1).sum(-1))
        # pts_pred = pts_pred.view(num_preds, -1)
        # pts_gt = pts_gt.contiguous().flatten(2).view(num_gts * num_orders, -1)
        # cost_matrix = torch.(pts_pred, pts_gt)
        # cost_matrix[..., 0] *= self.x_weight
        # cost_matrix[..., 1] *= self.y_weight
        return cost_matrix


class PointsDirLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pts_pred, pts_gt):
        num_pred = pts_pred.shape[0]
        num_gt = pts_gt.shape[0]
        assert num_pred == num_gt

        pts_pred_dir = pts_pred[..., 1:, :] - pts_pred[..., :-1, :]
        pts_gt_dir = pts_gt[..., 1:, :] - pts_gt[..., :-1, :]

        tgt_param = pts_gt_dir.new_ones((pts_gt_dir.shape[:2]))
        loss_matrix = F.cosine_embedding_loss(pts_pred_dir.flatten(0, 1), pts_gt_dir.flatten(0, 1),
                                              tgt_param.flatten(0), reduction='none').view(pts_gt_dir.shape[:2]).mean(
            -1)
        # print(loss_matrix.shape)
        return loss_matrix
