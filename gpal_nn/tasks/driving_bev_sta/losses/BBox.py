import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.giou_loss import generalized_box_iou_loss


class BBoxL1LossWithCost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bbox_pred, bbox_gt, weight=None):
        num_pred = bbox_pred.shape[0]
        num_gt = bbox_gt.shape[0]

        assert num_pred == num_gt
        loss_matrix = F.smooth_l1_loss(bbox_pred, bbox_gt, reduction='none')
        if weight is not None:
            loss_matrix = loss_matrix * weight
        loss_matrix = loss_matrix.mean(-1)
        return loss_matrix

    @torch.no_grad()
    def cost(self, bbox_pred, bbox_gt):
        # num_pred = bbox_pred.shape[0]
        # num_gt = bbox_gt.shape[0]

        box_l1_cost = torch.cdist(bbox_pred, bbox_gt, p=1)
        return box_l1_cost


class GIoULossWithCost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bbox_pred, bbox_gt):
        num_pred = bbox_pred.shape[0]
        num_gt = bbox_gt.shape[0]
        assert num_pred == num_gt
        loss_matrix = generalized_box_iou_loss(bbox_pred, bbox_gt)
        return loss_matrix

    @torch.no_grad()
    def cost(self, bbox_pred, bbox_gt):
        num_pred = bbox_pred.shape[0]
        num_gt = bbox_gt.shape[0]

        pred = bbox_pred[:, None, :].repeat(1, num_gt, 1).view(-1, 4)
        target = bbox_gt[None, :, :].repeat(num_pred, 1, 1).view(-1, 4)

        cost_matrix = generalized_box_iou_loss(pred, target).view(num_pred, num_gt)
        return cost_matrix


class GIoUCost(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred, bbox):
        ix1 = torch.max(pred[..., 0], bbox[..., 0])
        iy1 = torch.max(pred[..., 1], bbox[..., 1])
        ix2 = torch.min(pred[..., 2], bbox[..., 2])
        iy2 = torch.min(pred[..., 3], bbox[..., 3])

        iw = (ix2 - ix1 + 1.0).clamp(0.)
        ih = (iy2 - iy1 + 1.0).clamp(0.)

        inters = iw * ih

        uni = (pred[..., 2] - pred[..., 0] + 1.0) * (pred[..., 3] - pred[..., 1] + 1.0) + (
                bbox[..., 2] - bbox[..., 0] + 1.0) * (
                      bbox[..., 3] - bbox[..., 1] + 1.0) - inters + self.eps

        ious = inters / uni

        ex1 = torch.min(pred[..., 0], bbox[..., 0])
        ey1 = torch.min(pred[..., 1], bbox[..., 1])
        ex2 = torch.max(pred[..., 2], bbox[..., 2])
        ey2 = torch.max(pred[..., 3], bbox[..., 3])
        ew = (ex2 - ex1 + 1.0).clamp(min=0.)
        eh = (ey2 - ey1 + 1.0).clamp(min=0.)

        enclose = ew * eh + self.eps

        giou = ious - (enclose - uni) / enclose

        cost = 1 - giou

        return cost


if __name__ == "__main__":
    x = torch.rand((3, 4)) * 640
    y = torch.rand((3, 4)) * 640
    w = torch.rand((3, 4)) * 20
    h = torch.rand((3, 4)) * 20

    x1 = x - w / 2.
    y1 = y - h / 2.
    x2 = x + w / 2.
    y2 = y + h / 2.

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    pred = bboxes

    giou_cost = GIoUCost()
    cost_matrix = giou_cost(pred, bboxes)
    print(cost_matrix.shape)
    print(cost_matrix)
