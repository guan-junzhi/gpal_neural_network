import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.focal_loss import sigmoid_focal_loss


class ClassLabelLossWithCost(nn.Module):
    def __init__(self, num_label):
        super().__init__()
        self.num_label = num_label

    def forward(self, pred, gt):
        if len(pred.shape) != len(gt.shape):
            # print(pred.shape, gt.shape)
            gt = F.one_hot(gt.long(), self.num_label).float()
        # print(pred.shape, gt.shape)

        num_pred = pred.shape[0]
        num_gt = gt.shape[0]
        assert num_gt == num_pred, f"num gt {num_gt}, num pred {num_pred}"
        # print(pred)
        # print(gt)
        # loss_matrix = sigmoid_focal_loss(pred, gt).mean(-1)
        loss_matrix = F.cross_entropy(pred, gt, reduction='none').mean(-1)
        # loss_matrix = F.binary_cross_entropy_with_logits(pred, gt, reduction='none').mean(-1)
        # print(loss_matrix.shape)
        # print(loss_matrix)

        return loss_matrix

    @torch.no_grad()
    def cost(self, pred, gt):
        if len(pred.shape) != len(gt.shape):
            gt = F.one_hot(gt.long(), self.num_label).float()

        num_pred = pred.shape[0]
        num_gt = gt.shape[0]

        pred = pred[:, None, :].repeat(1, num_gt, 1)
        gt = gt[None, :, :].repeat(num_pred, 1, 1)
        # cost_matrix = sigmoid_focal_loss(pred, gt).mean(-1)
        # cost_matrix = F.cross_entropy(pred, gt, reduction='none').mean(-1)
        cost_matrix = F.binary_cross_entropy_with_logits(pred, gt, reduction='none').mean(-1)

        return cost_matrix


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_label, weight=1.0):
        super().__init__()
        self.num_label = num_label
        #self.loss_fc = nn.NLLLoss(ignore_index=255)
        self.loss_fc = nn.CrossEntropyLoss(ignore_index=255)
        self.weight = weight

    def forward(self, pred, gt):
        # import pdb;pdb.set_trace()
        b, c, h, w = pred.shape
        bt, _, ht, wt = gt.shape

        assert h == ht
        assert w == wt

        tmp_pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        tmp_gt = gt.view(-1)

        #loss = self.loss_fc(F.log_softmax(tmp_pred, dim=-1), tmp_gt)
        loss = self.weight * self.loss_fc(tmp_pred, tmp_gt)
        return loss
