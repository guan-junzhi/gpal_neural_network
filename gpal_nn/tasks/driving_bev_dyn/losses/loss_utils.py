import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from tools_scripts.data_format_cvt import ShowDataStruct


def _sigmoid(x):
    return torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)


def _neg_loss(pred, gt, track, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    # pos_inds = gt.eq(1)
    # neg_inds = gt.lt(1).float()

    pos_inds = gt.gt(0)  # greater than 0
    neg_inds = gt.eq(0).float()  # equal to 0
    
    # print(pos_inds.sum())
    # print(neg_inds.sum())
    # exit(1)
    # neg_weights = torch.pow(1 - gt, beta)
    if track:
        neg_weights = torch.pow(1 - gt, beta)
    else:
        neg_weights = torch.pow(1 - gt, beta)

    # neg_weights = torch.pow(1 - gt, beta)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds.float()
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               alpha) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = - neg_loss
    else:
        loss = - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, track):
        loss = self.neg_loss(out, target, track)
        return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind.long())

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # pred = _transpose_and_gather_feat(
        #     output, ind)  # 默认为0也被索引，所以需要消除，下面的mask起作用

        # mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss_Balanced(nn.Module):
    """Balanced L1 Loss
    paper: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    Code refer from: https://github.com/OceanPang/Libra_R-CNN
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0):
        super(L1Loss_Balanced, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        assert beta > 0
        self.beta = beta

    def forward(self, output, mask, ind, target):
        # pred = _transpose_and_gather_feat(output, ind)
        # mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = self.balanced_l1_loss(output * mask, target * mask)
        loss = loss.sum() / (mask.sum() + 1e-4)

        return loss

    def balanced_l1_loss(self, pred, target):
        assert pred.size() == target.size() and target.numel() > 0

        diff = torch.abs(pred - target)
        b = math.exp(self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta,
                           self.alpha / b *
                           (b * diff + 1) * torch.log(b * diff /
                                                      self.beta + 1) - self.alpha * diff,
                           self.gamma * diff**2 + self.gamma / b - self.alpha * self.beta)
        return loss


class L1Loss_Direction_Balanced(nn.Module):
    """
    优化版 Balanced L1 Loss for 方向预测
    改进特性：
    1. 动态参数调整（alpha/gamma自动缩放）
    2. 方向敏感加权（自动识别sin/cos形式）
    3. 多尺度特征聚集（增强小物体方向预测）
    4. 数值稳定性增强

    原始调用接口完全兼容：
    l_dir = L1Loss_Balanced()(pred, mask, indices, target)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, multiscale=True):
        super(L1Loss_Direction_Balanced, self).__init__()
        # 基础参数（保持原始配置）
        self.register_buffer('base_alpha', torch.tensor(alpha))
        self.register_buffer('base_gamma', torch.tensor(gamma))
        self.register_buffer('base_beta', torch.tensor(beta))

        # 动态调整参数（训练中自动优化）
        self.alpha_scale = nn.Parameter(torch.tensor(1.0))  # sigmoid缩放
        self.gamma_scale = nn.Parameter(torch.tensor(1.0))  # exp缩放

        # 方向感知权重（仅当预测为sin/cos时激活）
        self.dir_weight = nn.Parameter(torch.tensor(0.3))   # sigmoid限制到0~1

        # 多尺度特征开关
        self.multiscale = multiscale  # 可配置项

    def forward(self, output, mask, ind, target):
        # 多尺度特征聚集（自动选择）
        pred = self._gather_multiscale(
            output, ind) if self.multiscale else _transpose_and_gather_feat(output, ind)
        # target_feat = _transpose_and_gather_feat(target, ind) if target.dim() > 2 else target
        target_feat = target
        mask = mask.unsqueeze(2).expand_as(pred).float()

        # 动态参数计算（带约束）
        alpha = torch.sigmoid(self.alpha_scale) * self.base_alpha
        gamma = torch.exp(self.gamma_scale) * self.base_gamma
        beta = self.base_beta  # 保持固定

        # 核心损失计算（数值稳定版）
        diff = torch.abs(pred * mask - target_feat * mask)
        b = torch.exp(gamma / alpha) - 1

        loss = torch.where(
            diff < beta,
            (alpha / b) * (b * diff + 1) *
            torch.log(b * diff / beta + 1 + 1e-7) - alpha * diff,
            gamma * diff + (gamma / b - alpha * beta)
        )

        # 方向敏感加权（自动检测sin/cos形式）
        if pred.size(-1) == 2 and target_feat.size(-1) == 2:
            cos_sim = F.cosine_similarity(
                pred, target_feat, dim=-1).unsqueeze(-1)
            dir_weight = 1.0 + torch.sigmoid(self.dir_weight) * (1.0 - cos_sim)
            loss = loss * dir_weight

        return loss.sum() / (mask.sum() + 1e-7)

    def _gather_multiscale(self, feat, ind):
        """多尺度特征聚集（增强小物体方向预测）"""
        feat_list = []
        for stride in [1, 2, 4]:  # 多尺度采样
            scaled_ind = ind // stride
            f = _transpose_and_gather_feat(feat, scaled_ind.clamp(min=0))
            feat_list.append(f)
        return torch.stack(feat_list, dim=-1).mean(dim=-1)

    def extra_repr(self):
        return (f'base_alpha={self.base_alpha:.2f}, base_gamma={self.base_gamma:.2f}, '
                f'base_beta={self.base_beta:.2f}, multiscale={self.multiscale}')

# transform 检测头损失函数


class Points_Loss(nn.Module):
    def __init__(self):
        super(Points_Loss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.l1_loss_balanced = L1Loss_Balanced(alpha=0.5, gamma=2.0, beta=1.0)
        self.l1_loss_direction_balanced = L1Loss_Direction_Balanced(
            alpha=0.3, gamma=2.0, beta=0.8, multiscale=False)
        self.weight_hm_cen = 1.0
        self.weight_l_vel = 1.0
        self.weight_l_dir = 1.0
        self.weight_l_dim = 1.0
        self.weight_l_xy = 1.0

    # def forward(self,batch_dict,tb_dict):
    def forward(self, preds, trues):
        batch_size = trues["batchsize"]

        # 通过 forward_ret_dict 传入
        outputs = preds['Points_Loss']
        num_obj = outputs['estimation_dir'].size(-1)
        pred_dir = outputs['estimation_dir'].view(batch_size, -1, 1, num_obj)
        pred_dim = outputs['estimation_dim'].view(batch_size, -1, 1, num_obj)
        pred_cen = outputs['estimation_cen'].view(batch_size, -1, 1, num_obj)
        pred_z = outputs['estimation_z'].view(batch_size, -1, 1, num_obj)

        pred_vel = _sigmoid(outputs['estimation_vel'].view(
            batch_size, -1, 1, num_obj))
        pred_score = _sigmoid(
            outputs['estimation_score'].view(batch_size, -1, 1, num_obj))

        gt_score, _ = trues['score'].max(dim=1)
        gt_score = gt_score.view(batch_size, -1, 1, num_obj)

        mode = forward_gt = 'track_'

        l_score = self.focal_loss(pred_score, gt_score, True)

        l_cen = self.l1_loss(
            pred_cen, trues[mode+'obj_mask'], trues[mode+'indices_center'], trues[mode+'cen_offset'])
        l_z = self.l1_loss_balanced(
            pred_z, trues[mode+'obj_mask'], trues[mode+'indices_center'], trues[mode+'z_coor'])
        l_dim = self.l1_loss_balanced(
            pred_dim, trues[mode+'obj_mask'], trues[mode+'indices_center'], trues[mode+'dim'])
        l_dir = self.l1_loss_balanced(
            pred_dir, trues[mode+'obj_mask'], trues[mode+'indices_center'], trues[mode+'direction'])
        # l_dir = self.l1_loss_direction_balanced(pred_dir, trues[mode+'obj_mask'], trues[mode+'indices_center'], trues[mode+'direction'])
        l_vel = self.l1_loss(
            pred_vel, trues[mode+'obj_mask'], trues[mode+'indices_center'], trues[mode+'vel'])

        tb_dict = {}
        tb_dict['track_loss_score'] = l_score
        tb_dict['track_loss_cen'] = l_cen
        tb_dict['track_loss_z'] = l_z
        tb_dict['track_loss_dim'] = l_dim
        tb_dict['track_loss_dir'] = l_dir
        tb_dict['track_loss_vel'] = l_vel

        return tb_dict
        # return total_loss, 0.0


class Compute_Loss(nn.Module):
    def __init__(self):
        super(Compute_Loss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.l1_loss_balanced = L1Loss_Balanced(alpha=0.5, gamma=1.5, beta=1.0)
        self.weight_hm_cen = 1.
        self.weight_z_coor, self.weight_cenoff, self.weight_dim, self.weight_direction = 1, 1, 1, 1

    def forward(self, preds, trues):
        tb_dict = {}
        # import pickle as pkl
        # pkl.dump((preds, trues), open("Compute_Loss.pkl", 'wb'))
        # exit(1)

        # is_track_task = batch_dict["track"]
        is_track_task = True
        outputs = preds  # batch_dict['target']

        hm_cen_sigmoid = _sigmoid(outputs['hm_cen'])  # 固定的字段, 来自于yaml
        l_hm_cen = self.focal_loss(
            hm_cen_sigmoid, trues['gt_curr_hm_cen'], False)
        total_loss = l_hm_cen * self.weight_hm_cen
        # tb_dict['track_loss_heatmap'] = l_hm_cen.item()
        tb_dict['track_loss_hm'] = l_hm_cen  # with gradient

        # cen_offset_sigmoid = _sigmoid(outputs['cen_offset'])
        # vel_sigmoid = _sigmoid(outputs['vel'])
        # z_coor_sigmoid = outputs['z_coor']
        head_conv = preds["head_conv"].permute(0, 2, 3, 1).flatten(1, 2)
        # hm_cen_cls = preds["hm_cen"].permute(0, 2, 3, 1).flatten(1, 2)


        gt_mask = trues['track_obj_mask'].unsqueeze(-1)
        gt_idx = trues['track_indices_center'].long()

        B, _, D = head_conv.shape

        head_conv_sel = head_conv[torch.arange(B)[:, None, None], gt_idx.reshape(B, -1, 1).repeat(1, 1, D), torch.arange(D)[None, None, :]]
        # hm_cen_cls_sel = hm_cen_cls[torch.arange(B)[:, None, None], gt_idx.reshape(B, -1, 1).repeat(1, 1, 6), torch.arange(6)[None, None, :]]
        # hm_cen_cls_sel = hm_cen_cls_sel.max(dim = -1)[0]
        # print("hm_cen_cls_sel ", hm_cen_cls_sel.shape)

        # gt_curr_hm_cen = trues['gt_curr_hm_cen'].permute(0, 2, 3, 1).flatten(1, 2)


        # for i in range(-1,2):
        #     for j in range(-1,2):
        #         idx = 7759 +240 * i + j
        #         print(idx, gt_curr_hm_cen[0, idx, 1], hm_cen_cls[0, idx, 1].sigmoid())

        # print(gt_curr_hm_cen.min(), gt_curr_hm_cen.max())
        # gt_curr_hm_cen_sel = gt_curr_hm_cen[torch.arange(B)[:, None, None], gt_idx.reshape(B, -1, 1).repeat(1, 1, 6), torch.arange(6)[None, None, :]]
        # gt_curr_hm_cen_sel = gt_curr_hm_cen_sel.max(dim = -1)[0]
        # print("gt_curr_hm_cen ", gt_curr_hm_cen_sel.shape)


        head_conv_sel = torch.stack([map[idx] for idx, map in zip(gt_idx, head_conv)], dim = 0)
        estimation_cen = head_conv_sel[..., :2]
        estimation_z = head_conv_sel[..., 2:3]
        estimation_dim = head_conv_sel[..., 3:6]
        estimation_dir = head_conv_sel[..., 6:8]
        estimation_vel = head_conv_sel[..., 8:10]
        estimation_score = head_conv_sel[..., 10:11]


        point_cloud_range = [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0]
        voxel_size = [0.64, 0.64, 0.5]
        H = 96
        W = 240

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W) * voxel_size[1] + point_cloud_range[1]
        xs = xs.view(1, H, W) * voxel_size[0] + point_cloud_range[0]
        xys = torch.cat([ys, xs], dim=0).view(
            1, 2, -1).to(head_conv_sel).repeat(B, 1, 1).flip(1).permute(0,2,1)
        xys_sel = xys[torch.arange(B)[:, None, None], gt_idx.reshape(B, -1, 1).repeat(1, 1, 2), torch.arange(2)[None, None, :]]

        # print(xys.shape)

        # exit(1)
        # print(trues['track_cen_offset'][0][:10])
        # print(xys_sel[0][:10])
        l_z_coor = self.l1_loss(estimation_z, gt_mask, None, trues['track_z_coor'])
        l_dim = self.l1_loss_balanced(estimation_dim, gt_mask, None, trues['track_dim'])
        # l_vel = self.l1_loss(
        #     vel_sigmoid, batch_dict['obj_mask'], batch_dict['indices_center'], batch_dict['vel'])
        l_cen_offset = self.l1_loss(estimation_cen + xys_sel, gt_mask, None, trues['track_cen_offset'])
        l_direction = self.l1_loss(estimation_dir, gt_mask, None, trues['track_direction'])
        # box_loss = l_cen_offset * self.weight_cenoff + \
        #     l_dim * self.weight_dim + l_direction * self.weight_direction + \
        #     l_z_coor * self.weight_z_coor + l_vel
        # total_loss += box_loss
      


        tb_dict['track_loss_score'] = torch.tensor(0.0).to(head_conv_sel.device)
        tb_dict['track_loss_cen'] = l_cen_offset
        tb_dict['track_loss_z'] = l_z_coor
        tb_dict['track_loss_dim'] = l_dim
        tb_dict['track_loss_dir'] = l_direction
        tb_dict['track_loss_vel'] = torch.tensor(0.0).to(head_conv_sel.device)

        # print(gt_mask.shape, trues['track_z_coor'].shape, trues['track_cen_offset'].shape, trues['track_direction'].shape)

        # preds["Points_Loss"]["estimation_score"] = gt_mask.permute(0,2,1).float()
        # preds["Points_Loss"]["estimation_cen"] = trues['track_cen_offset'].permute(
        #     0, 2, 1).float()
        # preds["Points_Loss"]["estimation_z"] = trues['track_z_coor'].permute(
        #     0, 2, 1).float()

        # preds["Points_Loss"]["estimation_dim"] = trues['track_dim'].permute(
        #     0, 2, 1).float() + 0.2
        # preds["Points_Loss"]["estimation_dir"] = trues['track_direction'].permute(
        #     0, 2, 1).float()


        # print([trues['track_cen_offset'].float().shape, trues['track_dim'].float().shape])
        # print([(estimation_cen + xys_sel).float().shape, estimation_dim.permute(0, 2, 1).float().shape])

        # print(preds["head_conv"].shape)

        # dim_l = (preds["head_conv"][0].permute(1,2,0)[...,3]).clip(0,10.0).detach().cpu().numpy()
        # print(dim_l.shape, dim_l.min(), dim_l.max())
        # import cv2
        # import numpy as np
        # cv2.imwrite("bev_dim.jpg", (dim_l * 10).astype(np.uint8))

        # box_gt = torch.cat([gt_idx.unsqueeze(-1), gt_curr_hm_cen_sel.unsqueeze(-1),
        #                    trues['track_cen_offset'].float(), trues['track_dim'].float(), trues['track_direction'].float()], dim=-1)
        # box_sel = torch.cat([hm_cen_cls_sel.unsqueeze(-1).sigmoid(), (estimation_cen + xys_sel).float(),
        #                     estimation_dim.float(), estimation_dir.float()], dim=-1)
        # box_pred = torch.cat([preds["pred_curr_track_point_idx"].unsqueeze(-1), preds["Points_Loss"]["estimation_score"].permute(0, 2, 1).float().sigmoid(), preds["Points_Loss"]
        #                      ["estimation_cen"].permute(0, 2, 1).float(), preds["Points_Loss"]["estimation_dim"].permute(0, 2, 1).float(), preds["Points_Loss"]["estimation_dir"].permute(0, 2, 1).float()], dim=-1)
        # # print(box_gt.shape, box_sel.shape)

        # torch.set_printoptions(precision = 2, sci_mode = False, linewidth =120)
        # # print(gt_mask[0])
        # print(box_gt[0,gt_mask[0,:,0].bool()])
        # print(box_sel[0,gt_mask[0,:,0].bool()])
        # print(box_pred[0,:50])

        return total_loss, tb_dict


if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("Compute_Loss.pkl", 'rb'))
    print("hello")
    # r50 = EncoderRes50(*inputs)
    loss = Compute_Loss()


    print(ShowDataStruct("inputs", inputs))
    y = loss(*inputs)

    print(y)
    

    #     0<class 'torch.Tensor'> : torch.Size([4, 64, 40, 96]) torch.float32
