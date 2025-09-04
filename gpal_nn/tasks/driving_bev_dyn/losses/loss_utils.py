import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


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
        pred = _transpose_and_gather_feat(
            output, ind)  # 默认为0也被索引，所以需要消除，下面的mask起作用

        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
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
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = self.balanced_l1_loss(pred * mask, target * mask)
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

        # print(ShowDataStruct("pred_cen", pred_cen))
        # print(ShowDataStruct("trues[mode+'obj_mask']", trues[mode+'obj_mask']),
        #       trues[mode+'obj_mask'].min(), trues[mode+'obj_mask'].max())
        # print(ShowDataStruct(
        #     "trues[mode+'indices_center']", trues[mode+'indices_center']))
        # print(ShowDataStruct(
        #     "trues[mode+'cen_offset']", trues[mode+'cen_offset']))

        # import pickle as pkl
        # pkl.dump((pred_cen, trues), open("new_repo.pkl", 'wb'))

        # breakpoint()
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

        total_loss = l_score + l_cen + l_z + l_dim + l_dir + l_vel
        # total_loss = l_score + l_cen  # + l_z + l_dim + l_dir + l_vel

        tb_dict = {}
        # # tb_dict['track_loss_score'] = l_score.item()
        # # tb_dict['track_loss_cen']   = l_cen.item()
        # # tb_dict['track_loss_z']     = l_z.item()
        # # tb_dict['track_loss_dim']   = l_dim.item()
        # # tb_dict['track_loss_dir']   = l_dir.item()
        # # tb_dict['track_loss_vel']   = l_vel.item()
        tb_dict['track_loss_score'] = l_score
        tb_dict['track_loss_cen'] = l_cen
        tb_dict['track_loss_z'] = l_z
        tb_dict['track_loss_dim'] = l_dim
        tb_dict['track_loss_dir'] = l_dir
        tb_dict['track_loss_vel'] = l_vel

        return total_loss, tb_dict
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

        # is_track_task = batch_dict["track"]
        is_track_task = True
        outputs = preds  # batch_dict['target']

        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])  # 固定的字段, 来自于yaml
        l_hm_cen = self.focal_loss(
            outputs['hm_cen'], trues['gt_curr_hm_cen'], False)
        total_loss = l_hm_cen * self.weight_hm_cen
        # tb_dict['track_loss_heatmap'] = l_hm_cen.item()
        tb_dict['track_loss_hm'] = l_hm_cen  # with gradient

        # det, not track, drop
        if not is_track_task:
            raise NotImplementedError
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            outputs['vel'] = _sigmoid(outputs['vel'])
            outputs['z_coor'] = outputs['z_coor']
            l_z_coor = self.l1_loss(
                outputs['z_coor'], batch_dict['obj_mask'], batch_dict['indices_center'], batch_dict['z_coor'])
            l_dim = self.l1_loss_balanced(
                outputs['dim'], batch_dict['obj_mask'], batch_dict['indices_center'], batch_dict['dim'])
            l_vel = self.l1_loss(
                outputs['vel'], batch_dict['obj_mask'], batch_dict['indices_center'], batch_dict['vel'])
            l_cen_offset = self.l1_loss(
                outputs['cen_offset'], batch_dict['obj_mask'], batch_dict['indices_center'], batch_dict['cen_offset'])
            l_direction = self.l1_loss(
                outputs['direction'], batch_dict['obj_mask'], batch_dict['indices_center'], batch_dict['direction'])
            box_loss = l_cen_offset * self.weight_cenoff + \
                l_dim * self.weight_dim + l_direction * self.weight_direction + \
                l_z_coor * self.weight_z_coor + l_vel
            total_loss += box_loss
            tb_dict['z_coor'] = l_z_coor.item()
            tb_dict['dim'] = l_dim.item()
            tb_dict['cen_offset'] = l_cen_offset.item()
            tb_dict['direction'] = l_direction.item()
            tb_dict['vel'] = l_vel.item()

        return total_loss, tb_dict
