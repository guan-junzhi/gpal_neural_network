from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
# import etw_pytorch_utils as pt_utils
from gpal_nn.tasks.driving_bev_dyn.heads.transformer import TransformerDecoder, TransformerEncoder
from gpal_nn.tasks.driving_bev_dyn.heads.multihead_attention import MultiheadAttention
# from ..model_utils.losses import Points_Loss
import time


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


class PositionEmbeddingLearned(nn.Module):
    """ 
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=2, num_pos_feats=64):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class PointnetTransformerSiamese(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training, voxel_size):
        super().__init__()
        d_model = 64
        num_layers = 1
        self.with_pos_embed = True
        self.predict_boxes_when_training = predict_boxes_when_training

        self.fea_layer = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(
                                           64, 64, kernel_size=1, bias=False),
                                       )
        self.vote_layer = nn.Sequential(nn.Conv2d(67, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(
                                            64, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(
                                            64, 66, kernel_size=1, bias=False),
                                        )
        self.FC_proposal_score = nn.Sequential(nn.Conv2d(67, 67, kernel_size=1, bias=False),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(
                                                   67, 1, kernel_size=1, bias=False),
                                               )
        self.FC_proposal_dir = nn.Sequential(nn.Conv2d(67, 67, kernel_size=1, bias=False),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(
                                                 67, 2, kernel_size=1, bias=False),
                                             )
        self.FC_proposal_dim = nn.Sequential(nn.Conv2d(67, 67, kernel_size=1, bias=False),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(
                                                 67, 3, kernel_size=1, bias=False),
                                             )
        self.FC_proposal_vel = nn.Sequential(nn.Conv2d(67, 67, kernel_size=1, bias=False),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(
                                                 67, 2, kernel_size=1, bias=False),
                                             )
        self.FC_proposal_xy = nn.Sequential(nn.Conv2d(67, 67, kernel_size=1, bias=False),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(
                                                67, 2, kernel_size=1, bias=False),
                                            )
        self.FC_proposal_z = nn.Sequential(nn.Conv2d(67, 67, kernel_size=1, bias=False),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(
                                               67, 1, kernel_size=1, bias=False),
                                           )
        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=1, key_feature_dim=64)

        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
            decoder_pos_embed = PositionEmbeddingLearned(3, d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None

        self.encoder = TransformerEncoder(multihead_attn=multihead_attn,
                                          FFN=None,
                                          d_model=d_model,
                                          num_encoder_layers=num_layers,
                                          self_posembed=encoder_pos_embed
                                          )
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn,
                                          FFN=None,
                                          d_model=d_model,
                                          num_decoder_layers=num_layers,
                                          self_posembed=decoder_pos_embed
                                          )
        self.forward_ret_dict = {}

    def apply_kfpn(self, outs):
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        return ret_outs

    def transform_fuse(self, template_feature, template_xyz, search_feature, search_xyz):
        """Use transformer to fuse feature.
        template_feature : BxCxN
        template_xyz : BxNx3
        """
        # BxCxN -> NxBxC
        search_feature = search_feature.permute(2, 0, 1)
        template_feature = template_feature.permute(2, 0, 1)

        num_img_train = search_feature.shape[0]
        num_img_template = template_feature.shape[0]
        # encoder
        encoded_memory = self.encoder(
            template_feature, query_pos=template_xyz if self.with_pos_embed else None)

        encoded_feat, track_mask, output1 = self.decoder(
            search_feature, memory=encoded_memory, query_pos=search_xyz)

        # NxBxC -> BxCxN
        encoded_feat = encoded_feat.permute(1, 2, 0).unsqueeze(-1)
        encoded_feat = self.fea_layer(encoded_feat).squeeze(-1)
        output1 = output1.permute(1, 2, 0).unsqueeze(-1)
        output1 = self.fea_layer(output1).squeeze(-1)

        return encoded_feat, track_mask, output1

    def forward(self, batch_dict):

        # former_point_coords中第一个维度是bs维度 2+1=>3, former_score是从centerhead出来的得分
        template_xyz = torch.cat([batch_dict['pred_prev_point_coords'][:, :, :2],
                                 batch_dict['pred_prev_score']], dim=-1)  # -> [1, 256, 3]
        template_feature = batch_dict['pred_prev_point_features'].permute(
            0, 2, 1)  # -> [1, 64, 256]

        search_xyz = torch.cat([batch_dict['pred_curr_track_point_coords'][:, :, :2],
                               batch_dict['pred_curr_track_score']], dim=-1)  # -> [1, 256, 3]
        search_feature = batch_dict['pred_curr_track_point_features'].permute(
            0, 2, 1)

        fusion_feature, track_mask, fusion_featuret = self.transform_fuse(
            template_feature, template_xyz, search_feature, search_xyz)

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(
        ), fusion_feature), dim=1).unsqueeze(-1)  # 当前帧(位置,得分)和对应帧融合特征

        offset_feature = self.vote_layer(
            fusion_xyz_feature).squeeze(-1)  # -> [1, 66, 256]

        fusion_offset_feature = self.apply_kfpn(
            [search_feature, offset_feature[:, 2:, :]])  # [64, 64] -> [1, 64, 256]
        fusion_xyz_offset_featured = torch.cat((search_xyz.transpose(
            1, 2).contiguous(), fusion_offset_feature), dim=1).unsqueeze(-1)

        estimation_score = self.FC_proposal_score(
            fusion_xyz_offset_featured).squeeze(-1)
        estimation_dim = self.FC_proposal_dim(
            fusion_xyz_offset_featured).squeeze(-1)
        estimation_dir = self.FC_proposal_dir(
            fusion_xyz_offset_featured).squeeze(-1)
        estimation_vel = self.FC_proposal_vel(
            fusion_xyz_offset_featured).squeeze(-1)
        estimation_cen = self.FC_proposal_xy(
            fusion_xyz_offset_featured).squeeze(-1)
        estimation_z = self.FC_proposal_z(
            fusion_xyz_offset_featured).squeeze(-1)

        if self.training:
            batch_dict['Points_Loss'] = {
                'estimation_cen': estimation_cen,
                'estimation_z': estimation_z,
                'estimation_dim': estimation_dim,
                'estimation_dir': estimation_dir,
                'estimation_vel': estimation_vel,
                'estimation_score': estimation_score,
            }

            # 热力图原始得分(含通道) 训练和测试会切换
            # self.forward_ret_dict['score'] = batch_dict['score']
            # self.forward_ret_dict['track_cen_offset'] = batch_dict['gt_curr_cen_offset']
            # self.forward_ret_dict['track_dim'] = batch_dict['gt_curr_dim']
            # self.forward_ret_dict['track_direction'] = batch_dict['gt_curr_direction']
            # self.forward_ret_dict['track_z_coor'] = batch_dict['gt_curr_z_coor']
            # self.forward_ret_dict['track_vel'] = batch_dict['gt_curr_vel']
            # self.forward_ret_dict['track_obj_mask'] = batch_dict['gt_curr_obj_mask']
            # self.forward_ret_dict['track_indices_center'] = batch_dict['gt_curr_indices_center']

            # self.forward_ret_dict['batch_size'] = batch_dict['batch_size']

        else:
            _, label = batch_dict['score'].max(dim=1)  # 原始的score含通道
            batch_dict['batch_pred_labels'] = label.view(
                batch_dict['batch_size'], -1) + 1

            self.forward_ret_dict['forward'] = {
                'estimation_cen': estimation_cen,
                'estimation_z': estimation_z,
                'estimation_dim': estimation_dim,
                'estimation_dir': estimation_dir,
                'estimation_vel': estimation_vel,
                'score': estimation_score,
                'template_xyz': template_xyz,  # xy实际位置和上一帧得分
            }

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'])
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = True
            batch_dict['has_class_labels'] = True

        return batch_dict

    def get_loss(self, tb_dict):
        rpn_loss, tb_dict = Points_Loss()(self.forward_ret_dict, tb_dict)
        tb_dict['track_loss_rpn(tot)'] = rpn_loss
        return rpn_loss, tb_dict

    @torch.no_grad()
    def generate_predicted_boxes(self, batch_size, **kwargs):

        outputs = self.forward_ret_dict['forward']
        template_xyz = outputs['template_xyz'][:, :,
                                               :2].view(-1, 2)  # 前一帧的xy实际位置(从keypts获得)和得分
        pred_cen = outputs['estimation_cen'].permute(0, 2, 1)
        pred_z = outputs['estimation_z'].permute(0, 2, 1)
        pred_dim = outputs['estimation_dim'].permute(0, 2, 1)
        pred_dir = outputs['estimation_dir'].permute(0, 2, 1)
        pred_yaw = torch.atan2(pred_dir[:, :, :1], pred_dir[:, :, 1:])  # y/x
        pred_vel = outputs['estimation_vel'].sigmoid().permute(0, 2, 1)
        pred_score = outputs['score'].sigmoid().permute(
            0, 2, 1)  # 得分，时序特征融合后的得分

        indicee = torch.arange(0, 256).to(pred_vel).float().view(-1, 256, 1)

        track_boxes = torch.cat([pred_cen,
                                 pred_z,
                                 pred_dim,
                                 pred_yaw,
                                 pred_vel*40,
                                 #   template_xyz.view(-1, 256, 2),
                                 #   indicee
                                 ], dim=-1
                                )

        return pred_score, track_boxes
