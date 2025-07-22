import torch
import numpy as np
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.losses.loss import BaseLoss
from gpal_lightning.neural_network.tasks.builder import LOSSES
from gpal_nn.tasks.driving_bev_sta.losses.transform_gt import transform_gt_box, shift_polyline_points, shift_polygen_points
from gpal_nn.tasks.driving_bev_sta.losses.map_loss import BaseMapLossCost


def pack_polyline_gt_points(data):
    annos = []
    if 'points' in data['polylines']:
        annos.append(data['polylines']['points'])
    if 'points' in data['edges']:
        annos.append(data['edges']['points'])

    if len(annos) > 0:
        annos = np.concatenate(annos, axis=0)
    return annos


def lane_loss_computation(preds, data, loss_func):
    # bev_embed, all_cls_scores, all_bbox_pred, all_pts_pred = \
    #     preds['bev_embed'], preds['all_cls_scores'], preds['all_bbox_pred'], preds['all_pts_pred']
    bev_embed, all_cls_scores, all_bbox_pred, all_pts_pred = \
        None, preds['all_cls_scores'], preds['all_bbox_preds'], preds['all_pts_preds']
    # num_iter_layer, bs, num_query, score shape
    num_iter, bs, _, pts_per_vector, _ = all_pts_pred.shape

    loss_list = list()
    for k in range(num_iter):
        loss_dict = dict()
        for j in range(bs):
            score_pred, bbox_pred, pts_pred = all_cls_scores[k,
                                                             j], all_bbox_pred[k, j], all_pts_pred[k, j]
            #  [n, 2], [n, 4], [n, 20, 2]

            annos = pack_polyline_gt_points(data[j])
            start_x = 96
            start_y = 16
            # gt ploylines to gt bboxes  [n, 4], [n, 20, 2]
            bboxes_gt, points_gt = transform_gt_box(annos, start_x, start_y,
                                                    num_pts_per_vec=pts_per_vector, y_first=False, device=pts_pred.device)

            # [n,20, 2]->[n, 2, 20, 2]  矢量线翻转建模
            points_gt = shift_polyline_points(points_gt, pts_per_vector)
            # here to loss
            single_loss_dict = loss_func(
                (score_pred, bbox_pred, pts_pred), (bboxes_gt, points_gt))
            # exit(1)

            for key in single_loss_dict.keys():
                if key in loss_dict.keys():
                    loss_dict[key] += single_loss_dict[key]
                else:
                    loss_dict[key] = single_loss_dict[key]

        for key in loss_dict.keys():
            loss_dict[key] /= bs

        loss_list.append(loss_dict)
    total_dict = {}  # loss_list[-1]
    final_total_loss = 0.0
    for k in range(num_iter):
        d_loss_dict = loss_list[k]
        for key in d_loss_dict.keys():
            total_dict[f"lane_d{k}.{key}"] = d_loss_dict[key]

    for key in total_dict.keys():
        # if key.startswith("loss"):
        if "loss" in key:
            final_total_loss += total_dict[key]
    total_dict['total_loss'] = final_total_loss
    return total_dict


def loss_computation(preds, data, loss_func):

    total_dict = {}
    lane_total_dict = lane_loss_computation(preds, data, loss_func)
    total_dict.update(lane_total_dict)

    # if 'arrow_3d_output' in preds['static_3d_loss']:
    #     arrow_total_dict = arrow_loss_computation(preds['static_3d_loss']['arrow_3d_output'], data, {})
    #     total_dict.update(arrow_total_dict)

    return total_dict


@LOSSES.register_module()
class DRIVING_BEV_STALoss(BaseLoss):
    def __init__(self, global_config: GlobalConfig, task_config):

        pc_range = task_config.pc_range
        pc_range = [0, 0, 0, 32.0, 96.0, 0]
        super(DRIVING_BEV_STALoss, self).__init__(pc_range, task_config)
        self.polyline_loss = BaseMapLossCost(2, pc_range, cls_loss_weight=1.0, l1_loss_weight=4.0,
                                             giou_loss_weight=0.01, pts_l1_loss_weight=5.0, pts_dir_loss_weight=0.005)

    def forward(self, preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor) -> dict:
        """

        Args:
            preds: preds tensor from model
            trues: trues tensor from dataloader
            masks: masks tensor from preprocess modules, used for mask invalid labels

        Returns: dictionary contains the loss of current batch.

        """

        # output[] = lane_3d_output

# all_cls_scores'], preds['all_bbox_preds'
        # y = torch.tensor( [0.1314, 0.1696, 0.2066, 0.2199, 0.2761, 0.3148, 0.3648, 0.4160, 0.4479, 0.4948, 0.5780, 0.5887, 0.6450, 0.7211, 0.7590, 0.7782, 0.8092, 0.8711, 0.9319, 0.9711]).to("cuda")
        # x = torch.tensor( np.linspace(0.5, 0.6, 20).tolist()).to("cuda")

        # loss = {'total_loss':   (preds[0]['all_cls_scores'][-1,:,:5,:] - 10.).abs().mean() +
        #                         (preds[0]['all_cls_scores'][-1,:,5:,:] + 10.0).abs().mean() +
        #                         (preds[0]['all_cls_scores'][:-1,:,:,:] + 10.0).abs().mean() +
        #                         (preds[0]['all_pts_preds'][:,:,0,:,0] - x).abs().mean() +
        #                         (preds[0]['all_pts_preds'][:,:,1,:,0] - (x+0.1)).abs().mean() +
        #                         (preds[0]['all_pts_preds'][:,:,2,:,0] - (x-0.1)).abs().mean() +
        #                         (preds[0]['all_pts_preds'][:,:,3,:,0] - (x+0.2)).abs().mean() +
        #                         (preds[0]['all_pts_preds'][:,:,4,:,0] - (x-0.2)).abs().mean() +
        #                         (preds[0]['all_pts_preds'][:,:,:,:,1] - y).abs().mean()
        #         }
        # print(preds[0]['all_cls_scores'][-1,:,:7,:])

        # print(preds[0]['all_cls_scores'].shape, preds[0]['all_pts_preds'].shape)
        # exit(1)
        # print(preds[0]['all_cls_scores'][0,0,0], preds[0]['all_pts_preds'][0,0,0,0])
        # loss = {'total_loss': (preds[0] - 0.3).abs().sum()}
        # print(preds[0])

        # loss_computation(preds[0], trues, self.polyline_loss)
        loss = loss_computation(preds[0], trues, self.polyline_loss)
        # exit(1)
        # loss = {}
        # for attribute_obj in self.task_config.attributes.values():
        #     loss.update(attribute_obj.loss(preds, trues, masks))
        # loss.update({"total_loss": sum(loss.values())})
        return loss
