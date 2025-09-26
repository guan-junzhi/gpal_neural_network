import torch
import numpy as np
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.losses.loss import BaseLoss
from gpal_lightning.neural_network.tasks.builder import LOSSES
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.losses.loss_utils import Compute_Loss, Points_Loss


@LOSSES.register_module()
class DRIVING_BEV_DYNLoss(BaseLoss):
    def __init__(self, global_config: GlobalConfig, task_config):
        super(DRIVING_BEV_DYNLoss,
              self).__init__(global_config, task_config)
        # self.criterion_1 = Crit1_WideRange_L2_Loss(
        #     w1=1.0, w2=1.0, reduction='sum')

    def GtToTorch(self, trues, device):
        gt_torch_batch = {}
        for key in trues[0]:
            if key == "gt_boxes":
                continue
            gt_torch_batch[key] = torch.from_numpy(np.stack(
                [ele[key] for ele in trues], axis=0)).to(device).float()

        return gt_torch_batch

    def ProcessGt(self, trues, preds, num_key_points=256):
        processed_gt = {"batchsize": len(trues)}
        trues = self.GtToTorch(trues, preds["hm_cen"].device)
        # print("C", trues['gt_curr_indices_center'])

        B, C, H, W = trues['gt_curr_hm_cen'].shape
        # gt_curr_hm_cen = torch.stack(
        #     [trues['gt_curr_hm_cen'], trues['gt_prev_hm_cen']], dim=1).view(B*2, C, H, W)  #
        gt_curr_hm_cen = torch.stack(
            [trues['gt_curr_hm_cen']], dim=1).view(B, C, H, W)  #

        processed_gt["gt_curr_hm_cen"] = gt_curr_hm_cen

        # hm_gt = trues["gt_curr_hm_cen"].view(B, C, -1)
        # score_gt = hm_gt

        # hm_pred = preds['hm_cen'].view(
        #     B, C, -1)  # 经过 maxpool 和 == [1, 4, 23040]

        # score, _ = hm_pred.max(dim=1)  # 帧预测的热力图的通道最大值
        # _, indice_topk = torch.topk(score, k=num_key_points, dim=-1)
        # indice_topk = indice_topk.view(B, -1)

        # score_gt_topk = score_gt[torch.arange(B)[:, None, None],
        #                          torch.arange(score_gt.shape[1])[
        #     None, :, None],
        #     indice_topk.reshape(B, 1, -1).repeat(1, score_gt.shape[1], 1)]
        # processed_gt['score'] = score_gt_topk.view(
        #     B, -1, 1, num_key_points)  # -> [1, 4, 1, 256]  # 只是用当前帧的

        # mode_gt = "gt_curr_"
        # for bs_idx in range(B):
        #     indice_topk_single = indice_topk[bs_idx]
        #     gt_mask = trues[mode_gt + 'obj_mask'][bs_idx]
        #     track_ind = trues[mode_gt +
        #                       'indices_center'][bs_idx].clone()
        #     for idx in range(num_key_points):
        #         if gt_mask[idx] == 0:   # 当前帧没有真值点，跳过
        #             continue
        #         # 有真值点，但没有预测匹配上
        #         if (indice_topk_single == track_ind[idx]).sum() < 1:
        #             trues[mode_gt + 'obj_mask'][bs_idx][idx] = 0  #
        #             trues[mode_gt + 'indices_center'][bs_idx][idx] = 0
        #         else:  # 有真值点，有预测匹配上
        #             key_mask = indice_topk_single == track_ind[idx]
        #             key_range = torch.arange(
        #                 0, num_key_points, device=indice_topk_single.device)[key_mask]
        #             trues[mode_gt +
        #                   'indices_center'][bs_idx][idx] = key_range[0]

        processed_gt['track_cen_offset'] = trues['gt_curr_cen_offset']
        processed_gt['track_dim'] = trues['gt_curr_dim']
        processed_gt['track_direction'] = trues['gt_curr_direction']
        processed_gt['track_multibin_direction'] = trues['gt_curr_multibin_direction']
        processed_gt['track_z_coor'] = trues['gt_curr_z_coor']
        processed_gt['track_vel'] = trues['gt_curr_vel']
        processed_gt['track_obj_mask'] = trues['gt_curr_obj_mask']
        processed_gt['track_indices_center'] = trues['gt_curr_indices_center']
    
        # print("A", trues['gt_curr_indices_center'])
        
        return processed_gt

    def AddWeight(self, tb_dict):
        new_tb_dict = {}
        tot_loss = 0.0
        for loss_name, loss_value in tb_dict.items():
            if 'tot' in loss_name:
                continue

            # --- occ loss
            if 'occ_loss' in loss_name and 'hm' in loss_name:
                weight_loss = loss_value * 0.5
            elif 'occ_loss' in loss_name and 'pts_bev' in loss_name:
                weight_loss = loss_value * 3.0
            elif 'occ_loss' in loss_name and 'vel' in loss_name:
                weight_loss = loss_value * 10.0

            # --- track loss
            elif 'track_loss' in loss_name and 'dir' in loss_name:
                weight_loss = loss_value * 5.0
            elif 'track_loss' in loss_name and 'hm' in loss_name:
                weight_loss = loss_value * 5.0
            elif 'track_loss' in loss_name and 'vel' in loss_name:  # TODO 暂时
                weight_loss = loss_value * 1.0
            elif 'track_loss' in loss_name and 'score' in loss_name:  # TODO 暂时
                weight_loss = loss_value * 0.5
            elif 'track_loss' in loss_name:
                weight_loss = loss_value * 1.0

            # --- 2d loss
            elif '2d_loss' in loss_name:
                weight_loss = loss_value * 1.0

            # --- other loss
            else:
                raise NotImplementedError

            tot_loss += weight_loss

            new_tb_dict[loss_name] = weight_loss.item()

        return tot_loss, new_tb_dict

    def forward(self, preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor) -> dict:
        """

        Args:
            preds: preds tensor from model
            trues: trues tensor from dataloader
            masks: masks tensor from preprocess modules, used for mask invalid labels

        Returns: dictionary contains the loss of current batch.

        """
        # for ele_i, ele in enumerate(trues):
        #     print("B", ele['gt_curr_indices_center'])

        # print(ShowDataStruct("trues", trues))
        # print(ShowDataStruct("preds", preds))
        # import cv2
        # for ele_i, ele in enumerate(trues):
        #     hm_cen = ele["gt_curr_hm_cen"].max(axis = 0)
        #     print(f"pred: {hm_cen.min()} {hm_cen.max()}")
        #     cv2.imwrite(
        #         f"heatmap_vis/gt_curr_hm_cen_{ele_i}.jpg", (hm_cen * 254).astype(np.uint8))

        # for ele_i, (ele, xyz, gt) in enumerate(zip(preds[0]["hm_cen"], preds[0]["Points_Loss"]["template_xyz"], trues)):
        #     hm_cen = ele.max(axis = 0)[0].sigmoid().detach().cpu().numpy()
        #     print(f"pred: {hm_cen.min()} {hm_cen.max()}")

        #     hm_cen = (np.stack([hm_cen, hm_cen, hm_cen], axis = -1) * 254).astype(np.uint8)

        #     point_cloud_range = [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0]
        #     voxel_size = [0.64, 0.64, 0.5]

        #     for p in xyz:
        #         y = int((p[0] - point_cloud_range[1]) / voxel_size[1])
        #         x = int((p[1] - point_cloud_range[0]) / voxel_size[0])
        #         cv2.circle(hm_cen, (x, y), 1, [255, 0, 0], -1)
        #         # ys = ys.view(1, H, W) * self.voxel_size[1] + self.point_cloud_range[1]
        #         # xs = xs.view(1, H, W) * self.voxel_size[0] + self.point_cloud_range[0]
            
        #     gt_hm_cen = gt["gt_curr_hm_cen"].max(axis=0)
        #     gt_hm_cen = (gt_hm_cen * 254).astype(np.uint8)

        #     hm_cen[..., -1] = gt_hm_cen

        #     cv2.imwrite(
        #         f"heatmap_vis/pred_curr_hm_cen_{ele_i}.jpg", hm_cen)

        #     hm_cen2 = np.zeros([96, 240], dtype = np.uint8)
        #     hm_cen2 = hm_cen2.flatten()
        #     hm_cen2[gt["gt_curr_indices_center"]] = 255
        #     hm_cen2 = hm_cen2.reshape(96, 240)
        #     cv2.imwrite(
        #         f"heatmap_vis/gt_curr_indices_center_{ele_i}.jpg", hm_cen2)
            

        #     print(gt["gt_curr_indices_center"])

        processed_gt = self.ProcessGt(trues, preds[0])

        loss = {}
        loss_hm, tb_dict = Compute_Loss()(preds[0], processed_gt)
        loss['track_loss_hm'] = loss_hm
        loss.update(tb_dict)
        # tb_dict = Points_Loss()(preds[0], processed_gt)
        # loss.update(tb_dict)

        total_loss, loss = self.AddWeight(loss)
        loss['total_loss'] = total_loss

        return loss
