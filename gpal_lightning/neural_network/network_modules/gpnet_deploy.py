from collections import defaultdict
from typing import Callable, List, Union

import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
import onnxruntime as ort
import torch.nn.functional as F

from gpal_lightning import const
from gpal_lightning.data.dataloader_helpers.gpal_collate import gpal_collate
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.network_modules.gpnet import GpNet
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.models.transformers.od_view_transform import GetProjectGridByEgo2Imgs


def DistGridMap(src_w, src_h, dist, intrins, tgt_w, tgt_h, top_crop_len, top_crop_bgn):
    ws = np.linspace(-1.0, 1.0, src_w,
                     endpoint=True)[np.newaxis, :, np.newaxis].repeat(src_h, 0)
    hs = np.linspace(-1.0, 1.0, src_h,
                     endpoint=True)[:, np.newaxis, np.newaxis].repeat(src_w, 1)
    # cv2.imwrite("ws.jpg", (ws * 127+128).astype(np.uint8))
    # cv2.imwrite("hs.jpg", (hs * 127+128).astype(np.uint8))
    src_map = np.concatenate([ws, hs], axis=-1)
    target_map = cv2.undistort(
        src=src_map, cameraMatrix=intrins, distCoeffs=dist, newCameraMatrix=intrins)
    target_map = cv2.resize(target_map, [tgt_w, tgt_h])
    target_map = target_map[top_crop_bgn:top_crop_len+top_crop_bgn, :]
    return target_map


class GpNetDeploy(GpNet):
    def __init__(
            self,
            global_config: GlobalConfig,
            tasks: List,
            automatic_optimization: bool = False,
            collate_fn: Callable = gpal_collate,
    ):
        super().__init__(global_config, tasks, automatic_optimization, collate_fn)
        self.session = ort.InferenceSession(global_config.onnx_path)
        
        self.calib_data_cnt = 0

    def forward_one_DRIVING_BEV_DYN(self, x, calib, metadata):
        batch_ret = {
            'with_postprocess': True,
            'Points_Loss': {
                'estimation_cen': [],
                'estimation_z': [],
                'estimation_dim': [],
                'estimation_dir': [],
                'estimation_vel': [],
                'estimation_score': [],
                'estimation_score_cls': [],
            }
        }
        save_path = f'calib'
        batch_size = len(metadata)
        for i in tqdm(range(batch_size)):
            # img_slice = torch.stack(
            #     [x[k][i] for k in x], dim=0).float().detach().cpu().numpy()

            img_slice = {k: x[k][i].unsqueeze(0).float().detach().cpu().numpy() for k in x}
            # print(ShowDataStruct("img_slice", img_slice))
            intrins = calib["intrinsic"][i].detach().cpu().numpy()
            cam_dists = calib["cam_dist"][i].detach().cpu().numpy()
            img_crop_dict = calib["img_crop_dict"]
            images_grid = np.stack([DistGridMap(img_slice[k].shape[2],
                                                img_slice[k].shape[1],
                                                cam_dists[ki],
                                                intrins[ki],
                                                int(img_crop_dict["IMAGE_RESIZE"][1][i]),
                                                int(img_crop_dict["IMAGE_RESIZE"][0][i]),
                                                int(img_crop_dict["IMAGE_CROP_H_LEN"][i]),
                                                int(img_crop_dict["CROP_HeSai_ID4"]
                                                    ["CROP_START"][ki][i])
                                                )
                                   for ki, k in enumerate(img_slice)], axis=0)


            # for ki, k in enumerate(img_slice):
            #     # cv2.imwrite(f"img_{k}.jpg", (img_slice[k].squeeze(0)).astype(np.uint8))
            #     src = torch.from_numpy(img_slice[k]).float().cuda().permute(0, 3, 1, 2)
            #     grid = torch.from_numpy(images_grid[ki:ki+1]).float().cuda()
            #     udist_img8ms = F.grid_sample(
            #         src, grid, align_corners=True, padding_mode='border', mode="nearest")
            #     cv2.imwrite(
            #         f"eval_imgs/eval_imgs_{i}_{k}_u2.jpg", (udist_img8ms.squeeze(0)).permute(1, 2, 0).cpu().numpy().astype(np.uint8))

            ego2imgs = calib["ego2imgs"][i]
            xyz_camAX = self.model[self._transformers["DRIVING_BEV_DYN"]].xyz_camA.clone()
            vt_grid, vt_grid_valid = GetProjectGridByEgo2Imgs(
                ego2imgs, H=40, W=96, div=8, Z=4, Y=96, X=240, sample_pts_3d=xyz_camAX.to(ego2imgs.device).clone())
            vt_grid = torch.clip(vt_grid, -1.1, 1.1)
            
            inputs_dict = {}
            inputs_dict.update(img_slice)
            inputs_dict.update({"images_grid": images_grid.astype(np.float32)})
            # inputs_dict.update({"ego2imgs": ego2imgs})
            inputs_dict.update(
                {"vt_grid": vt_grid.float().detach().cpu().numpy(), "vt_grid_valid": vt_grid_valid.float().detach().cpu().numpy()})
            # print(ShowDataStruct("inputs_dict", inputs_dict))
            if self.global_config.calib_data_save_path != "None":
                for k in inputs_dict:
                    single_calib_data_save_path = f'{self.global_config.calib_data_save_path}/{k}/{self.calib_data_cnt}.npy'
                    os.makedirs(os.path.dirname(single_calib_data_save_path), exist_ok=True)
                    np.save(single_calib_data_save_path, inputs_dict[k])
                self.calib_data_cnt+=1

            outputs = self.session.run(None, inputs_dict)
            for k, o in zip(batch_ret["Points_Loss"], outputs):
                batch_ret["Points_Loss"][k].append(o)
        # exit(1)
        for k in batch_ret["Points_Loss"]:
            batch_ret["Points_Loss"][k] = torch.from_numpy(np.stack(
                batch_ret["Points_Loss"][k], axis = 0)).cuda()
        # print(ShowDataStruct("batch_ret", batch_ret))
        return [batch_ret]

    def forward(self, x, calib=None, metadata=None, phase=const.PHASE_TRAINING):
        forward_outputs = []
        for task in self.tasks:
            # print(task)
            # print(ShowDataStruct("x", x))
            # print(ShowDataStruct("calib", calib))
            if "DRIVING_BEV_DYN" == task:
                output = self.forward_one_DRIVING_BEV_DYN(x, calib, metadata)
                forward_outputs.append(output)

        return forward_outputs
        

# pth 先resize
# ====================================================================================================
# Metrics, Range |                 Range 7: [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0]                 |
# ----------------------------------------------------------------------------------------------------
# Metrics        |     car     |    truck    |construction_|   cyclist   |  tricycle   |  destrian   |
# ----------------------------------------------------------------------------------------------------
# num_Dt         |     265     |     40      |      -      |     96      |      -      |     37      |
# num_Gt         |     307     |     46      |      -      |     72      |      -      |     29      |
# max_tp         |     237     |     34      |      -      |     38      |      -      |     10      |
# ----------------------------------------------------------------------------------------------------
# Precision      |   0.8943    |   0.8500    |      -      |   0.3958    |      -      |   0.2703    |
# Recall         |   0.7720    |   0.7391    |      -      |   0.5278    |      -      |   0.3448    |
# R@P0.7         |   0.7720    |   0.7391    |      -      |   0.0000    |      -      |   0.0000    |
# AP             |   0.7589    |   0.6881    |      -      |   0.3009    |      -      |   0.1770    |
# F1             |   0.8287    |   0.7907    |      -      |   0.4524    |      -      |   0.3030    |
# ----------------------------------------------------------------------------------------------------
# Ref_x_mean     |   0.4799    |   0.5007    |      -      |   0.7039    |      -      |   0.1889    |
# Ref_y_mean     |   0.1590    |   0.2746    |      -      |   0.3018    |      -      |   0.2571    |
# E_x_max@0.9    |   2.6402    |   2.3845    |      -      |   2.3872    |      -      |   1.1439    |
# E_y_max@0.9    |   0.6378    |   0.8050    |      -      |   1.2336    |      -      |   1.4019    |
# E_x_mean       |   0.8836    |   0.9556    |      -      |   1.2001    |      -      |   0.5172    |
# E_y_mean       |   0.2881    |   0.3251    |      -      |   0.5589    |      -      |   0.8250    |
# E_z_mean       |   0.0861    |   0.1150    |      -      |   0.0822    |      -      |   0.0841    |
# E_l_mean       |   0.1580    |   1.1215    |      -      |   0.1328    |      -      |   0.1461    |
# E_w_mean       |   0.0669    |   0.1215    |      -      |   0.0710    |      -      |   0.0650    |
# E_h_mean       |   0.0649    |   0.1997    |      -      |   0.0486    |      -      |   0.0837    |
# E_r_mean       |   0.6958    |   0.7580    |      -      |   0.9552    |      -      |   1.6579    |
# E_v_mean       |   2.6882    |   4.1937    |      -      |   2.2381    |      -      |   1.9508    |
# ----------------------------------------------------------------------------------------------------
# ATE            |   0.9835    |   1.0612    |      -      |   1.4298    |      -      |   1.0263    |
# ASE            |   0.1036    |   0.1830    |      -      |   0.1753    |      -      |   0.3152    |
# AOE            |   0.0394    |   0.0355    |      -      |   0.2150    |      -      |   0.5501    |
# AVE            |   2.6882    |   4.1937    |      -      |   2.2381    |      -      |   1.9508    |
# ====================================================================================================

# 全区域: P 匹配的预测框数量: 321/440 = 0.730
# 全区域: R 匹配的预测框数量: 321/473 = 0.679

# onnx 先去畸变
# ====================================================================================================
# Metrics, Range |                 Range 7: [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0]                 |
# ----------------------------------------------------------------------------------------------------
# Metrics        |     car     |    truck    |construction_|   cyclist   |  tricycle   |  destrian   |
# ----------------------------------------------------------------------------------------------------
# num_Dt         |     261     |     37      |      1      |     91      |      -      |     45      |
# num_Gt         |     307     |     46      |      5      |     72      |      -      |     29      |
# max_tp         |     236     |     31      |      1      |     40      |      -      |     11      |
# ----------------------------------------------------------------------------------------------------
# Precision      |   0.9042    |   0.8378    |   1.0000    |   0.4396    |      -      |   0.2444    |
# Recall         |   0.7687    |   0.6739    |   0.2000    |   0.5556    |      -      |   0.3793    |
# R@P0.7         |   0.7687    |   0.6739    |   0.2000    |   0.0694    |      -      |   0.0000    |
# AP             |   0.7546    |   0.6134    |   0.2000    |   0.3379    |      -      |   0.1343    |
# F1             |   0.8310    |   0.7470    |   0.3333    |   0.4908    |      -      |   0.2973    |
# ----------------------------------------------------------------------------------------------------
# Ref_x_mean     |   0.5333    |   0.6528    |   0.5493    |   0.6943    |      -      |   0.2638    |
# Ref_y_mean     |   0.1607    |   0.2260    |   0.1605    |   0.2771    |      -      |   0.2908    |
# E_x_max@0.9    |   2.6446    |   3.0458    |   0.0638    |   2.6052    |      -      |   1.2333    |
# E_y_max@0.9    |   0.5631    |   0.7915    |   0.3227    |   1.1242    |      -      |   1.2838    |
# E_x_mean       |   0.9131    |   1.0579    |   0.0638    |   1.1041    |      -      |   0.6404    |
# E_y_mean       |   0.2701    |   0.2950    |   0.3227    |   0.4678    |      -      |   0.8831    |
# E_z_mean       |   0.0854    |   0.1295    |   0.0758    |   0.0911    |      -      |   0.0883    |
# E_l_mean       |   0.1579    |   0.9852    |   1.2762    |   0.1473    |      -      |   0.1287    |
# E_w_mean       |   0.0668    |   0.1236    |   0.1296    |   0.0707    |      -      |   0.0712    |
# E_h_mean       |   0.0665    |   0.1981    |   0.4486    |   0.0564    |      -      |   0.0870    |
# E_r_mean       |   0.6981    |   0.6226    |   0.0165    |   0.9262    |      -      |   1.2687    |
# E_v_mean       |   2.6661    |   4.3655    |   3.1593    |   2.4490    |      -      |   1.8452    |
# ----------------------------------------------------------------------------------------------------
# ATE            |   0.9994    |   1.1374    |   0.3289    |   1.2896    |      -      |   1.1517    |
# ASE            |   0.1044    |   0.1857    |   0.2546    |   0.1848    |      -      |   0.3057    |
# AOE            |   0.0386    |   0.0306    |   0.0165    |   0.1858    |      -      |   0.3756    |
# AVE            |   2.6661    |   4.3655    |   3.1593    |   2.4490    |      -      |   1.8452    |
# ====================================================================================================

# 全区域: P 匹配的预测框数量: 321/438 = 0.733
# 全区域: R 匹配的预测框数量: 321/473 = 0.679

# pth 先去畸变
# ====================================================================================================
# Metrics, Range |                 Range 7: [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0]                 |
# ----------------------------------------------------------------------------------------------------
# Metrics        |     car     |    truck    |construction_|   cyclist   |  tricycle   |  destrian   |
# ----------------------------------------------------------------------------------------------------
# num_Dt         |     272     |     41      |      -      |     98      |      -      |     43      |
# num_Gt         |     307     |     46      |      -      |     72      |      -      |     29      |
# max_tp         |     241     |     35      |      -      |     40      |      -      |     11      |
# ----------------------------------------------------------------------------------------------------
# Precision      |   0.8860    |   0.8537    |      -      |   0.4082    |      -      |   0.2558    |
# Recall         |   0.7850    |   0.7609    |      -      |   0.5556    |      -      |   0.3793    |
# R@P0.7         |   0.7850    |   0.7609    |      -      |   0.1111    |      -      |   0.0000    |
# AP             |   0.7683    |   0.7049    |      -      |   0.3348    |      -      |   0.1663    |
# F1             |   0.8325    |   0.8046    |      -      |   0.4706    |      -      |   0.3056    |
# ----------------------------------------------------------------------------------------------------
# Ref_x_mean     |   0.5124    |   0.6102    |      -      |   0.7057    |      -      |   0.2638    |
# Ref_y_mean     |   0.1591    |   0.3002    |      -      |   0.3194    |      -      |   0.2974    |
# E_x_max@0.9    |   2.6480    |   2.6604    |      -      |   2.6030    |      -      |   1.2274    |
# E_y_max@0.9    |   0.5988    |   0.8493    |      -      |   1.1711    |      -      |   1.3015    |
# E_x_mean       |   0.9403    |   1.0500    |      -      |   1.0745    |      -      |   0.6056    |
# E_y_mean       |   0.3001    |   0.3416    |      -      |   0.5121    |      -      |   0.8360    |
# E_z_mean       |   0.0865    |   0.1241    |      -      |   0.0859    |      -      |   0.0876    |
# E_l_mean       |   0.1620    |   1.0668    |      -      |   0.1456    |      -      |   0.1363    |
# E_w_mean       |   0.0650    |   0.1203    |      -      |   0.0753    |      -      |   0.0703    |
# E_h_mean       |   0.0655    |   0.2053    |      -      |   0.0518    |      -      |   0.0889    |
# E_r_mean       |   0.7247    |   0.7366    |      -      |   0.8913    |      -      |   1.2711    |
# E_v_mean       |   2.6980    |   4.1769    |      -      |   2.2275    |      -      |   2.0496    |
# ----------------------------------------------------------------------------------------------------
# ATE            |   1.0331    |   1.1563    |      -      |   1.2900    |      -      |   1.0921    |
# ASE            |   0.1038    |   0.1814    |      -      |   0.1870    |      -      |   0.3139    |
# AOE            |   0.0393    |   0.0347    |      -      |   0.2104    |      -      |   0.4163    |
# AVE            |   2.6980    |   4.1769    |      -      |   2.2275    |      -      |   2.0496    |
# ====================================================================================================

# 全区域: P 匹配的预测框数量: 329/456 = 0.721
# 全区域: R 匹配的预测框数量: 329/473 = 0.696