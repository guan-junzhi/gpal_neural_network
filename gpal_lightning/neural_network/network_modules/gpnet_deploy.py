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
from gpal_lightning.utils.deploy_utils import bgr_to_nv12_split, rgb_to_nv12_split, DistGridMap
from horizon_tc_ui import HBRuntime

from tools_scripts.driving_bev_sta.create_images_grid import PreproModule
from gpal_nn.models.transformers.bevformer.view_transformer import SingleBevFormerViewTransformer


class GpNetDeploy(GpNet):
    def __init__(
            self,
            global_config: GlobalConfig,
            tasks: List,
            automatic_optimization: bool = False,
            collate_fn: Callable = gpal_collate,
    ):
        super().__init__(global_config, tasks, automatic_optimization, collate_fn)
        self.global_config = global_config
        self.session = HBRuntime(self.global_config.onnx_path)
        self.output_names = self.session.output_names

        self.model_file = global_config.onnx_path
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
            if "quantized_model.bc" in self.model_file:
                # 添加 bgr 2 nv12 转换
                print("nv12 input ...")
                img_slice_nv12 = {}
                for img_name, img_data in img_slice.items():
                    y_data, uv_data = bgr_to_nv12_split(img_data)
                    img_slice_nv12[f"{img_name}_y"] = y_data
                    img_slice_nv12[f"{img_name}_uv"] = uv_data
                inputs_dict.update(img_slice_nv12)
            else:
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

            self.session = HBRuntime(self.global_config.onnx_path)
            outputs = self.session.run(self.output_names, inputs_dict)
            for k, o in zip(batch_ret["Points_Loss"], outputs):
                batch_ret["Points_Loss"][k].append(o)
        # exit(1)
        for k in batch_ret["Points_Loss"]:
            batch_ret["Points_Loss"][k] = torch.from_numpy(np.stack(
                batch_ret["Points_Loss"][k], axis = 0)).cuda()
        # print(ShowDataStruct("batch_ret", batch_ret))
        return [batch_ret]
    
    def forward_park_slot(self, x, calib, metadata):
        batch_size = len(metadata)
        for i in tqdm(range(batch_size)):
            img_slice = {k: x[k][i].unsqueeze(0).float().detach().cpu().numpy() for k in x}
            # print(ShowDataStruct("img_slice", img_slice))
        inputs_dict = {}
        if "quantized_model.bc" in self.model_file:
                # 添加 bgr 2 nv12 转换
            print("nv12 input ...")
            img_slice_nv12 = {}
            for img_name, img_data in img_slice.items():
                y_data, uv_data = bgr_to_nv12_split(img_data)
                img_slice_nv12[f"{img_name}_y"] = y_data
                img_slice_nv12[f"{img_name}_uv"] = uv_data
            inputs_dict.update(img_slice_nv12)
        else:
            inputs_dict.update(img_slice)
            if self.global_config.calib_data_save_path != "None":
                for k in inputs_dict:
                    single_calib_data_save_path = f'{self.global_config.calib_data_save_path}/{k}/{self.calib_data_cnt}.npy'
                    os.makedirs(os.path.dirname(single_calib_data_save_path), exist_ok=True)
                    np.save(single_calib_data_save_path, inputs_dict[k])
                self.calib_data_cnt+=1

            self.session = HBRuntime(self.global_config.onnx_path)
            outputs = self.session.run(self.output_names, inputs_dict)
        

    def forward_one_DRIVING_BEV_STA(self, x, calib, metadata):
        """Single-frame STA deploy forward.
        Builds images_grid and BEV sampling grids using the same logic as
        tools_scripts/driving_bev_sta/to_npy_for_calib.py, then runs ONNX.
        """
        self.phase = metadata[0]['eval_phase']
        out_keys = [
                'all_cls_scores',
                'all_pts_preds',
                'all_lane_marking_types_preds',
                'all_lane_marking_colors_preds',
                'all_shape_types_preds',
                'all_centerline_types_preds',
                'all_keypoint_classes_preds',
                'all_keypoint_regs_preds',
            ]
        batch_ret ={}
        for k in out_keys:
            batch_ret[k] = []
        batch_size = len(metadata)
        for i in tqdm(range(batch_size)):
            img_slice = {k: x[k][i].unsqueeze(0).float().detach().cpu().numpy() * 255.0 for k in x}
            # Intrinsics and distortions
            ks = calib['_Ks'][i].detach().cpu().numpy()
            dists = calib['_dists'][i].detach().cpu().numpy()
            cut_start_h = self.global_config.Tasks[self.global_config.tasks[0]]['datasets']['validation'][0]['cut_start_h']
            ori_h, ori_w = int(metadata[i]['ori_shape'][0][0]), int(metadata[i]['ori_shape'][0][1])
            dst_h, dst_w = 540, 960
            # images_grid
            pp = PreproModule()
            images_grid = pp.export_input(
                batch_size=1,
                K1=ks[0], dists1=dists[0],
                K2=ks[1], dists2=dists[1],
                ori_shape=(ori_h, ori_w),
                dst_h=dst_h, dst_w=dst_w,
                cut_start_h=cut_start_h
            ).detach().cpu().numpy()

            if not hasattr(self, '_sta_vt'):
                transformer_cfg = self.global_config.Transformer.get('transformer_config', {})
                self._sta_vt = SingleBevFormerViewTransformer(self.global_config, transformer_cfg)
                self._sta_vt_device = torch.device('cpu')
                self._sta_vt.to(self._sta_vt_device)
                self._sta_vt.eval()
                _, self._sta_ref3d = self._sta_vt.export_reference_points(bs=1, device=self._sta_vt_device)

            ego2imgs = calib['ego2imgs'][i]
            if isinstance(ego2imgs, np.ndarray):
                ego2imgs_t = torch.from_numpy(ego2imgs)
            else:
                ego2imgs_t = ego2imgs

            if ego2imgs_t.dim() == 2 and tuple(ego2imgs_t.shape) == (4, 4):
                ego2imgs_t = ego2imgs_t.view(1, 1, 4, 4)
            elif ego2imgs_t.dim() == 1 and ego2imgs_t.numel() == 16:
                ego2imgs_t = ego2imgs_t.view(1, 1, 4, 4)
            elif ego2imgs_t.dim() == 3:
                # (N_cam, 4, 4)
                if tuple(ego2imgs_t.shape[-2:]) == (4, 4):
                    ego2imgs_t = ego2imgs_t.unsqueeze(0)
                else:
                    n_cam = ego2imgs_t.numel() // 16
                    ego2imgs_t = ego2imgs_t.view(1, n_cam, 4, 4)
            elif ego2imgs_t.dim() == 4:
                # Expect (B, N_cam, 4, 4); if channels-first by mistake, try to fix
                if tuple(ego2imgs_t.shape[-2:]) != (4, 4):
                    # Fallback infer N_cam from numel with B=1
                    n_cam = ego2imgs_t.numel() // 16
                    ego2imgs_t = ego2imgs_t.view(1, n_cam, 4, 4)
            else:
                # Generic fallback: infer from numel with B=1
                n_cam = ego2imgs_t.numel() // 16
                ego2imgs_t = ego2imgs_t.view(1, n_cam, 4, 4)

            ego2imgs_t = ego2imgs_t.to(device=self._sta_vt_device, dtype=torch.float32)

            bev_real2aug_t = torch.eye(4, dtype=torch.float32, device=self._sta_vt_device)

            (
                reference_points_rebatch,
                queries_rebatch_grid,
                restore_bev_grid,
                bev_pillar_counts,
            ) = self._sta_vt.point_sampling(
                reference_points=self._sta_ref3d,
                pc_range=self._sta_vt.pc_range,
                img_metas={'ego2imgs': ego2imgs_t},
                im_shape=(dst_h - cut_start_h, dst_w),
                bev_real2aug=bev_real2aug_t,
            )
            inputs_dict = {}
            img_input = {'img_front_30': 'img_30', 'img_front_120': 'img_120'}
            if ".bc" in self.model_file or '.hbm' in self.model_file:
                img_slice_nv12 = {}
                for img_name, img_data in img_slice.items():
                    y_data, uv_data = rgb_to_nv12_split(img_data)
                    img_slice_nv12[f"{img_input[img_name]}_y"] = y_data
                    img_slice_nv12[f"{img_input[img_name]}_uv"] = uv_data
                inputs_dict.update(img_slice_nv12)
            elif ".onnx" in self.model_file:
                for img_name, img_data in img_slice.items():
                    inputs_dict.update({f"{img_input[img_name]}": img_data})

            inputs_dict.update({
                'images_grid': images_grid.astype(np.float32),
                'queries_rebatch_grid': queries_rebatch_grid.detach().cpu().numpy(),
                'reference_points_rebatch': reference_points_rebatch.detach().cpu().numpy(),
                'restore_bev_grid': restore_bev_grid.detach().cpu().numpy(),
                'bev_pillar_counts': bev_pillar_counts.detach().cpu().numpy(),
            })

            if getattr(self.global_config, 'calib_data_save_path', "None") != "None":
                for k in inputs_dict:
                    single_calib_data_save_path = f'{self.global_config.calib_data_save_path}/{k}/{self.calib_data_cnt}.npy'
                    os.makedirs(os.path.dirname(single_calib_data_save_path), exist_ok=True)
                    np.save(single_calib_data_save_path, inputs_dict[k])
                self.calib_data_cnt += 1
            # self.session = ort.InferenceSession(self.global_config.onnx_path)
            # outputs = self.session.run(self.session.output_names, inputs_dict)
            self.session = HBRuntime(self.global_config.onnx_path)
            outputs = self.session.run(self.output_names, inputs_dict)

            for k, o in zip(out_keys, outputs):
                batch_ret[k].append(o)

        # Stack to tensors
        for k in out_keys:
            batch_ret[k] = torch.from_numpy(np.stack(batch_ret[k], axis=0)).cuda()
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
            elif "DRIVING_BEV_STA" == task:
                output = self.forward_one_DRIVING_BEV_STA(x, calib, metadata)
                forward_outputs.append(output)
            elif "PARKING_IPM_STA" == task:
                output = self.forward_park_slot(x, calib, metadata)
                forward_outputs.append(output)

        return forward_outputs
