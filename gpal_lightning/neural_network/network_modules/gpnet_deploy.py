from collections import defaultdict
from typing import Callable, List, Union

import os
import cv2
import copy
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
from gpal_nn.models.transformers.od_view_transform import (GetProjectGridByEgo2Imgs, gridcloud3d, 
                                                           GetProjectGridByEgo2ImgsFisheye)
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
        
        
        self.dyn_od_stream_feature_bank = None
        self.dyn_od_stream_metas_bank = None

        for task in self.tasks:
            if "DRIVING_BEV_DYN" == task:
                self.xyz_camA = self.gen_xyz_camA()
                self.image_crop_config = global_config.Tasks['DRIVING_BEV_DYN']['image_crop_config']
                self.subtask_name = self.global_config.Tasks['DRIVING_BEV_DYN'].get("SWITCH_SUBTASK", "DRIVING_BEV_DYN")

    def gen_xyz_camA(self):
        transformer_config = self.global_config.Transformer["transformer_config"]
        
        self.point_cloud_range = transformer_config["bev_map_range"]
        self.voxel_size = transformer_config["bev_map_voxel_size"]
        
        self.grid_size = [int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.voxel_size[0]),
                          int((self.point_cloud_range[4]-self.point_cloud_range[1])/self.voxel_size[1])]
        
        xyz_camA = gridcloud3d(
            1, 1, self.grid_size[1], self.grid_size[0], norm=False, device='cpu')
        xyz_camA[:, :, 0] = xyz_camA[:, :, 0] * self.voxel_size[0] + \
            self.voxel_size[0]/2 + self.point_cloud_range[0]
        xyz_camA[:, :, 1] = xyz_camA[:, :, 1] * self.voxel_size[1] + \
            self.voxel_size[1]/2 + self.point_cloud_range[1]
        xyz_camA[:, :, 2] = xyz_camA[:, :, 2] * self.voxel_size[2] + \
            self.voxel_size[2]/2 + self.point_cloud_range[2]
        re_xyz_camA = xyz_camA[:,:,[0,1,3],:]
        
        return re_xyz_camA


    def GetCur2Prev(self, cur_metas, dts):
        B = len(cur_metas)
        rt = torch.zeros([B,3,3], device = "cpu", dtype = torch.float)
        rt[:,0,0] = 1.0
        rt[:,1,1] = 1.0
        rt[:,2,2] = 1.0
        yaw_batch = torch.tensor([ele["ego_yaw_rate"] for ele in cur_metas])
        speed_batch = torch.tensor([ele["ego_speed"] for ele in cur_metas])

        final_yaw = dts * yaw_batch
        speed_vec = final_yaw * 0.5
        pos_x = torch.cos(speed_vec) * speed_batch * dts
        pos_y = torch.sin(speed_vec) * speed_batch * dts

        rt[:,0,0] = torch.cos(final_yaw)
        rt[:,1,1] = torch.cos(final_yaw)
        rt[:,0,1] = -torch.sin(final_yaw)
        rt[:,1,0] = torch.sin(final_yaw)
        rt[:,0,2] = pos_x
        rt[:,1,2] = pos_y

        return rt

    def SeqCheck(self, prev_metas, cur_metas, tth = 0.25):
        B = len(cur_metas)
        seq_flag = torch.zeros(B, device = "cpu", dtype = torch.bool)
        dt = torch.zeros(B, device = "cpu", dtype = torch.float)
        if prev_metas is None:
            return seq_flag, dt

        for i, (m_p, m_c) in enumerate(zip(prev_metas, cur_metas)):
            clip_p = m_p["clip_id"]
            clip_c = m_c["clip_id"]
            ts_p = float(m_p["timestamp"])
            ts_c = float(m_c["timestamp"])

            flag = ((ts_c - ts_p) > 0.0) and ((ts_c - ts_p) < tth) and (clip_p == clip_c)
            seq_flag[i] = flag
            if flag:
                dt[i] = (ts_c - ts_p)
        return seq_flag, dt


    def gen_shift_feature_grid(self, grid, cur2prev, prev_feat, bev_h_resolution, bev_w_resolution):
        bs, _, h, w = prev_feat.shape
        grid = grid.view(bs, h, w, 3, 1)
        if torch.onnx.is_in_onnx_export():
            grid = cur2prev.matmul(grid)
        else:
            for idx in range(bs):
                grid[idx] = cur2prev[idx].matmul(grid[idx])
        # bev2feat
        grid_x = (grid[..., 0, 0].clone() - self.point_cloud_range[0]) / bev_w_resolution
        grid_y = (grid[..., 1, 0].clone() - self.point_cloud_range[1]) / bev_h_resolution
        grid[..., 0, 0] = grid_x.clone()
        grid[..., 1, 0] = grid_y.clone()
        # grid = torch.cat([grid_x.clone(), grid_y.clone()], dim = -1).unsqueeze(-1)
        # todo 需要仔细分辨一下应该用哪个
        normalize_factor = torch.tensor([w, h],
                                        dtype=prev_feat.dtype,
                                        device=prev_feat.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        # output = F.grid_sample(prev_feat, grid.to(
        #     prev_feat.dtype), align_corners=False)
        return grid

    
    def forward_one_DRIVING_BEV_DYN_fisheye(self, x, calib, metadata):
        def create_composite_grid_map(src_w, src_h, norm=True):
            """
            创建复合网格映射，将两个变换序列合并为一个
            变换序列：1920x1536 → resize(2214x1772) → crop(1080x1920) → resize(960x540) → crop(512x960)
            正确实现：通过像素坐标计算，然后转换为归一化坐标
            """
            # 目标图像尺寸
            tgt_h, tgt_w = 512, 960
            
            if norm:
                # 生成目标图像(512x960)的归一化坐标网格
                ws = np.linspace(-1.0, 1.0, tgt_w, endpoint=True)[np.newaxis, :, np.newaxis].repeat(tgt_h, 0)
                hs = np.linspace(-1.0, 1.0, tgt_h, endpoint=True)[:, np.newaxis, np.newaxis].repeat(tgt_w, 1)
            else:
                ws = np.linspace(0.0, tgt_w-1.0, tgt_w, endpoint=True)[np.newaxis, :, np.newaxis].repeat(tgt_h, 0)
                hs = np.linspace(0.0, tgt_h-1.0, tgt_h, endpoint=True)[:, np.newaxis, np.newaxis].repeat(tgt_w, 1)
            
            # 目标图像坐标网格
            target_map = np.concatenate([ws, hs], axis=-1)
            
            # 转换为像素坐标进行计算
            if norm:
                # 将归一化坐标转换为像素坐标
                pixel_map = (target_map + 1.0) * 0.5
                pixel_map[..., 0] *= tgt_w  # x坐标
                pixel_map[..., 1] *= tgt_h  # y坐标
            else:
                pixel_map = target_map.copy()
            
            # 第一步逆变换：逆裁剪(512x960 → 540x960)
            # 裁剪从0开始，裁剪长度为512，所以需要将坐标映射回540x960
            pixel_map[..., 1] = pixel_map[..., 1] + 0  # 裁剪起始位置为0，不需要偏移
            
            # 第二步逆变换：逆resize(540x960 → 1080x1920)
            pixel_map[..., 0] = pixel_map[..., 0] * (1920 / 960)  # x方向缩放
            pixel_map[..., 1] = pixel_map[..., 1] * (1080 / 540)  # y方向缩放
            
            # 第三步逆变换：逆裁剪(1080x1920 → 1772x2214)
            # 裁剪起始位置：y=314, x=127
            pixel_map[..., 1] = pixel_map[..., 1] + 314  # y方向偏移
            pixel_map[..., 0] = pixel_map[..., 0] + 127  # x方向偏移
            
            # 第四步逆变换：逆resize(1772x2214 → 1536x1920)
            pixel_map[..., 0] = pixel_map[..., 0] * (1920 / 2214)  # x方向缩放
            pixel_map[..., 1] = pixel_map[..., 1] * (1536 / 1772)  # y方向缩放
            
            # 将像素坐标转换回归一化坐标
            if norm:
                # 转换为归一化坐标
                target_map[..., 0] = (pixel_map[..., 0] / src_w) * 2 - 1  # x坐标归一化
                target_map[..., 1] = (pixel_map[..., 1] / src_h) * 2 - 1  # y坐标归一化
            else:
                target_map = pixel_map
            
            return target_map
        
        
        def DistGridMapFisheyeID4(src_w, src_h, dist, intrins, tgt_w, tgt_h, top_crop_len, top_crop_bgn, norm=True):
            if norm:
                ws = np.linspace(-1.0, 1.0, src_w,
                                endpoint=True)[np.newaxis, :, np.newaxis].repeat(src_h, 0)
                hs = np.linspace(-1.0, 1.0, src_h,
                                endpoint=True)[:, np.newaxis, np.newaxis].repeat(src_w, 1)
            else:
                ws = np.linspace(0.0, src_w-1.0, src_w,
                                endpoint=True)[np.newaxis, :, np.newaxis].repeat(src_h, 0)
                hs = np.linspace(0.0, src_h-1.0, src_h,
                                endpoint=True)[:, np.newaxis, np.newaxis].repeat(src_w, 1)
            # cv2.imwrite("ws.jpg", (ws * 127+128).astype(np.uint8))
            # cv2.imwrite("hs.jpg", (hs * 127+128).astype(np.uint8))
            src_map = np.concatenate([ws, hs], axis=-1)
            # target_map = cv2.undistort(
            #     src=src_map, cameraMatrix=intrins, distCoeffs=dist, newCameraMatrix=intrins)
            target_map = src_map
            target_map = cv2.resize(target_map, [tgt_w, tgt_h])
            target_map = target_map[top_crop_bgn:top_crop_len+top_crop_bgn, :]
            return target_map
            
            
        
        batch_ret = {
            'fish_head_conv': [],
            'fish_hm_center': [],
            'fish_prev_feats_output': [],
               
        }
        save_path = f'calib'
        batch_size = B = len(metadata)

        if (self.dyn_od_stream_feature_bank == None):
            self.dyn_od_stream_feature_bank = torch.zeros(B, 128, 48, 60).cuda()

        # 矩阵 and torch, 再分发到batch
        seq_flags, dts = self.SeqCheck(self.dyn_od_stream_metas_bank, metadata)
        rts = self.GetCur2Prev(metadata, dts)
        feats_shifted_grid = self.gen_shift_feature_grid(
            self.xyz_camA.repeat(B, 1, 1, 1).to(self.dyn_od_stream_feature_bank.device).clone(), 
            rts.to(self.dyn_od_stream_feature_bank.device), 
            self.dyn_od_stream_feature_bank.clone(), 
            self.voxel_size[0], 
            self.voxel_size[1]
        )
        
        ONLINE_HW = (1536, 1920)
        OFFLINE_HW = (1080, 1920)
        
        image_crop_config = copy.deepcopy(self.image_crop_config)
        image_crop_config['CROP_HeSai_ID4']['CROP_START'] = [0, 0, 0, 0]  # ATTENTION: 泛化车和实际车都需要crop_start为0，统一为0
        
        x_draw = {k: [] for k in x}
        for i in tqdm(range(batch_size)):
            
            # 输入到onnx的 255 1HW3
            img_slice = {k: x[k][i].unsqueeze(0).float().detach().cpu().numpy() for k in x}

            # L4为原图
            if 'SKYWELL' in metadata[i]['clip_id']:
                # curr_view_img = cv2.resize(curr_view_img, (2214, 1772))
                # curr_view_img = curr_view_img[314:314 + 1080]
                # curr_view_img = curr_view_img[:, 127:127 + 1920].transpose(2, 0, 1)
                # x[img_name] = curr_view_img.to(img_tensor) / 255.0
                images_grid = np.stack([
                    create_composite_grid_map(
                    img_slice[k].shape[2],
                    img_slice[k].shape[1],
                    
                ) for img_i, k in enumerate(img_slice)], axis=0)
                
            # ID4鱼眼尺寸是 1080*1920, 需要pad到 1536*1920, 和线上对齐
            # elif 'HeSai' in metadata[i]['clip_id']:
            else:
                H_gap = ONLINE_HW[0] - OFFLINE_HW[0]
                img_slice = {k: np.concatenate([img_slice[k], np.zeros_like(img_slice[k])[:, :H_gap, ...]], axis=1) for k in img_slice}
                images_grid = np.stack([
                    DistGridMapFisheyeID4(
                    # src_w, src_h, dist, intrins, tgt_w, tgt_h, top_crop_len, top_crop_bgn, norm=True
                    img_slice[k].shape[2],
                    img_slice[k].shape[1],
                    None,
                    None,
                    int(ONLINE_HW[1]/2),
                    int(ONLINE_HW[0]/2),
                    int(image_crop_config["IMAGE_CROP_H_LEN"]),
                    int(image_crop_config['CROP_HeSai_ID4']['CROP_START'][img_i]),
                ) for img_i, k in enumerate(img_slice)], axis=0)
            
            # 为可视化
            curr_bs_i_tensor_cat = torch.concat([torch.from_numpy(img_slice[k]).to(self.dyn_od_stream_feature_bank) 
                                                 for k in img_slice], dim=0).permute(0, 3, 1, 2) # BC HW
            curr_bs_i_tensor = F.grid_sample(curr_bs_i_tensor_cat, 
                                             torch.from_numpy(images_grid).float().to(curr_bs_i_tensor_cat.device), 
                                             align_corners=True, padding_mode='border', mode="nearest") / 255.0
            for draw_img_i, img_name in enumerate(x_draw.keys()):
                x_draw[img_name].append(curr_bs_i_tensor[[draw_img_i]])
            
            extrinsic_matrix = calib['extrinsic'][i]
            distortion_coeffs= calib['cam_dist'][i]
            intrinsic_matrix = calib['intrinsic'][i]
            H, W, div, Z, Y, X = 64, 120, 8, 4, 48, 60
            xyz_camAX = self.model[self._transformers["DRIVING_BEV_DYN"]].xyz_camA.clone()
            
            vt_grid, vt_grid_valid = GetProjectGridByEgo2ImgsFisheye(
                extrinsic_matrix,
                distortion_coeffs,
                intrinsic_matrix,
                H, W, div, Z, Y, X,
                xyz_camAX.to(extrinsic_matrix.device).clone(),
                image_crop_config=image_crop_config,
                )
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
            
            bank_feats = self.dyn_od_stream_feature_bank[i].unsqueeze(0).float().detach().cpu().numpy()
            prev_feats_grid = feats_shifted_grid[i].unsqueeze(0).float().detach().cpu().numpy()
            
            prev_feats = bank_feats * seq_flags[i].cpu().numpy()
            
            inputs_dict.update({
                "fish_images_grid": images_grid.astype(np.float32),
                "fish_vt_grid": vt_grid.float().detach().cpu().numpy(),
                "fish_prev_feats": prev_feats,
                "fish_prev_feats_grid": prev_feats_grid,  # 部署外挂计算
                }
            )

            if self.global_config.calib_data_save_path != "None":

                if self.calib_data_cnt %10 == 0:
                    for k in inputs_dict:
                        single_calib_data_save_path = f'{self.global_config.calib_data_save_path}/{k}/{self.calib_data_cnt}.npy'
                        os.makedirs(os.path.dirname(single_calib_data_save_path), exist_ok=True)
                        np.save(single_calib_data_save_path, inputs_dict[k])
                self.calib_data_cnt+=1

            self.session = HBRuntime(self.global_config.onnx_path)
            outputs = self.session.run(self.output_names, inputs_dict)
            for k, o in zip(batch_ret, outputs):
                batch_ret[k].append(o)

        # 可视化, 同时验证image grid 是否正确
        for k in x_draw:
            x_draw[k] = torch.concat(x_draw[k], dim=0)
        for key_name in x:
            x[key_name] = x_draw[key_name]

        for k in batch_ret:
            batch_ret[k] = torch.from_numpy(np.concatenate(batch_ret[k], axis = 0)).cuda()
        # breakpoint()
        self.dyn_od_stream_feature_bank = batch_ret['fish_prev_feats_output'].clone()
        self.dyn_od_stream_metas_bank = copy.deepcopy(metadata)
        
        # 去往topk
        batch_ret = {
            'head_conv': batch_ret['fish_head_conv'],
            'hm_cen': batch_ret['fish_hm_center'],
            'prev_feats_output': batch_ret['fish_prev_feats_output'],
        }
        
        return [batch_ret]
        
    def forward_one_DRIVING_BEV_DYN(self, x, calib, metadata):
        
        if self.subtask_name in ['DRIVING_BEV_DYN_FISHEYE']:
            data_dict = self.forward_one_DRIVING_BEV_DYN_fisheye(x, calib, metadata)
            return data_dict
        
        batch_ret = {
            'head_conv': [],
            'hm_cen': [],
            'prev_feats_output': [],
               
        }
        save_path = f'calib'
        batch_size = B = len(metadata)

        if (self.dyn_od_stream_feature_bank == None):
            self.dyn_od_stream_feature_bank = torch.zeros(B, 128, 48, 120).cuda()

        # 矩阵 and torch, 再分发到batch
        seq_flags, dts = self.SeqCheck(self.dyn_od_stream_metas_bank, metadata)
        rts = self.GetCur2Prev(metadata, dts)
        feats_shifted_grid = self.gen_shift_feature_grid(
            self.xyz_camA.repeat(B, 1, 1, 1).to(self.dyn_od_stream_feature_bank.device).clone(), 
            rts.to(self.dyn_od_stream_feature_bank.device), 
            self.dyn_od_stream_feature_bank.clone(), 
            self.voxel_size[0], 
            self.voxel_size[1]
        )
        
        
        # breakpoint()
        x_draw = {k: [] for k in x}
        for i in tqdm(range(batch_size)):

            img_slice = {k: x[k][i].unsqueeze(0).float().detach().cpu().numpy() for k in x}
            # print(ShowDataStruct("img_slice", img_slice))
            intrins = calib["intrinsic"][i].detach().cpu().numpy()
            cam_dists = calib["cam_dist"][i].detach().cpu().numpy()
            img_crop_dict = self.image_crop_config
            
            scale = copy.deepcopy(metadata[i]["scale"])
            crop_start = copy.deepcopy(metadata[i]["crop"])
            
            images_grid = np.stack([DistGridMap(img_slice[k].shape[2],
                                                img_slice[k].shape[1],
                                                cam_dists[ki],
                                                intrins[ki],
                                                int(img_crop_dict["IMAGE_RESIZE"][1]),
                                                int(img_crop_dict["IMAGE_RESIZE"][0]),
                                                int(img_crop_dict["IMAGE_CROP_H_LEN"]),
                                                # int(img_crop_dict["CROP_HeSai_ID4"]["CROP_START"][ki]),
                                                int(crop_start[ki]),
                                                )
                                   for ki, k in enumerate(img_slice)], axis=0)
            
            # 为可视化
            front_120_30 = ['img_front_120', 'img_front_30']
            # 顺序和定义一致
            curr_bs_i_tensor_cat_120_30 = torch.concat([torch.from_numpy(img_slice[k]).to(self.dyn_od_stream_feature_bank) 
                                                        for k in img_slice if k in front_120_30], dim=0).permute(0, 3, 1, 2) # BC HW
            curr_bs_i_tensor_cat_100 = torch.concat([torch.from_numpy(img_slice[k]).to(self.dyn_od_stream_feature_bank) 
                                                     for k in img_slice if k not in front_120_30], dim=0).permute(0, 3, 1, 2) # BC HW
            curr_bs_i_tensor_120_30 = F.grid_sample(curr_bs_i_tensor_cat_120_30, 
                                                    torch.from_numpy(images_grid[:2]).float().to(curr_bs_i_tensor_cat_120_30.device), 
                                                    align_corners=True, padding_mode='border', mode="nearest") / 255.0
            curr_bs_i_tensor_100 = F.grid_sample(curr_bs_i_tensor_cat_100, 
                                                 torch.from_numpy(images_grid[2:]).float().to(curr_bs_i_tensor_cat_100.device), 
                                                 align_corners=True, padding_mode='border', mode="nearest") / 255.0
            curr_bs_i_tensor = torch.concat([curr_bs_i_tensor_120_30, curr_bs_i_tensor_100], dim=0)
            
            for draw_img_i, img_name in enumerate(x_draw.keys()):
                x_draw[img_name].append(curr_bs_i_tensor[[draw_img_i]])
                
            """
             H, W, div, Z, Y, X,
             (40, 96, 8, 4, 48, 120)
            """
            ego2imgs = calib["ego2imgs"][i]
            xyz_camAX = self.model[self._transformers["DRIVING_BEV_DYN"]].xyz_camA.clone()
            vt_grid, _ = GetProjectGridByEgo2Imgs(
                ego2imgs, 
                H=40, 
                W=96, 
                div=8, 
                Z=4, 
                Y=48, 
                X=120, 
                sample_pts_3d=xyz_camAX.to(ego2imgs.device).clone())
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
            
            bank_feats = self.dyn_od_stream_feature_bank[i].unsqueeze(0).float().detach().cpu().numpy()
            prev_feats_grid = feats_shifted_grid[i].unsqueeze(0).float().detach().cpu().numpy()
            
            prev_feats = bank_feats * seq_flags[i].cpu().numpy()
            
            inputs_dict.update({
                "images_grid": images_grid.astype(np.float32),
                "vt_grid": vt_grid.float().detach().cpu().numpy(),
                "prev_feats": prev_feats,
                "prev_feats_grid": prev_feats_grid,  # 部署外挂计算
                }
            )

            if self.global_config.calib_data_save_path != "None":
                if self.calib_data_cnt % 10 == 0:
                    for k in inputs_dict:
                        single_calib_data_save_path = f'{self.global_config.calib_data_save_path}/{k}/{self.calib_data_cnt}.npy'
                        os.makedirs(os.path.dirname(single_calib_data_save_path), exist_ok=True)
                        np.save(single_calib_data_save_path, inputs_dict[k])
                self.calib_data_cnt+=1

            self.session = HBRuntime(self.global_config.onnx_path)
            outputs = self.session.run(self.output_names, inputs_dict)
            for k, o in zip(batch_ret, outputs):
                batch_ret[k].append(o)
        
        # 可视化, 同时验证image grid 是否正确
        for k in x_draw:
            x_draw[k] = torch.concat(x_draw[k], dim=0)
        for key_name in x:
            x[key_name] = x_draw[key_name]

        for k in batch_ret:
            batch_ret[k] = torch.from_numpy(np.concatenate(batch_ret[k], axis = 0)).cuda()
        # breakpoint()
        self.dyn_od_stream_feature_bank = batch_ret['prev_feats_output'].clone()
        self.dyn_od_stream_metas_bank = copy.deepcopy(metadata)

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
                'all_centerline_directions_preds',
                'all_keypoint_classes_preds',
                'all_keypoint_regs_preds',
                'all_polygon_classes_preds',
                'all_arrow_classes_preds',
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
            cut_start_h = self.global_config.Tasks['DRIVING_BEV_STA']['datasets']['validation'][0]['cut_start_h']
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
                'navi_info': calib['navi_info']['points'][i:i+1].detach().cpu().numpy(),
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
            batch_ret[k] = torch.from_numpy(np.concatenate(batch_ret[k], axis=0)).cuda()
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
