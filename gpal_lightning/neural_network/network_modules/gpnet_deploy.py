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
from gpal_nn.models.base_modules.pointpreprocess import CenterPointPreProcess


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
                self.deploy_cfg = self.global_config.Tasks['DRIVING_BEV_DYN'].get('DEPLOY_CFG', None)
                if self.global_config.Backbones.get("backbone1") is None :
                    continue
                if self.global_config.Backbones["backbone1"]["point_process_config"] is None:
                    continue
                preprocess_cfg = self.global_config.Backbones["backbone1"]["point_process_config"]
                max_voxels_num = tuple(preprocess_cfg['max_voxels_num'])
                self.centerpoint_preprocess = CenterPointPreProcess(
                    pc_range=preprocess_cfg["pc_range"],
                    voxel_size=preprocess_cfg["voxel_size"],
                    max_voxels_num=max_voxels_num,
                    max_points_in_voxel=preprocess_cfg["max_points_in_voxel"],
                    norm_range=preprocess_cfg["norm_range"],
                    norm_dims=preprocess_cfg["norm_dims"],
                )
                self.feature_map_shape = self.get_feature_map_size(
                    preprocess_cfg['pc_range'],
                    preprocess_cfg['voxel_size'],
                )

    def get_feature_map_size(self, point_cloud_range, voxel_size):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        return grid_size

    def gen_xyz_camA(self):
        transformer_config = self.global_config.Transformer["transformer_config"]
        
        self.point_cloud_range = transformer_config["bev_map_range"]
        self.voxel_size = transformer_config["bev_map_voxel_size"]
        
        self.grid_size = [int(round((self.point_cloud_range[3]-self.point_cloud_range[0])/(self.voxel_size[0]),2)),
                          int(round((self.point_cloud_range[4]-self.point_cloud_range[1])/(self.voxel_size[1]),2))]
        
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
            # x_draw = copy.deepcopy({k: x[k][i].unsqueeze(0) for k in x})  # need torch
            for _, img_name in enumerate(x_draw.keys()):
                x_draw[img_name].append(x[img_name][i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0)
            
            
            extrinsic_matrix = calib['extrinsic'][i]
            distortion_coeffs= calib['cam_dist'][i]
            intrinsic_matrix = calib['intrinsic'][i]
            H, W, div, Z, Y, X = 64, 120, 8, 4, 48, 60
            xyz_camAX = self.model[self._transformers["DRIVING_BEV_DYN"]].xyz_camA.clone()
            
            vt_grid, _ = GetProjectGridByEgo2ImgsFisheye(
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
        if 'points' in x.keys():
            points = x['points']
            points = [points[j].cuda() for j in range(len(points))]
            features,coors = self.centerpoint_preprocess(points,True)
            # features = features.permute(0, 3, 2, 1)
            batch_size = 1
            nx = self.feature_map_shape[0]
            ny = self.feature_map_shape[1]
            nchannels = features.shape[1]
            max_points_in_voxel = features.shape[2]
            dense_features = features.new_zeros(
                (
                    batch_size, 
                    nchannels, 
                    max_points_in_voxel, 
                    nx * ny
                )
            ) 

            for batch_id in range(batch_size):
                batch_mask = coors[:, 0] == batch_id
                this_coords = coors[batch_mask, :]
                indices = this_coords[:, 2] * nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                cur_features = features[: , : , : , batch_mask]
                dense_features[batch_id, :, :, indices] = cur_features

            features = dense_features
            features = features.permute(0, 3, 2, 1)

            x.pop('points')
        else:
            features = None
            coors = None
            points = None
        
        if self.subtask_name in ['DRIVING_BEV_DYN_FISHEYE']:
            data_dict = self.forward_one_DRIVING_BEV_DYN_fisheye(x, calib, metadata)
            return data_dict
        
        batch_ret = {
            # 'fish_head_conv': [],
            # 'fish_hm_center': [],
            # 'fish_prev_feats_output': [],
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
        
        inputs_dict = {}
        
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
            
            ego2imgs = calib["ego2imgs"][i]
            xyz_camAX = self.model[self._transformers["DRIVING_BEV_DYN"]].xyz_camA.clone()
            
            if self.deploy_cfg is not None:
                if self.deploy_cfg['mode'] == "gpal30_in_model_with_small_image":
                    # x_draw = copy.deepcopy({k: x[k][i].unsqueeze(0) for k in x})  # need torch
                    for _, img_name in enumerate(x_draw.keys()):
                        x_draw[img_name].append(x[img_name][i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0)
                    vt_grid, _ = GetProjectGridByEgo2Imgs(
                        ego2imgs, 
                        H=52, 
                        W=96, 
                        div=8, 
                        Z=6, 
                        Y=48, 
                        X=120, 
                        sample_pts_3d=xyz_camAX.to(ego2imgs.device).clone())
            else:
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
                vt_grid, _ = GetProjectGridByEgo2Imgs(
                    ego2imgs, 
                    H=40, 
                    W=96, 
                    div=8, 
                    Z=4, 
                    Y=48, 
                    X=120, 
                    sample_pts_3d=xyz_camAX.to(ego2imgs.device).clone())
                
                inputs_dict.update({
                    "images_grid": images_grid.astype(np.float32),
                })
            
            vt_grid = torch.clip(vt_grid, -1.1, 1.1)


            # inputs_dict = {}
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
                "vt_grid": vt_grid.float().detach().cpu().numpy(),
                "prev_feats": prev_feats,
                "prev_feats_grid": prev_feats_grid,  # 部署外挂计算
                # "img_front_fisheye": np.zeros((1, 512, 960, 3), dtype=np.float32),  
                # "img_right_fisheye": np.zeros((1, 512, 960, 3), dtype=np.float32),  
                # "img_rear_fisheye": np.zeros((1, 512, 960, 3), dtype=np.float32),  
                # "img_left_fisheye": np.zeros((1, 512, 960, 3), dtype=np.float32),  
                # "fish_vt_grid": np.zeros((4, 192, 60, 2), dtype=np.float32),  
                # "fish_prev_feats": np.zeros((1, 128, 48, 60), dtype=np.float32),  
                # "fish_prev_feats_grid": np.zeros((1, 48, 60, 2), dtype=np.float32),  
                # "img_front_fisheye_y": np.zeros((1, 512, 960, 1), dtype=np.uint8),  
                # "img_front_fisheye_uv": np.zeros((1, 256, 480, 2), dtype=np.uint8),  
                # "img_right_fisheye_y": np.zeros((1, 512, 960, 1), dtype=np.uint8),  
                # "img_right_fisheye_uv": np.zeros((1, 256, 480, 2), dtype=np.uint8),  
                # "img_rear_fisheye_y": np.zeros((1, 512, 960, 1), dtype=np.uint8),  
                # "img_rear_fisheye_uv": np.zeros((1, 256, 480, 2), dtype=np.uint8),  
                # "img_left_fisheye_y": np.zeros((1, 512, 960, 1), dtype=np.uint8),  
                # "img_left_fisheye_uv": np.zeros((1, 256, 480, 2), dtype=np.uint8),  
                # "fish_vt_grid": np.zeros((4, 192, 60, 2), dtype=np.float32),  
                # "fish_prev_feats": np.zeros((1, 128, 48, 60), dtype=np.float32),  
                # "fish_prev_feats_grid": np.zeros((1, 48, 60, 2), dtype=np.float32),  

                }
            )
            if features is not  None:
                inputs_dict.update({
                    "features":features.detach().cpu().numpy(),
                    # "coors": coors.view(1, 1,coors.shape[0], coors.shape[1]).detach().cpu().numpy().astype(np.int32),
                    })
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
        x['points'] = points
        for k in batch_ret:
            batch_ret[k] = torch.from_numpy(np.concatenate(batch_ret[k], axis = 0)).cuda()
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
                # 'images_grid': images_grid.astype(np.float32),
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
