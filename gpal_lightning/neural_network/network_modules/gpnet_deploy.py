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
from gpal_lightning.utils.deploy_utils import bgr_to_nv12_split, DistGridMap
from gpal_nn.tasks.driving_bev_dyn.postprocess.bev_points import Bev_To_Points

from horizon_tc_ui import HBRuntime

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
        self.image_crop_config = global_config.Tasks['DRIVING_BEV_DYN']['image_crop_config']


    def forward_one_DRIVING_BEV_DYN(self, x, calib, metadata):
        batch_ret = {
            'head_conv': [],
            'hm_cen': [],
               
        }
        save_path = f'calib'
        batch_size = len(metadata)
        for i in tqdm(range(batch_size)):

            img_slice = {k: x[k][i].unsqueeze(0).float().detach().cpu().numpy() for k in x}
            # print(ShowDataStruct("img_slice", img_slice))
            intrins = calib["intrinsic"][i].detach().cpu().numpy()
            cam_dists = calib["cam_dist"][i].detach().cpu().numpy()
            img_crop_dict = self.image_crop_config
            images_grid = np.stack([DistGridMap(img_slice[k].shape[2],
                                                img_slice[k].shape[1],
                                                cam_dists[ki],
                                                intrins[ki],
                                                int(img_crop_dict["IMAGE_RESIZE"][1]),
                                                int(img_crop_dict["IMAGE_RESIZE"][0]),
                                                int(img_crop_dict["IMAGE_CROP_H_LEN"]),
                                                int(img_crop_dict["CROP_HeSai_ID4"]
                                                    ["CROP_START"][ki])
                                                )
                                   for ki, k in enumerate(img_slice)], axis=0)


            ego2imgs = calib["ego2imgs"][i]
            xyz_camAX = self.model[self._transformers["DRIVING_BEV_DYN"]].xyz_camA.clone()
            vt_grid, _ = GetProjectGridByEgo2Imgs(
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
            inputs_dict.update({"images_grid": images_grid.astype(np.float32),
                                "vt_grid": vt_grid.float().detach().cpu().numpy()})

            if self.global_config.calib_data_save_path != "None":
                for k in inputs_dict:
                    single_calib_data_save_path = f'{self.global_config.calib_data_save_path}/{k}/{self.calib_data_cnt}.npy'
                    os.makedirs(os.path.dirname(single_calib_data_save_path), exist_ok=True)
                    np.save(single_calib_data_save_path, inputs_dict[k])
                self.calib_data_cnt+=1

            self.session = HBRuntime(self.global_config.onnx_path)
            outputs = self.session.run(self.output_names, inputs_dict)
            for k, o in zip(batch_ret, outputs):
                batch_ret[k].append(o)
        for k in batch_ret:
            batch_ret[k] = torch.from_numpy(np.concatenate(
                batch_ret[k], axis = 0)).cuda()
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
