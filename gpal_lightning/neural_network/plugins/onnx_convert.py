import os
from typing import List
import torch
import logging
import onnx
import numpy as np
from onnxsim import simplify
import torch.nn.functional as F

from gpal_lightning import const
from gpal_lightning.neural_network.network_modules.gpnet import GpNet
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.utils.get_checkpoint_path import get_checkpoint_path

from onnx import helper
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.postprocess.bev_points import Bev_To_Points
from gpal_nn.tasks.parking_ipm_sta.datasets.parking_ipm_sta_dataset import preprocess_img


class WrappedGpNet(GpNet):
    """
    This wrapped function is for onnx generation
    """
    def __init__(self,
                 global_config: GlobalConfig,
                 tasks: List,
                 ):
        super().__init__(global_config, tasks)
        self.is_onnx = True
    

    def forward(self, input):
        if input["task"] == "DRIVING_BEV_DYN":
            return self.forward_dyn(input) 
        elif input["task"] == "PARKING_IPM_STA":
            return self.forward_park(input)
        elif input["task"] == "DRIVING_BEV_STA":
            return self.forward_sta(input)
        
    def forward_dyn(self, input):
        outputs = []
        x = input["image"]
        calib = input["calib"]
        task = input["task"]
        images_grid = calib["images_grid"]
        metadata = input["metadata"]

        cam8m_set = ["img_front_120", "img_front_30"]
        cam2m_set = ["img_back", "img_front_left", "img_front_right", "img_rear_left", "img_rear_right"]

        img8ms = torch.cat([x[k] for k in cam8m_set], dim=0).permute(0, 3, 1, 2)
        img2ms = torch.cat([x[k] for k in cam2m_set], dim=0).permute(0, 3, 1, 2)

        udist_img8ms = F.grid_sample(
            img8ms, images_grid[:len(cam8m_set)].float(), align_corners=True, padding_mode='border', mode="nearest")
        udist_img2ms = F.grid_sample(
            img2ms, images_grid[len(cam8m_set):].float(), align_corners=True, padding_mode='border', mode="nearest")

        imgs = torch.cat([udist_img8ms, udist_img2ms], dim = 0) / 255.0
        mono3d, mono_od, gate_lever, side_od, neck2_in = [], [], [], [], []

        for backbone_name, camera_list in self.backbone_camera_mapping.items():
            bb_output = self.model[backbone_name](imgs)
            g0_output = self.model['group0'](bb_output)
            neck0_output = self.model['neck0'](g0_output)

        bev_feature = self.model[self._transformers[task]](neck0_output, calib)
      
        for task_name in self.tasks_to_run.keys():
            if task_name == "DRIVING_BEV_DYN":
                output = self.model[task_name](bev_feature,metadata = metadata )
                output = output[0]
                # print("DRIVING_BEV_DYN", output[0].shape)
                # BEV_TO_POINTS = dict(
                #             NAME="Bev_To_Points",
                #             NUM_BEV_FEATURES=64,
                #             VOXEL_SIZE=[0.64, 0.64],
                #             SCORE_THRESH=0.28,
                #             DOWN_RATIO=2,
                #             NUM_KEYPOINTS=256,
                #             TRAIN=True,
                #             NUM_OUTPUT_FEATURES=64
                #         )
                # bev_2_points = Bev_To_Points(model_cfg=BEV_TO_POINTS,
                #                                   grid_size=[480, 192,  12],
                #                                   voxel_size=[0.32, 0.32, 0.5],
                #                                   point_cloud_range=[-51.2, -
                #                                                      30.72, -1., 102.4, 30.72, 5.],
                #                                   num_bev_features=[64, 64, 128, 64, 128, 128, 128])
                # output = bev_2_points(output[0])
                print(ShowDataStruct("DRIVING_BEV_DYN", output))

                # exit(1)
        
        return output

    def forward_sta(self, input):
        outputs = []
        x = input["image"]
        calib = input["calib"]
        task = input["task"]
        # images_grid = calib["images_grid"]
        cam8m_set = ["img_front_30", "img_front_120"]
        image_30 = x["img_front_30"].permute(0, 3, 1, 2).contiguous()
        image_120 = x["img_front_120"].permute(0, 3, 1, 2).contiguous()
        udist_img8ms = torch.cat([image_30, image_120], dim=0)  # 2,3,2160,3840
        # udist_img8ms = F.grid_sample(images, images_grid.float(), align_corners=True, padding_mode='border', mode="nearest")
        imgs = udist_img8ms / 255.0

        for backbone_name, camera_list in self.backbone_camera_mapping.items():
            bb_output = self.model[backbone_name](imgs)
            g0_output = self.model['group0'](bb_output)
            neck0_output = self.model['neck0'](g0_output)
        
        neck0_output = {'img_front_30': [neck0_output[0][0:1, ...]], 'img_front_120': [neck0_output[0][1:2, ...]]}
        self.model[self._transformers[task]].is_compile = True
        print(self._transformers)
        bev_feature = self.model[self._transformers[task]](neck0_output, calib)
      
        for task_name in self.tasks_to_run.keys():
            outputs = self.model[task_name](bev_feature, calib)[0]
            print(ShowDataStruct("DRIVING_BEV_STA", outputs))
            outputs_classes, intermediate_reference_points = outputs['all_cls_scores'].detach(), outputs['all_pts_preds'].detach()
            lane_marking_types_preds, lane_marking_colors_preds = outputs['all_lane_marking_types_preds'].detach(), outputs['all_lane_marking_colors_preds'].detach()
            shape_types, centerline_types = outputs['all_shape_types_preds'].detach(), outputs['all_centerline_types_preds'].detach()
            centerline_directions, keypoint_classes, keypoint_regs = outputs['all_centerline_directions_preds'].detach(), outputs['all_keypoint_classes_preds'].detach(), outputs['all_keypoint_regs_preds'].detach()
            polygon_classes, arrow_classes = outputs['all_polygon_classes_preds'].detach(), outputs['all_arrow_classes_preds'].detach()

        return (outputs_classes[-1], intermediate_reference_points[-1],
                lane_marking_types_preds[-1], lane_marking_colors_preds[-1],
                shape_types[-1], centerline_types[-1], centerline_directions[-1], 
                keypoint_classes[-1], keypoint_regs[-1], polygon_classes[-1], arrow_classes[-1])
        

    def forward_park(self, input):
        img = input["image"]
        fisheye_rear = img['fisheye_img_rear']
        fisheye_front = img['fisheye_img_front']
        fisheye_left = img['fisheye_img_left']
        fisheye_right = img['fisheye_img_right']
       
        grid_rear_and_front = input['grid_rear_and_front']
        grid_left_and_right = input['grid_left_and_right']

        mask_rear = input['mask_rear']
        mask_front = input['mask_front']
        mask_left = input['mask_left']
        mask_right = input['mask_right']
        
        mask_rear = mask_rear.squeeze(0)
        mask_front = mask_front.squeeze(0)
        mask_left = mask_left.squeeze(0)
        mask_right = mask_right.squeeze(0)

        fisheye_rear_and_front = torch.cat((fisheye_rear,fisheye_front), dim=0)
        fisheye_left_and_right = torch.cat((fisheye_left,fisheye_right), dim=0)

        fisheye_rear_and_front = fisheye_rear_and_front.permute(0,3,1,2)
        fisheye_left_and_right = fisheye_left_and_right.permute(0,3,1,2)
      
      
        avm_rear_and_front = F.grid_sample(fisheye_rear_and_front, grid_rear_and_front, align_corners=True,padding_mode='zeros') 
        avm_rear_and_front = avm_rear_and_front.sum(0).permute(1,2,0)

        avm_left_and_right = F.grid_sample(fisheye_left_and_right, grid_left_and_right, align_corners=True,padding_mode='zeros') 
        avm_left_and_right = avm_left_and_right.sum(0).permute(1,2,0)

        avm = avm_left_and_right * mask_left / 255.0 + avm_rear_and_front * mask_rear / 255.0 + avm_rear_and_front * mask_front / 255.0 + avm_left_and_right * mask_right / 255.0 
        avm_input = preprocess_img(avm)
        avm_input = avm_input[None]
        # out = self.model(avm_input)

        bb_output = self.model['backbone0'](avm_input)
        g0_output = self.model['group0'](bb_output)
        neck0_output = self.model['neck0'](g0_output)
        bev_feature = neck0_output
            
        for task_name in self.tasks_to_run.keys():
            if task_name == "PARKING_IPM_STA":
                output = self.model[task_name](bev_feature)
        # exit(1)
        return [avm, output]


class PytorchToOnnx:
    @staticmethod
    def TaskImageShapeDict(task_name):
        if task_name in ["DRIVING_BEV_DYN"]:
            # used_image_shapes: dict = {
            #     "img_front_120": [768, 320, 3],
            #     "img_front_30": [768, 320, 3],
            #     "img_back": [768, 320, 3],
            #     "img_front_left": [768, 320, 3],
            #     "img_front_right": [768, 320, 3],
            #     "img_rear_left": [768, 320, 3],
            #     "img_rear_right": [768, 320, 3],
            # }
            used_image_shapes: dict = {
                "img_front_120": [3840, 2160, 3],
                "img_front_30": [3840, 2160, 3],
                "img_back": [1920, 1080, 3],
                "img_front_left": [1920, 1080, 3],
                "img_front_right": [1920, 1080, 3],
                "img_rear_left": [1920, 1080, 3],
                "img_rear_right": [1920, 1080, 3],
            }
            return used_image_shapes
        elif task_name in ["DRIVING_BEV_STA"]:
            used_image_shapes: dict = {
                "img_front_30": [960, 512, 3],
                "img_front_120": [960, 512, 3],
            }

            return used_image_shapes
        elif task_name in ["PARKING_IPM_STA"]:
            used_image_shapes: dict = {
                "fisheye_img_rear": [1920, 1536, 3],
                "fisheye_img_front": [1920, 1536, 3],
                "fisheye_img_left": [1920, 1536, 3],
                "fisheye_img_right": [1920, 1536, 3],
            }
            return used_image_shapes

    @staticmethod
    def prepare_dummy_input(config, tasks):
        input_dict = {}
           
        for task in tasks:
            used_image_shapes = PytorchToOnnx.TaskImageShapeDict(task.name)
            for cam in used_image_shapes:
                if cam not in input_dict.keys():
                    vector_shape = 1, used_image_shapes[cam][1], used_image_shapes[cam][0], 3
                    input_dict[cam] = torch.rand(*vector_shape).cuda()
            print(ShowDataStruct("input_dict", input_dict))

            if task.name == "DRIVING_BEV_DYN":
                merged_input_dict = {"task": task.name, "image": input_dict}
                merged_input_dict["calib"]={}
                merged_input_dict["calib"]["images_grid"] = torch.rand(
                    7, 320, 768, 2).cuda()
                merged_input_dict["calib"]["vt_grid"] = torch.rand(
                    7, 192, 120, 2).cuda()
                merged_input_dict["metadata"] = {}
                merged_input_dict["metadata"]["prev_feats"] = torch.rand(1, 128, 48, 120).cuda()
                merged_input_dict["metadata"]["prev_feats_grid"] = torch.rand(1, 48, 120, 2).cuda()

            elif task.name == "DRIVING_BEV_STA":
                merged_input_dict = {"task": tasks[0].name, "image": input_dict}
                merged_input_dict["calib"]={}
                # merged_input_dict["calib"]["images_grid"] = torch.rand(2, 512, 960, 2).cuda()
                merged_input_dict["calib"]["reference_points_rebatch"] = torch.rand(2, 5000, 4, 2).cuda()
                merged_input_dict["calib"]["queries_rebatch_grid"] = torch.rand(2, 50, 100,2).cuda()
                merged_input_dict["calib"]["restore_bev_grid"] = torch.rand(1, 100, 100, 2).cuda()
                merged_input_dict["calib"]["bev_pillar_counts"] = torch.rand(1, 5000, 1).cuda()
                merged_input_dict['calib']['navi_info'] = {"points": torch.rand(1, 20, 2).cuda()}

            elif task.name == "PARKING_IPM_STA":
                avm_w = 768
                avm_h = 768
                merged_input_dict = {"task": task.name, "image": input_dict}
                merged_input_dict['grid_rear_and_front'] = torch.rand(2, avm_w, avm_h, 2).cuda()
                merged_input_dict['grid_left_and_right'] = torch.rand(2, avm_w, avm_h, 2).cuda()
                # print(grid_rear_and_front.shape)

                merged_input_dict['mask_rear'] = torch.rand(1, avm_w, avm_h, 1).cuda()
                merged_input_dict['mask_front'] = torch.rand(1, avm_w, avm_h, 1).cuda()
                merged_input_dict['mask_left'] = torch.rand(1, avm_w, avm_h, 1).cuda()
                merged_input_dict['mask_right'] = torch.rand(1, avm_w, avm_h, 1).cuda()
                ######################################################################3
                
        return merged_input_dict

    @staticmethod
    def init_net(global_config, tasks):
        net = WrappedGpNet(global_config, tasks)
        if global_config.load_from:
            # checkpoint_path = get_checkpoint_path(global_config.load_from)
            checkpoint_path = global_config.load_from
            logging.info("Loading from {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            try:
                net.load_state_dict(checkpoint["state_dict"], strict=False)
            except Exception as e:
                print(e)
            # PytorchToOnnx.reset_weight(net)
            logging.info("Model was loaded successfully")
        return net

    @staticmethod
    def to_onnx(config, tasks, save_path):
        
        # transformer_type = config.Transformer["type"]
        # print(f"transformer_type = {transformer_type}")
        
        input_dummy = PytorchToOnnx.prepare_dummy_input(config, tasks)
        print(ShowDataStruct("input_dummy", input_dummy))

        
        net = PytorchToOnnx.init_net(config, tasks).cuda()
        
        onnx_path = os.path.join(save_path, const.CHECKPOINT_PATH, \
                                 config.load_from.split('/')[-1].replace(const.FILE_EXTENSION, const.ONNX_EXTENSION))

        
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        

        for task in tasks:
            do_constant_folding = True
            if task.name == "DRIVING_BEV_DYN":
                input_names = ["img_front_120", "img_front_30", "img_back", "img_front_left",
                       "img_front_right", "img_rear_left", "img_rear_right", "images_grid", "vt_grid",  "prev_feats","prev_feats_grid"]  # occ_od

                output_names = ["head_conv", "hm_center","prev_feats_output"]
            if task.name == "PARKING_IPM_STA":
                input_names=["img_rear", "img_front","img_left", "img_right",  "grid_rear_and_front", "grid_left_and_right","mask_rear", "mask_front", "mask_left", "mask_right"]
                output_names=['avm', 'slot_point', 'slot_line']

            if task.name == "DRIVING_BEV_STA":
                input_names = ["img_30", "img_120", "reference_points_rebatch", "queries_rebatch_grid", "restore_bev_grid", "bev_pillar_counts", "navi_info"]
                output_names = ["cls_scores", "pts_preds", 'lane_marking_types_preds', 'lane_marking_colors_preds', "shape_types_preds", "centerline_types_preds", "centerline_directions_preds", 
                                "keypoint_classes_preds", "keypoint_regs_preds", "polygon_classes_preds", "arrow_classes_preds"]
                do_constant_folding = False
            with torch.no_grad():
                torch.onnx.export(
                    net,
                    (input_dummy, {}),
                    onnx_path,
                    verbose=False,
                    opset_version=16,
                    # keep_initializers_as_inputs=False,
                    input_names=input_names,
                    output_names=output_names,
                    do_constant_folding=do_constant_folding
                )
            
    
        onnx_sim_path = onnx_path.replace(".onnx", "_sim.onnx")
        model = onnx.load(onnx_path)
        # convert model
        model_sim, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_sim, onnx_sim_path)

        print(f"onnx_path = {onnx_path}")
        print(f"onnx_sim_path = {onnx_sim_path}")
