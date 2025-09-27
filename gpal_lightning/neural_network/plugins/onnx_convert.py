import os
from typing import List
import torch
import logging
import onnx
import numpy as np
from onnxsim import simplify
from gpal_lightning import const
from gpal_lightning.neural_network.network_modules.gpnet import GpNet
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.utils.get_checkpoint_path import get_checkpoint_path
# from gpal_lightning.tasks.bev_od.heads.sparse4d_plugin.sparse4d_head_config import get_sparse4d_head_cfg

from onnx import helper
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.postprocess.bev_points import Bev_To_Points


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
        outputs = []
        x = input["image"]
        calib = input["calib"]
        task = input["task"]


        mono3d, mono_od, gate_lever, side_od, neck2_in = [], [], [], [], []

        for backbone_name, camera_list in self.backbone_camera_mapping.items():
        
            # # backbone inference
            # self.model[backbone_name].feature_id = [2]

            print(x.shape)
            bb_output = self.model[backbone_name](x)
            print(backbone_name, bb_output[0].shape)

            g0_output = self.model['group0'](bb_output)
            print("group0", backbone_name, g0_output[0].shape)

            neck0_output = self.model['neck0'](g0_output)
            print("neck0", neck0_output[0].shape)
                     
        print(self._transformers)
        bev_feature = self.model[self._transformers[task]](
            neck0_output,  calib)

        print("transformers", bev_feature.shape)
        
      
        for task_name in self.tasks_to_run.keys():
            if task_name == "DRIVING_BEV_DYN":
                output = self.model[task_name](bev_feature)
                # print("DRIVING_BEV_DYN", output[0].shape)
                BEV_TO_POINTS = dict(
                            NAME="Bev_To_Points",
                            NUM_BEV_FEATURES=64,
                            VOXEL_SIZE=[0.64, 0.64],
                            SCORE_THRESH=0.28,
                            DOWN_RATIO=2,
                            NUM_KEYPOINTS=256,
                            TRAIN=True,
                            NUM_OUTPUT_FEATURES=64
                        )
                bev_2_points = Bev_To_Points(model_cfg=BEV_TO_POINTS,
                                                  grid_size=[480, 192,  12],
                                                  voxel_size=[0.32, 0.32, 0.5],
                                                  point_cloud_range=[-51.2, -
                                                                     30.72, -1., 102.4, 30.72, 5.],
                                                  num_bev_features=[64, 64, 128, 64, 128, 128, 128])
                output = bev_2_points(output[0])
                print(ShowDataStruct("DRIVING_BEV_DYN", output))

                # exit(1)
        
        return output


# batch<class 'dict'>
#     meta<class 'list'> len = 2
#         0<class 'dict'>
#             frame_id<class 'str'>:1752111924.974980/1752111924.874973
#             camera_name<class 'list'> len = 7
#                 0<class 'str'>:img_front_120
#                 1<class 'str'>:img_front_left
#                 2<class 'str'>:img_rear_left
#                 3<class 'str'>:img_front_right
#                 4<class 'str'>:img_rear_right
#                 5<class 'str'>:img_back
#                 6<class 'str'>:img_front_30
#             task_name<class 'str'>:DRIVING_BEV_DYN
#             img_path<class 'dict'>
#                 img_front_120<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_120/1752111924.974980.jpg
#                 img_front_left<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_left/1752111924.974980.jpg
#                 img_rear_left<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_rear_left/1752111924.974980.jpg
#                 img_front_right<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_right/1752111924.974980.jpg
#                 img_rear_right<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_rear_right/1752111924.974980.jpg
#                 img_back<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_back/1752111924.974980.jpg
#                 img_front_30<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_30/1752111924.974980.jpg
#             clip_id<class 'str'>:EKART_ID4001_2025-07-10-09-30-00_2025-07-10_09-45-14-102
#             frame_num<class 'str'>:0_20
#         1<class 'dict'>
#             frame_id<class 'str'>:1752111925.475056/1752111925.375045
#             camera_name<class 'list'> len = 7
#                 0<class 'str'>:img_front_120
#                 1<class 'str'>:img_front_left
#                 2<class 'str'>:img_rear_left
#                 3<class 'str'>:img_front_right
#                 4<class 'str'>:img_rear_right
#                 5<class 'str'>:img_back
#                 6<class 'str'>:img_front_30
#             task_name<class 'str'>:DRIVING_BEV_DYN
#             img_path<class 'dict'>
#                 img_front_120<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_120/1752111925.475056.jpg
#                 img_front_left<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_left/1752111925.475056.jpg
#                 img_rear_left<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_rear_left/1752111925.475056.jpg
#                 img_front_right<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_right/1752111925.475056.jpg
#                 img_rear_right<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_rear_right/1752111925.475056.jpg
#                 img_back<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_back/1752111925.475056.jpg
#                 img_front_30<class 'str'>:/data/ai_group/workdirs/od_occ_group/huiquyang/codes/DetZero/data/id4/raw_data/obstacle_data/EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-45-14-102/img_front_30/1752111925.475056.jpg
#             clip_id<class 'str'>:EKART_ID4001_2025-07-10-09-30-00_2025-07-10_09-45-14-102
#             frame_num<class 'str'>:0_21
#     image<class 'dict'>
#         img_front_120<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#         img_front_left<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#         img_rear_left<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#         img_front_right<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#         img_rear_right<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#         img_back<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#         img_front_30<class 'torch.Tensor'> : torch.Size([2, 3, 320, 768]) torch.float32
#     label<class 'list'> len = 2
#         0<class 'dict'>
#             gt_curr_vel<class 'numpy.ndarray'> : (256, 2) float32
#             gt_curr_hm_cen<class 'numpy.ndarray'> : (6, 96, 240) float32
#             gt_curr_cen_offset<class 'numpy.ndarray'> : (256, 2) float32
#             gt_curr_direction<class 'numpy.ndarray'> : (256, 2) float32
#             gt_curr_multibin_direction<class 'numpy.ndarray'> : (256, 6) float32
#             gt_curr_z_coor<class 'numpy.ndarray'> : (256, 1) float32
#             gt_curr_dim<class 'numpy.ndarray'> : (256, 3) float32
#             gt_curr_indices_center<class 'numpy.ndarray'> : (256,) int64
#             gt_curr_obj_mask<class 'numpy.ndarray'> : (256,) uint8
#             gt_boxes<class 'numpy.ndarray'> : (13, 11) float64
#         1<class 'dict'>
#             gt_curr_vel<class 'numpy.ndarray'> : (256, 2) float32
#             gt_curr_hm_cen<class 'numpy.ndarray'> : (6, 96, 240) float32
#             gt_curr_cen_offset<class 'numpy.ndarray'> : (256, 2) float32
#             gt_curr_direction<class 'numpy.ndarray'> : (256, 2) float32
#             gt_curr_multibin_direction<class 'numpy.ndarray'> : (256, 6) float32
#             gt_curr_z_coor<class 'numpy.ndarray'> : (256, 1) float32
#             gt_curr_dim<class 'numpy.ndarray'> : (256, 3) float32
#             gt_curr_indices_center<class 'numpy.ndarray'> : (256,) int64
#             gt_curr_obj_mask<class 'numpy.ndarray'> : (256,) uint8
#             gt_boxes<class 'numpy.ndarray'> : (13, 11) float64
#     calib<class 'dict'>
#         intrinsic<class 'torch.Tensor'> : torch.Size([2, 7, 3, 3]) torch.float32
#         cam_dist<class 'torch.Tensor'> : torch.Size([2, 7, 1, 5]) torch.float32
#         extrinsic<class 'torch.Tensor'> : torch.Size([2, 7, 4, 4]) torch.float64
#         img_crop_dict<class 'dict'>
#             IMAGE_RESIZE<class 'list'> len = 2
#                 0<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                 1<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#             IMAGE_CROP_H_LEN<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#             IMAGE_CROP_SIZE<class 'list'> len = 2
#                 0<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                 1<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#             CROP_HeSai_ID4<class 'dict'>
#                 SCALE<class 'list'> len = 7
#                     0<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     1<class 'torch.Tensor'> : torch.Size([2]) torch.float64
#                     2<class 'torch.Tensor'> : torch.Size([2]) torch.float64
#                     3<class 'torch.Tensor'> : torch.Size([2]) torch.float64
#                     4<class 'torch.Tensor'> : torch.Size([2]) torch.float64
#                     5<class 'torch.Tensor'> : torch.Size([2]) torch.float64
#                     6<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                 CROP_START<class 'list'> len = 7
#                     0<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     1<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     2<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     3<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     4<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     5<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#                     6<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#         img_shapes<class 'torch.Tensor'> : torch.Size([2, 7, 3]) torch.int64
#         bev_real2aug<class 'torch.Tensor'> : torch.Size([2, 4, 4]) torch.float32
#         ego2imgs<class 'torch.Tensor'> : torch.Size([2, 7, 4, 4]) torch.float64
#     fast_buf_try_cnt<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#     fast_buf_sec_cnt<class 'torch.Tensor'> : torch.Size([2]) torch.int64
#     dataloader_time<class 'torch.Tensor'> : torch.Size([2, 1]) torch.float64

class PytorchToOnnx:

    @staticmethod
    def TaskImageShapeDict(task_name):
        if task_name in ["DRIVING_BEV_DYN"]:
            used_image_shapes: dict = {
                "img_front_120": [768, 320, 3],
                "img_front_left": [768, 320, 3],
                "img_rear_left": [768, 320, 3],
                "img_front_right": [768, 320, 3],
                "img_rear_right": [768, 320, 3],
                "img_back": [768, 320, 3],
                "img_front_30": [768, 320, 3],
            }

            return used_image_shapes

    @staticmethod
    def prepare_dummy_input(config, tasks):
        print(tasks[0].name)
        input_dict = {}
        
        task = tasks[0]
        used_image_shapes = PytorchToOnnx.TaskImageShapeDict(task.name)
        for cam in used_image_shapes:
            if cam not in input_dict.keys():
                vector_shape = 1, 3, used_image_shapes[cam][1], used_image_shapes[cam][0]
                input_dict[cam] = torch.rand(*vector_shape)
        print(ShowDataStruct("input_dict", input_dict))
        merged_input_dict = {"task": tasks[0].name, "image": torch.concat(
            [input_dict[cam] for cam in input_dict]).cuda()}
                        


        # for backbone in config.Backbones.keys():
        #     input_source = config.Backbones[backbone]['input_source']
        #     # pts voxelize shape
        #     if "points" in input_source:
        #         pts_vector_shape = (1, 12, 760, 280)
        #         merged_input_dict["points_feature"] = torch.rand(*pts_vector_shape).cuda()
        #     elif "lidar_points" in input_source:
        #         lidar_feat_shape = (1, 256, 190, 70)
        #         merged_input_dict["lidar_feature"] = torch.rand(*lidar_feat_shape).cuda()
        #     elif "input_m1" in input_source:
        #         depth_image_shape = (3, 1, 512, 960)
        #         merged_input_dict["depth_image_m1"] = torch.rand(*depth_image_shape).cuda()
        #     else:
        #         merged_input_dict["x"] = torch.concat([input_dict[cam] for cam in input_source]).cuda()
        # for task in tasks:
        #     if task.name == "MOTIONNET":
        #         merged_input_dict['prev_feat_od'] = torch.zeros([2, 64, 64, 120]).cuda()
        #     if task.name == "OCC_OD":
        #         bev_h = int((task.task_config.transformer_x_range[1] - task.task_config.transformer_x_range[0]) / task.task_config.grid_resolution[0])
        #         bev_w = int((task.task_config.transformer_y_range[1] - task.task_config.transformer_y_range[0]) / task.task_config.grid_resolution[1])
        #         bev_z = int((task.task_config.ref_z_range[1] - task.task_config.ref_z_range[0]) / task.task_config.grid_resolution[2])
        #         if "calib" not in merged_input_dict:
        #             merged_input_dict["calib"] = {}
   
        #         if config.Transformer["type"] == "DeformableTransformer":
        #             reference_points = torch.randn((1, bev_h * bev_w, len(task.task_config.cam_used) * bev_z, 2)).cuda()
        #             valid_weight = torch.randn((1, bev_h * bev_w, 1)).cuda()
        #             valid_index = torch.randn((1, bev_h * bev_w, len(task.task_config.cam_used) * bev_z)).cuda()
        #             merged_input_dict["calib"]["reference_points"] = reference_points
        #             merged_input_dict["calib"]["valid_weight"] = valid_weight
        #             merged_input_dict["calib"]["valid_index"] = valid_index.int()
        #         elif config.Transformer["type"] == "ViewInverseProjectionEgoPEV2":
        #             merged_input_dict["calib"]["img_coors"] = torch.randint(0, 1, (bev_h * bev_w * bev_z * 7 // 2, 4)).cuda()
        #             merged_input_dict["calib"]["bev_coors"] = torch.randint(0, 1, (bev_h * bev_w * bev_z * 7 // 2, 2)).cuda()
                
        #         bev_grid = torch.randn((1, bev_h, bev_w, 3)).cuda()
        #         cur2prev = torch.randn((1, 4, 4)).cuda()
        #         prev_feat = torch.randn((1, task.task_config.Head["head1"]["layers_config"]["in_channels"], bev_h, bev_w)).cuda()
        #         merged_input_dict["calib"]["bev_grid"] = bev_grid
        #         merged_input_dict["calib"]["cur2prev"] = cur2prev
        #         merged_input_dict["calib"]["prev_feat"] = prev_feat

        #     if task.name == "BEV_OD" and  task.task_config.Head["head1"].get("type", None) == "Sparse4DHead":
        #         head_cfg = get_sparse4d_head_cfg(**task.task_config.Head["head1"])
        #         num_anchor = head_cfg["instance_bank"]["num_anchor"]
        #         num_cache_anchor = head_cfg["instance_bank"]["num_temp_instances"]
        #         embed_dims = head_cfg["instance_bank"]["embed_dims"]
        #         kps_generator = head_cfg["deformable_model"]["kps_generator"]
        #         num_levels = head_cfg["deformable_model"]["num_levels"]
        #         num_cams = head_cfg["deformable_model"]["num_cams"]
        #         num_pts = kps_generator["num_learnable_pts"] + len(kps_generator["fix_scale"])
        #         sparse4d_prev_exist = torch.ones((1), dtype=torch.bool, device=merged_input_dict["x"].device)
        #         sparse4d_prev2cur = torch.randn((1, 4, 4), device=merged_input_dict["x"].device)
        #         sparse4d_projection_mat = torch.randn((1, num_cams, 4, 4), device=merged_input_dict["x"].device)
        #         sparse4d_confidence_prev = torch.randn((1, num_cache_anchor), device=merged_input_dict["x"].device)
        #         sparse4d_anchor_prev = torch.randn((1, num_cache_anchor, len(head_cfg["reg_weights"])), device=merged_input_dict["x"].device)
        #         sparse4d_feat_prev = torch.randn((1, num_cache_anchor, embed_dims), device=merged_input_dict["x"].device)
        #         sparse4d_time_interval = torch.tensor([[0.1]], device=merged_input_dict["x"].device)

        #         if "calib" not in merged_input_dict:
        #             merged_input_dict["calib"] = {}
        #         merged_input_dict["calib"]["sparse4d_prev_exist"] = sparse4d_prev_exist.int()
        #         merged_input_dict["calib"]["sparse4d_prev2cur"] = sparse4d_prev2cur
        #         merged_input_dict["calib"]["projection_mat"] = sparse4d_projection_mat
        #         merged_input_dict["calib"]["sparse4d_time_interval"] = sparse4d_time_interval
        #         merged_input_dict["calib"]["sparse4d_confidence_prev"] = sparse4d_confidence_prev
        #         merged_input_dict["calib"]["sparse4d_anchor_prev"] = sparse4d_anchor_prev
        #         merged_input_dict["calib"]["sparse4d_feat_prev"] = sparse4d_feat_prev
        
        # exit(1)
        for task in tasks:
            if task.name == "DRIVING_BEV_DYN":
                merged_input_dict["calib"]={}
                # merged_input_dict["calib"]["intrinsic"]= torch.rand(1, 7, 3, 3).cuda()
                merged_input_dict["calib"]["ego2imgs"] = torch.rand(
                    1, 7, 4, 4).cuda()


        return merged_input_dict

    # @staticmethod
    # def reset_weight(net):
    #     from zpilot_nn.tasks.mono_od.heads.parrellel_head import LightComposedHead
    #     def reset(module):
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 reset(child)
    #         if isinstance(module, LightComposedHead):
    #             module.reset_weight()
    #     reset(net)

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
        transformer_type = config.Transformer["type"]

        print(f"transformer_type = {transformer_type}")
        
        input_dummy = PytorchToOnnx.prepare_dummy_input(config, tasks)
        print(ShowDataStruct("input_dummy", input_dummy))

        
        net = PytorchToOnnx.init_net(config, tasks).cuda()
        
        onnx_path = os.path.join(save_path, const.CHECKPOINT_PATH, \
                                 config.load_from.split('/')[-1].replace(const.FILE_EXTENSION, const.ONNX_EXTENSION))

        
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        input_names = ["image", "ego2imgs"]  # occ_od

        output_names = ["center", "z",
                                   "size", "heading", "velocity", "score"]

        with torch.no_grad():
            torch.onnx.export(
                net,
                (input_dummy, {}),
                onnx_path,
                verbose=True,
                opset_version=16,
                # keep_initializers_as_inputs=False,
                input_names=input_names,
                output_names=output_names
            )
        
        # exit(1)

        onnx_sim_path = onnx_path.replace(".onnx", "_sim.onnx")
        model = onnx.load(onnx_path)
        # convert model
        model_sim, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_sim, onnx_sim_path)
