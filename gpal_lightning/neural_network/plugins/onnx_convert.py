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
                        
        for task in tasks:
            if task.name == "DRIVING_BEV_DYN":
                merged_input_dict["calib"]={}
                # merged_input_dict["calib"]["intrinsic"]= torch.rand(1, 7, 3, 3).cuda()
                merged_input_dict["calib"]["ego2imgs"] = torch.rand(
                    1, 7, 4, 4).cuda()


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
                        "size", "heading", "velocity", "score", "score_cls"]

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
        
        onnx_sim_path = onnx_path.replace(".onnx", "_sim.onnx")
        model = onnx.load(onnx_path)
        # convert model
        model_sim, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_sim, onnx_sim_path)
