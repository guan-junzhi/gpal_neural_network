import os
import numpy as np
import torch
import random
import json
import copy

from gpal_lightning import const
from gpal_lightning.neural_network.tasks.base.task import BaseTask
from gpal_lightning.neural_network.tasks.builder import TASKS
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_lightning.utils.json_helpers.dict_to_json import dict_to_json

def GetBoxTf(x,y,yaw):
    tf = np.array([[np.cos(yaw), -np.sin(yaw), 0, x],
                   [np.sin(yaw), np.cos(yaw), 0, y],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return tf



@TASKS.register_module()
class DRIVING_BEV_DYNTask(BaseTask):
    def __init__(self, global_config, task_config, name):
        super().__init__(global_config, task_config, name, None)
        pass

    def GetVis(self, preds, gts, metadata, idx):
        from tools_scripts.vis_2d import Vis2D
        vis1 = Vis2D([-30, 100], [-30, 30], 0.1)
        try:
            pred_objs = self.vector_to_json(preds, metadata, False)
            for box in gts[idx]['gt_boxes']:
                #  [x, y, z, dx, dy, dz, heading]
                vis1.DrawBbox(GetBoxTf(box[0], box[1], box[6]), [box[3], box[4]], None, [0, 255, 0],
                            [255, 255, 255], line_width=1)
            for box, score in zip(pred_objs[idx]['boxes_lidar'], pred_objs[idx]['score']):
                vis1.DrawBbox(GetBoxTf(box[0], box[1], box[6]), [box[3], box[4]], None, [0, 0, 255],
                            [255, 255, 255], line_width=1)
        except:
            print("DRIVING_BEV_DYNTask GetVis faild")
            pass
        vis_draw1 = vis1.Draw()
        return vis_draw1

    def heavy_log(self, iteration, phase, log_writer, data, preds, masks, trues, metadata, loss_info=None):
        imgs = []
        for idx in range(4):
            vis = self.GetVis(preds, trues, metadata, idx)
            imgs.append(vis)
        imgs = np.concatenate(imgs, axis=1)
        self.logger.image_log(iteration, phase, log_writer,
                              0, torch.from_numpy(imgs).permute(2, 0, 1).flip(0))

    def vectors_to_json(self, metadata, data, dataloader_idx, vectors, is_gt):
        trues_or_preds = const.TRUES if is_gt else const.PREDS
        json_list, metadata_list = [], []
        meta = copy.deepcopy(metadata)

        if "clip_id" in meta:
            clip_id = meta["clip_id"]
        else:
            clip_id = ""
        uuid = clip_id + "_" + meta[0]["frame_num"]
        metadata_list.append(meta)
        
        json_dict = self.vector_to_json(vectors, meta, is_gt)
        json_list.append(json_dict)

        if not self.global_config.dump_json_during_validation or not self.global_config.dump_path:
            inference_root = os.path.join(
                const.JOB_EVALUATION_PATH, self.name, str(dataloader_idx))
        else:
            inference_root = os.path.join(
                self.global_config.dump_path, const.CURRENT_TIME, self.name, str(dataloader_idx))

        for file_type, file_object in zip((trues_or_preds, const.METADATA), (json_dict, meta)):
            file_root = os.path.join(inference_root, file_type, clip_id)
            if not os.path.exists(file_root):
                os.makedirs(file_root, exist_ok=True)
            if const.EVALUATION_FILES_EXTENSION.lower() == ".json":
                dict_to_json(os.path.join(file_root, uuid +
                                          const.EVALUATION_FILES_EXTENSION), file_object)
                # print(os.path.join(file_root, uuid +
                #                    const.EVALUATION_FILES_EXTENSION))
            else:
                raise ValueError(
                    f"Unrecognized EVALUATION_FILES_EXTENSION: {const.EVALUATION_FILES_EXTENSION}")
        return json_list, metadata_list
