import os
import numpy as np
import torch
import random
import json
import copy
import cv2

from gpal_lightning import const
from gpal_lightning.neural_network.tasks.base.task import BaseTask
from gpal_lightning.neural_network.tasks.builder import TASKS
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_lightning.utils.json_helpers.dict_to_json import dict_to_json

def GetBoxTf(x,y,z,yaw):
    tf = np.array([[np.cos(yaw), -np.sin(yaw), 0, x],
                   [np.sin(yaw), np.cos(yaw), 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return tf


def DistPoint(d, nx, ny):
    x2 = nx * nx
    y2 = ny * ny
    xy = nx * ny
    r2 = x2 + y2

    nx_new = nx * (1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 *
                   r2 * r2) + 2 * d[2] * xy + d[3] * (r2 + 2 * x2)
    ny_new = ny * (1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 *
                   r2 * r2) + 2 * d[3] * xy + d[2] * (r2 + 2 * y2)

    return nx_new, ny_new


def LineWithTruncated(p1, p2, f, cx, cy, d):
    if (p1[2] < 0) and (p2[2] < 0):
        return None, None

    if (p1[2] < 0):
        p1 = p1 + (p2 - p1) / (p2[2] - p1[2]) * (0.1-p1[2])
    if (p2[2] < 0):
        p2 = p2 + (p1 - p2) / (p1[2] - p2[2]) * (0.1-p2[2])

    np1 = p1 / p1[2]
    np2 = p2 / p2[2]
    np1[0], np1[1] = DistPoint(d, np1[0], np1[1])
    np2[0], np2[1] = DistPoint(d, np2[0], np2[1])

    p1u = int(np.clip((np1[0] * f + cx), -1e6, 1e6))
    p1v = int(np.clip((np1[1] * f + cy), -1e6, 1e6))
    p2u = int(np.clip((np2[0] * f + cx), -1e6, 1e6))
    p2v = int(np.clip((np2[1] * f + cy), -1e6, 1e6))

    return (p1u, p1v), (p2u, p2v)


def DrawBbox2D(img, pts, f, cx, cy, dist, color):
    for pair in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        p1 = pts[pair[0]]
        p2 = pts[pair[1]]
        uv1, uv2 = LineWithTruncated(p1, p2, f, cx, cy, dist)
        if (uv1 is not None) and (uv2 is not None):
            cv2.line(img, uv1, uv2, color, 4)
    return pts

def GetBboxInWorld(tf, size):
    hsx = size[0] * 0.5
    hsy = size[1] * 0.5
    hsz = size[2] * 0.5

    p = np.matrix([[-hsx, -hsy, -hsz, 1],
                   [-hsx, hsy, -hsz, 1],
                   [hsx, hsy, -hsz, 1],
                   [hsx, -hsy, -hsz, 1],

                   [-hsx, -hsy, hsz, 1],
                   [-hsx, hsy, hsz, 1],
                   [hsx, hsy, hsz, 1],
                   [hsx, -hsy, hsz, 1],
                   ])
    p_w = np.array(np.matrix(tf) * p.T)
    p_w = p_w[:3]
    return p_w


def Draw3DObjectsOnImage(img, objects, intrin, extrin, dist, color):
    img = copy.deepcopy(img)
    f = intrin[0, 0]
    cx = intrin[0, 2]
    cy = intrin[1, 2]
    tf_2_cam = extrin
    for box in objects:
        obj_tf = GetBoxTf(box[0], box[1], box[2], box[6])
        pts = GetBboxInWorld(obj_tf, box[3:6])
        obj_cam = np.matmul(tf_2_cam, np.concatenate(
            [pts, np.ones_like(pts[:1, :])], axis=0))[:3]
        if np.sum(obj_cam[2, :] > 0) > 0:
            DrawBbox2D(img, obj_cam.T, f, cx, cy, dist, color)
    return img

@TASKS.register_module()
class DRIVING_BEV_DYNTask(BaseTask):
    def __init__(self, global_config, task_config, name):
        super().__init__(global_config, task_config, name, None)
        pass

    def GetVis(self, imgs, preds, gts, metadata, calib, idx):
        # print(ShowDataStruct("imgs", imgs))
        # print(ShowDataStruct("calib", calib))
        from tools_scripts.vis_2d import Vis2D
        vis1 = Vis2D([-30, 100], [-30, 30], 0.1)
        try:
            pred_objs = self.vector_to_json(preds, metadata, False)
            for box in gts[idx]['gt_boxes']:
                #  [x, y, z, dx, dy, dz, heading]
                vis1.DrawBbox(GetBoxTf(box[0], box[1], box[2], box[6]), [box[3], box[4]], None, [0, 255, 0],
                            [255, 255, 255], line_width=1)
            for box, score in zip(pred_objs[idx]['boxes_lidar'], pred_objs[idx]['score']):
                vis1.DrawBbox(GetBoxTf(box[0], box[1], box[2], box[6]), [box[3], box[4]], None, [0, 0, 255],
                            [255, 255, 255], line_width=1)


            vis_imgs = []
            for cam_idx, cam_name in enumerate(metadata[idx]["camera_name"]):
                calib_intrin = copy.deepcopy(calib["intrinsic"][idx][cam_idx]).detach().cpu().numpy()
                calib_extrin = copy.deepcopy(calib["extrinsic"][idx][cam_idx]).detach().cpu().numpy()
                calib_dist = copy.deepcopy(calib["cam_dist"][idx][cam_idx]).detach().cpu().numpy()
                img_crop_dict = copy.deepcopy(calib["img_crop_dict"])
                img = (imgs[cam_name][idx].detach().cpu().numpy().transpose(
                    1, 2, 0) * 254).astype(np.uint8)
                calib_intrin[:2, :] /= float(img_crop_dict['CROP_HeSai_ID4']['SCALE'][cam_idx][idx])
                calib_intrin[1, 2] -= float(img_crop_dict['CROP_HeSai_ID4']['CROP_START'][cam_idx][idx])
                img = cv2.undistort(img, calib_intrin, calib_dist, calib_intrin)


                calib_dist *=0.0

                gt_boxes = gts[idx]['gt_boxes']
                img = Draw3DObjectsOnImage(
                    img, gt_boxes, calib_intrin, calib_extrin, calib_dist[0], [0, 255, 0])

                gt_boxes = pred_objs[idx]['boxes_lidar']
                img = Draw3DObjectsOnImage(
                    img, gt_boxes, calib_intrin, calib_extrin, calib_dist[0], [0, 0, 255])

                vis_imgs.append(img)
            vis_imgs = np.concatenate(vis_imgs, axis=0)

        except:
            print("DRIVING_BEV_DYNTask GetVis faild")
            pass
        
        vis_draw1 = vis1.Draw()
        
        try:
            vis_imgs = cv2.resize(vis_imgs, (int(vis_imgs.shape[1] / vis_imgs.shape[0] * vis_draw1.shape[0]), vis_draw1.shape[0]))
            vis_draw1 = np.concatenate([vis_draw1, vis_imgs], axis=1)

        except:
            print("DRIVING_BEV_DYNTask GetVis faild")
            pass

        
        return vis_draw1

    def heavy_log(self, iteration, phase, log_writer, data, preds, masks, trues, metadata, calib, loss_info=None):
        imgs = []
        for idx in range(4):
            vis = self.GetVis(data, preds, trues, metadata, calib, idx)
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
