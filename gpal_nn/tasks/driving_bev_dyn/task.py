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
            cv2.line(img, uv1, uv2, color, 2)
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

def draw_uvboxes_on_image(image, projected_boxes, color=(0, 255, 0), thickness=2):
    img = image.copy()
    edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 竖边
        ]
    
    for vertices_2d, valid_mask in projected_boxes:
        # 绘制边
        for edge in edges:
            if valid_mask[edge[0]] and valid_mask[edge[1]]:
                pt1 = tuple(vertices_2d[edge[0]].astype(int))
                pt2 = tuple(vertices_2d[edge[1]].astype(int))
                cv2.line(img, pt1, pt2, color, thickness)
        
        # 绘制顶点
        for i, vertex in enumerate(vertices_2d):
            if valid_mask[i]:
                pt = tuple(vertex.astype(int))
                if i in [2,3,6,7]:
                    cv2.circle(img, pt, 1, (254, 0, 0), -1)
                # else:
                #     cv2.circle(img, pt, 1, (0, 0, 255), -1)
                    
    return img


def project_points_to_fisheye_raw_uv(points_world, intrinsic_matrix, distortion_coeffs, extrinsic_matrix):
    """
    将世界坐标系下的3D点投影到鱼眼相机图像平面
    
    参数:
    - points_world: 世界坐标系下的3D点 (Nx3) numpy数组
    - intrinsic_matrix: 相机内参矩阵 (3x3)
                       [[fx,  0, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]]
    - distortion_coeffs: 鱼眼畸变系数 (1xN) 或 (Nx1)
                        通常为 [k1, k2, k3, k4, k5] (5参数模型)
    - extrinsic_matrix: 相机外参矩阵 (4x4) 
                       [[R | t],
                        [0 | 1]]
    
    返回:
    - points_image: 图像坐标 (Nx2) numpy数组 [[u1, v1], [u2, v2], ...]
    - valid_mask: 布尔数组 (N,)，标记哪些点在相机前方且投影有效
    """
    
    # 步骤1: 世界坐标系 -> 相机坐标系
    # 转换为齐次坐标
    N = points_world.shape[0]
    points_homo = np.hstack([points_world, np.ones((N, 1))])
    
    # 应用外参变换
    points_camera_homo = (extrinsic_matrix @ points_homo.T).T
    points_camera = points_camera_homo[:, :3]
    
    # 过滤相机后方的点
    valid_mask = points_camera[:, 2] > 0
    
    # if not np.any(valid_mask):
    #     return np.zeros((N, 2)), valid_mask
    
    # 步骤2: 相机坐标系 -> 归一化平面
    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]
    
    # 避免除零
    Z = np.where(Z > 1e-6, Z, 1e-6)
    
    x = X / Z
    y = Y / Z
    
    # 步骤3: 计算入射角 theta
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    
    # 步骤4: 鱼眼畸变模型 (等距投影 + 畸变)
    # r_distorted = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8 + k5*theta^10)
    
    # 提取畸变系数
    dist = distortion_coeffs.flatten()
    k1 = dist[0] if len(dist) > 0 else 0
    k2 = dist[1] if len(dist) > 1 else 0
    k3 = dist[2] if len(dist) > 2 else 0
    k4 = dist[3] if len(dist) > 3 else 0
    k5 = dist[4] if len(dist) > 4 else 0
    
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta10 = theta8 * theta2
    
    # 畸变后的径向距离 (5项式)
    r_distorted = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8 + k5*theta10)
    
    # 避免除零
    r = np.where(r > 1e-6, r, 1e-6)
    
    # 计算畸变后的归一化坐标
    scale = r_distorted / r
    x_distorted = x * scale
    y_distorted = y * scale
    
    # 步骤5: 归一化平面 -> 像素坐标
    # 提取内参
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    # 组合结果
    points_image = np.column_stack([u, v])
    
    # 标记无效点
    # points_image[~valid_mask] = 0
    
    return points_image, points_camera


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

def Draw3DObjectsOnFisheyeImageOneView(img, objects, intrin, extrin, dist, color, scale, crop_start, cam_name):
    img = copy.deepcopy(img)
    
    # f = intrin[0, 0] / scale
    # cx = intrin[0, 2]
    # cy = intrin[1, 2] / scale - crop_start

    projected_boxes = []
    
    for box in objects:
        obj_tf = GetBoxTf(box[0], box[1], box[2], box[6])
        pts = GetBboxInWorld(obj_tf, box[3:6])
        
        obj_cam, points_camera = project_points_to_fisheye_raw_uv(pts.T, intrin, dist, extrin)
        obj_cam[:, 0] = obj_cam[:, 0] / scale
        obj_cam[:, 1] = obj_cam[:, 1] / scale - crop_start
        
        valid_mask = points_camera[:, 2] > 0
            
        projected_boxes.append((obj_cam, valid_mask))
        
        img = draw_uvboxes_on_image(img, projected_boxes, color, thickness=1)
    cv2.putText(img, cam_name, (30, 30), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[255, 255, 255], thickness=2, fontScale=1)
    return img

@TASKS.register_module()
class DRIVING_BEV_DYNTask(BaseTask):
    def __init__(self, global_config, task_config, name):
        super().__init__(global_config, task_config, name, None)
        self.image_crop_config = global_config.Tasks['DRIVING_BEV_DYN']['image_crop_config']
        
        self.subtask_name = global_config.Tasks['DRIVING_BEV_DYN']['SWITCH_SUBTASK']
        self.deploy_cfg = global_config.Tasks['DRIVING_BEV_DYN'].get('DEPLOY_CFG', None)

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
                vis1.DrawBbox(GetBoxTf(box[0], box[1], box[2], box[6]), [box[3], box[4]], [box[8], box[9]], [0, 255, 0],
                            [255, 255, 255], line_width=1)
            for box, score in zip(pred_objs[idx]['boxes_lidar'], pred_objs[idx]['score']):
                vis1.DrawBbox(GetBoxTf(box[0], box[1], box[2], box[6]), [box[3], box[4]], [box[7], box[8]], [0, 0, 255],
                            [255, 255, 255], line_width=1)
            if imgs.get('points', None) is not None:
                point = imgs['points'][idx].cpu().numpy()
                color = [255, 255, 255]
                vis1.DrawPointcloud(point, color)

            if self.subtask_name in ['DRIVING_BEV_DYN_FISHEYE']:
                vis_imgs = []
                for cam_idx, cam_name in enumerate(metadata[idx]["camera_name"]):
                    calib_intrin = copy.deepcopy(calib["intrinsic"][idx][cam_idx]).detach().cpu().numpy()
                    calib_extrin = copy.deepcopy(calib["extrinsic"][idx][cam_idx]).detach().cpu().numpy()
                    calib_dist = copy.deepcopy(calib["cam_dist"][idx][cam_idx]).detach().cpu().numpy()
                    img_crop_dict = copy.deepcopy(self.image_crop_config)
                    img = (imgs[cam_name][idx].detach().cpu().numpy().transpose(1, 2, 0) * 254).astype(np.uint8)
                    img = np.ascontiguousarray(img)
                    
                    scale = img_crop_dict['CROP_HeSai_ID4']['SCALE'][cam_idx]
                    crop_start = img_crop_dict['CROP_HeSai_ID4']['CROP_START'][cam_idx]
                    
                    gt_boxes = gts[idx]['gt_boxes']
                    img = Draw3DObjectsOnFisheyeImageOneView(
                        img, gt_boxes, calib_intrin, calib_extrin, calib_dist[0], [0, 255, 0], scale, crop_start, cam_name)
                    
                    gt_boxes = pred_objs[idx]['boxes_lidar']
                    img = Draw3DObjectsOnFisheyeImageOneView(
                        img, gt_boxes, calib_intrin, calib_extrin, calib_dist[0], [0, 0, 255], scale, crop_start, cam_name)

                    vis_imgs.append(img)
                vis_imgs = np.concatenate(vis_imgs, axis=0)
            
            elif self.subtask_name in ['DRIVING_BEV_DYN']:
                vis_imgs = []
                # breakpoint()
                for cam_idx, cam_name in enumerate(metadata[idx]["camera_name"]):
                    calib_intrin = copy.deepcopy(calib["intrinsic"][idx][cam_idx]).detach().cpu().numpy()
                    calib_extrin = copy.deepcopy(calib["extrinsic"][idx][cam_idx]).detach().cpu().numpy()
                    calib_dist = copy.deepcopy(calib["cam_dist"][idx][cam_idx]).detach().cpu().numpy()
                    
                    scale = copy.deepcopy(metadata[idx]["scale"][cam_idx])
                    crop_start = copy.deepcopy(metadata[idx]["crop"][cam_idx])
                    
                    # img_crop_dict = copy.deepcopy(self.image_crop_config)
                    img = (imgs[cam_name][idx].detach().cpu().numpy().transpose(
                        1, 2, 0) * 254).astype(np.uint8)
                    # calib_intrin[:2, :] /= float(img_crop_dict['CROP_HeSai_ID4']['SCALE'][cam_idx])
                    # calib_intrin[1, 2] -= float(img_crop_dict['CROP_HeSai_ID4']['CROP_START'][cam_idx])
                    
                    calib_intrin[:2, :] /= scale
                    calib_intrin[1, 2] -= crop_start
                    
                    calib_dist *= 0.0
                    img = cv2.undistort(img, calib_intrin, calib_dist, calib_intrin)

                    gt_boxes = gts[idx]['gt_boxes']
                    img = Draw3DObjectsOnImage(
                        img, gt_boxes, calib_intrin, calib_extrin, calib_dist[0], [0, 255, 0])

                    gt_boxes = pred_objs[idx]['boxes_lidar']
                    img = Draw3DObjectsOnImage(
                        img, gt_boxes, calib_intrin, calib_extrin, calib_dist[0], [0, 0, 255])

                    vis_imgs.append(img)
                vis_imgs = np.concatenate(vis_imgs, axis=0)
            
            else:
                raise NotImplementedError(f"DRIVING_BEV_DYNTask GetVis faild {self.subtask_name}")
            
        except ValueError as e:
            print(f"DRIVING_BEV_DYNTask GetVis faild {e}")
            pass
        
        vis_draw1 = vis1.Draw()
        
        try:
            vis_imgs = cv2.resize(vis_imgs, (int(vis_imgs.shape[1] / vis_imgs.shape[0] * vis_draw1.shape[0]), vis_draw1.shape[0]))
            vis_draw1 = np.concatenate([vis_draw1, vis_imgs], axis=1)

            cv2.putText(vis_draw1, metadata[idx]['frame_num'], (50, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[0, 0, 255], thickness=2, fontScale=0.5)
            cv2.putText(vis_draw1, metadata[idx]['clip_id'], (50, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[0, 0, 255], thickness=2, fontScale=0.5)
            # cv2.putText(vis_draw1, '/'.join(metadata[idx]['img_path']['img_front_120'].split('/')[-4:]), (50, 90),
            # fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[0, 0, 255], thickness=2, fontScale=0.5)
            cv2.putText(vis_draw1, "{} {:.2f}m/s {:.4f}rad/s".format(metadata[idx]['timestamp'],metadata[idx]['ego_speed'],metadata[idx]['ego_yaw_rate']) , (50, 90),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[0, 0, 255], thickness=2, fontScale=0.5)



        except ValueError as e:
            print(f"DRIVING_BEV_DYNTask GetVis faild {e}")
            pass

        
        return vis_draw1

    def heavy_log(self, iteration, phase, log_writer, data, preds, masks, trues, metadata, calib, loss_info=None):
        imgs = []
        bs = min(self.global_config.image_per_gpu, self.head.global_config.Train['max_visu_img_num'])
        for idx in range(bs):
            vis = self.GetVis(data, preds, trues, metadata, calib, idx)
            imgs.append(vis)
            # import cv2
            # cv2.imwrite(f"eval_vis_online/{metadata[idx]['frame_id'].split('/')[0]}.jpg", vis)

        imgs = np.concatenate(imgs, axis=1)

        # import cv2
        # cv2.imwrite(f"eval_vis_single2/{iteration}.jpg", imgs)

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
                                          const.EVALUATION_FILES_EXTENSION), file_object, indent=4)
                # print(os.path.join(file_root, uuid +
                #                    const.EVALUATION_FILES_EXTENSION))
            else:
                raise ValueError(
                    f"Unrecognized EVALUATION_FILES_EXTENSION: {const.EVALUATION_FILES_EXTENSION}")
        return json_list, metadata_list

    def eval_visualize(self, save_root, metadata, data, dataloader_idx, preds, trues, batch, json_list, calib):
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        
        bs = len(metadata)
        for idx in range(bs):
            vis_draw1 = self.GetVis(data, preds, trues, metadata, calib, idx)
            # import cv2
            # cv2.imwrite(f"eval_vis_online/{metadata[idx]['frame_id'].split('/')[0]}.jpg", vis)
            image_filename = metadata[idx]['clip_id'] + '^' + metadata[idx]['timestamp']
            cv2.imwrite(os.path.join(save_root, f"{image_filename}.jpg"), vis_draw1, [int(cv2.IMWRITE_JPEG_QUALITY), 100])