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
from tools_scripts.point_projection import point_projection_lane
from tools_scripts.data_format_cvt import ShowDataStruct



@TASKS.register_module()
class DRIVING_BEV_STATask(BaseTask):
    def __init__(self, global_config, task_config, name):
        super().__init__(global_config, task_config, name, None)
        pass

    def GetVis(self, preds, gts, idx):

        from tools_scripts.vis_2d import Vis2D
        linetype_list = ['solid', 'dashed']
        vis1 = Vis2D([-30, 130], [-20, 20], 0.1)
        try:
            for l in gts[idx]['edges']['points']:
                vis1.DrawPolyline(l, [0, 255, 255], 2, 'solid')
            for i,l in enumerate(gts[idx]['polylines']['points']):
                shape_type = gts[idx]['polylines']['shape_type'][i]
                vis1.DrawPolyline(l, [0, 255, 0], 2, linetype_list[shape_type])
        except:
            pass
        vis_draw1 = vis1.Draw()
        pre_pts = preds['all_pts_preds']
        # print(ShowDataStruct("preds", preds))
        color_list=[(0, 0, 255), (255, 0, 0)]
        pre_pts_denorm = torch.stack(
            [(1-pre_pts[..., 1]) * 120, ((1-pre_pts[..., 0])-0.5) * 32], dim=-1)

        vis2 = Vis2D([-30, 130], [-20, 20], 0.1)
        for l, ln, s, shape_type in zip(pre_pts_denorm[-1, idx], pre_pts[-1, idx], preds['all_cls_scores'][-1, idx], preds['all_shape_types_preds'][-1, idx]):
            # if s[1:].sigmoid().max() > 0.3:
            cls_score_pred = s.squeeze().sigmoid()
            value, cls_pred = cls_score_pred.max(-1)
            if value > 0.3:
                # print(f"ln \n{s.sigmoid().max()}")
                # color = [random.randint(0, 255), random.randint(
                #     0, 255), random.randint(0, 255)]
                _, shape_type = shape_type.max(-1)
                vis2.DrawPolyline(l.detach().cpu().numpy(), color_list[cls_pred], 2, linetype_list[shape_type])
        # exit()
        vis_draw2 = vis2.Draw()

        return np.concatenate([vis_draw1, vis_draw2], axis=1)

    def heavy_log(self, iteration, phase, log_writer, data, preds, masks, trues, metadata, loss_info=None):
        imgs = []
        for idx in range(4):
            vis = self.GetVis(preds[0], trues, idx)
            imgs.append(vis)

        imgs = np.concatenate(imgs, axis=1)
        self.logger.image_log(iteration, phase, log_writer,
                              0, torch.from_numpy(imgs).permute(2, 0, 1).flip(0))

    def vectors_to_json(self, metadata, data, dataloader_idx, vectors, is_gt):
        # print(f"vectors_to_json {dataloader_idx} {is_gt}")
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
    
    def draw_point_uv(self, images, data, pts_uv, color=(0, 0, 255)):
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
            images = images * 255
            images = images.astype(np.uint8)
        elif isinstance(images, list):
            images = np.array(images)
        if isinstance(pts_uv, torch.Tensor):
            pts_uv = pts_uv.cpu().numpy()

        if images.ndim == 5:  # b,n,3,h,w
            images = images[0]
            images = images.transpose(0, 2, 3, 1)[:, :, :, ::-1]
        if pts_uv.ndim == 4:  # b,n,N,2
            pts_uv = pts_uv[0]
        assert images.shape[0] == pts_uv.shape[0]

        images_list = []
        pta_uv_list = []
        ori_shape_list = []
        # cut_h = data["meta"][0]["cut_h_value"] if "cut_h_value" in data["meta"][0] else 0
        # cut_w = data["meta"][0]["cut_w_value"] if "cut_w_value" in data["meta"][0] else 0
        ori_shape = data["ori_shape"]
        n = images.shape[0]
        for i in range(n):
            images_list.append(images[i])
            pta_uv_list.append(pts_uv[i])
            ori_shape_list.append(ori_shape[i])
        imgs = []
        for img, pts, shape in zip(images_list, pta_uv_list, ori_shape_list):
            img = np.ascontiguousarray(img)
            pts = pts.astype(np.int32)
            # pts[..., 0] = pts[..., 0] * (img.shape[1] + cut_w) / shape[1] - cut_w
            # pts[..., 1] = pts[..., 1] * (img.shape[0] + cut_h) / shape[0] - cut_h
            # pts[..., 0] = pts[..., 0] - cut_w
            # pts[..., 1] = pts[..., 1] - cut_h

            mask_x_min = pts[..., 0] > 0
            mask_x_max = pts[..., 0] < img.shape[1]
            mask_y_min = pts[..., 1] > 0
            mask_y_max = pts[..., 1] < img.shape[0]
            mask_pixel = mask_y_min * mask_y_max * mask_x_min * mask_x_max
            pts = pts[mask_pixel]

            for point in pts:
                cv2.circle(img, (point[0], point[1]), int(3), color, -1)

            imgs.append(img)
        # images = np.vstack(imgs)[:, :, ::-1]
        return imgs
    def concat_imgvis_and_bevvis(self,images, paint):
        # assert isinstance(images, list)
        images = np.vstack(images)

        new_paint = np.ones((max(images.shape[0], paint.shape[0]), images.shape[1] + paint.shape[1], 3),
                            dtype=np.uint8) * 255
        new_paint[:images.shape[0], :images.shape[1], :] = images[:, :, :]
        new_paint[:paint.shape[0], images.shape[1]:(images.shape[1] + paint.shape[1]), :] = paint[:, :, :]

        cv2.line(new_paint, (images.shape[1], 0), (images.shape[1], new_paint.shape[0]), (125, 125, 125), 1)

        return new_paint

    def convert_to_tensor(self, data):
        """
        将输入的字典数据转换为PyTorch张量
        
        参数:
            data: 包含'vectors'键的字典，其中每个vector包含'pts'字段
                'pts'是二维坐标列表，格式为[[x1, y1], [x2, y2], ...]
        
        返回:
            torch.Tensor: 三维张量，形状为[N, M, 3]，其中:
                        - N是向量组数量
                        - M是每个向量组中的点数量
                        - 3表示每个点包含[x, y, 0.0]三个维度
        """
        tensor_data = []
        if 'vectors' in data:
            iterable = data['vectors']
        elif 'gt_vectors' in data:
            iterable = data['gt_vectors']
        else:
            iterable = []

        for vector in iterable:
            pts = vector['pts']
            # 每个点从[x,y]转换为[x,y,0.0]
            transformed_pts = [[p[0], p[1], 0.0] for p in pts]
            tensor_data.append(transformed_pts)
        
        # 转换为torch.Tensor并返回
        return torch.tensor(tensor_data)

    def GetImgVis(self, data, metadata, calib, bev_real2aug, preds, gts, idx):
        front_30 = data['img_front_30'][idx]  # 形状变为 [3, 320, 768]
        front_120 = data['img_front_120'][idx]  # 形状变为 [3, 320, 768]

        # 在新维度上堆叠这两个张量，形成 [2, 3, 320, 768]
        images = torch.stack([front_30, front_120], dim=0).unsqueeze(0)
        pred_pts_to_tensor = self.convert_to_tensor(preds[idx])
        if pred_pts_to_tensor.shape[-1] != 0:
            pred_pts_to_cam_uv = point_projection_lane(pred_pts_to_tensor, calib['exts'][idx:idx+1], calib['ists'][idx:idx+1], calib['dists'][idx:idx+1], bev_real2aug=bev_real2aug[idx:idx+1])
            images = self.draw_point_uv(images, metadata[idx], pred_pts_to_cam_uv, color=(0, 0, 255))
        gt_pts_to_tensor = self.convert_to_tensor(gts[idx])
        if gt_pts_to_tensor.shape[-1] != 0:
            gt_pts_to_cam_uv = point_projection_lane(gt_pts_to_tensor, calib['exts'][idx:idx+1], calib['ists'][idx:idx+1], calib['dists'][idx:idx+1], bev_real2aug=bev_real2aug[idx:idx+1])
            images = self.draw_point_uv(images, metadata[idx], gt_pts_to_cam_uv, color=(0, 255, 0))
            # 确保返回格式一致
        if isinstance(images, torch.Tensor):
            # 如果没有绘制任何点，转换为与draw_point_uv相同的输出格式
            images = images.cpu().numpy()
            images = images * 255
            images = images.astype(np.uint8)
            if images.ndim == 5:
                images = images[0].transpose(0, 2, 3, 1)[:, :, :, ::-1].tolist()
        return images
