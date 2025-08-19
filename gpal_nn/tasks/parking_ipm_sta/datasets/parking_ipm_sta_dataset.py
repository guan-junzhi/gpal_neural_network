
import copy
from multiprocessing import Pool
import random
import os
import cv2
import pickle
from typing import List, Union
from torch import distributed
import numpy as np

from gpal_lightning import const
from gpal_lightning.neural_network.tasks.builder import DATASETS
from gpal_lightning.neural_network.tasks.base.datasets.image_base_dataset import ImageBaseDataset
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_nn.tasks.driving_bev_sta.datasets.transform import *
from gpal_nn.tasks.driving_bev_sta.datasets.letter_box import letterbox_image, random_scale_and_translate
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *
from gpal_nn.tasks.driving_bev_sta.datasets.collect import _fix_pts_interpolate
from gpal_lightning.utils.profiling import TimeProf
import random
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf
import time
import multiprocessing
from shapely.geometry import LineString
import json
from gpal_nn.tasks.parking_ipm_sta.datasets.txtlabel_instance_p3 import TXTLabelLoader


class PLAssigner:
    def __init__(self, h, w, point_sigma, line_sigma, line_pad):
        super(PLAssigner, self).__init__()
        self.h = h
        self.w = w
        self.point_sigma = point_sigma
        self.line_sigma = line_sigma
        self.line_pad = line_pad

    def assign(self, anno):
        point_list = anno['points']
        line_list = anno['lines']

        gt_maps = np.zeros((2, self.h, self.w), dtype=np.float32)  # ch h w
        for ct in point_list:
            gt_maps[0] = self.draw_msra_gaussian(
                gt_maps[0], ct, self.point_sigma)

        for ln in line_list:
            gt_maps[1] = self.draw_guass_line(
                gt_maps[1], ln, self.line_sigma, self.line_pad)
        # print(gt_maps[0].shape)
        # point_save = gt_maps[0] * 255
        # line_save = gt_maps[1] * 255
        # cv2.imwrite("/tmpnfs/yaoming.zhang/landmark_pytorch/ldmk_data/j3_test/194_out/torch_test_31/gt_point_small.jpg", point_save)
        # cv2.imwrite("/tmpnfs/yaoming.zhang/landmark_pytorch/ldmk_data/j3_test/194_out/torch_test_31/gt_line_small.jpg", line_save)
        return gt_maps

    def draw_msra_gaussian(self, heatmap, center, sigma):
        tmp_size = sigma * 3
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[0], heatmap.shape[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
        img_x = max(0, ul[0]), min(br[0], h)
        img_y = max(0, ul[1]), min(br[1], w)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap

    def draw_guass_line(self, linemap, line, line_sigma, line_pad):
        line_pad = int(line_sigma * 3 + 0.5)
        st_x = int(line[0] + 0.5)
        st_y = int(line[1] + 0.5)
        ed_x = int(line[2] + 0.5)
        ed_y = int(line[3] + 0.5)
        vec = [float(ed_x - st_x), float(ed_y - st_y)]
        norm = vec[0] ** 2 + vec[1] ** 2
        norm = norm ** 0.5
        if (norm < 1.0):
            return linemap
        vec_norm = vec
        vec_norm[0] = vec[0] / norm
        vec_norm[1] = vec[1] / norm
        h, w = linemap.shape[0], linemap.shape[1]
        min_x = max(0, min(st_x, ed_x) - line_pad)
        max_x = min(w, max(st_x, ed_x) + line_pad)
        min_y = max(0, min(st_y, ed_y) - line_pad)
        max_y = min(h, max(st_y, ed_y) + line_pad)
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                vec_cur = [x - st_x, y - st_y]
                vert_dist = abs(
                    vec_cur[0] * vec_norm[1] - vec_cur[1] * vec_norm[0])
                para_proj = vec_cur[0] * vec_norm[0] + vec_cur[1] * vec_norm[1]
                if (para_proj >= 0 and vert_dist <= line_pad):
                    gauss = np.exp(-(vert_dist * vert_dist) /
                                   (2 * line_sigma * line_sigma))
                    linemap[y][x] = max(linemap[y][x], gauss)
        return linemap


def heatmap_to_point(heatmap: np.ndarray) -> Tuple[int, int]:
    """将热力图转换为关键点坐标（最大值点）"""
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (idx[1], idx[0])  # (x, y) 顺序


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    point_radius: int = 5,
    point_color: Tuple[int, int, int] = (0, 0, 255),
    show_text: bool = True
) -> np.ndarray:
    """
    将热力图叠加到原始图像上

    参数:
        image: 原始图像 (H, W, 3)，BGR格式
        heatmap: 热力图 (H, W) 或 (H, W, 1)
        alpha: 叠加透明度
        colormap: OpenCV颜色映射
        point_radius: 关键点半径
        point_color: 关键点颜色 (B, G, R)
        show_text: 是否显示关键点坐标

    返回:
        叠加后的图像 (H, W, 3)
    """
    # 确保热力图维度正确
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(2)

    # 调整热力图大小以匹配原始图像
    # heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # 将热力图归一化到0-255
    heatmap_normalized = cv2.normalize(
        heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)

    # 叠加热力图到原始图像
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    # 绘制关键点（热力图最大值点）
    x, y = heatmap_to_point(heatmap)
    cv2.circle(overlaid, (x, y), point_radius, point_color, -1)

    # 显示坐标文本
    if show_text:
        cv2.putText(overlaid, f"({x}, {y})", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 2)

    return overlaid


@DATASETS.register_module()
class PARKING_IPM_STADataset(ImageBaseDataset):
    def __init__(self,
                 global_config: GlobalConfig,
                 task_config,
                 phase: str,
                 preprocess,
                 root_dir,
                 data_list,
                 dataset_name: str,
                 pseudo_labels_path: Union[str, list] = None,
                 worker: int = 0,
                 shuffle: bool = True,
                 shuffle_seed: int = 0,
                 sql_filter: str = "",
                 ratio: float = 0.0,
                 camera_name=['img_front_120'],
                 w=768,
                 h=768,
                 point_sigma=3,
                 line_sigma=1.6,
                 line_pad=8,
                 fast_buffer_path=""
                 ):

        # import pickle as pkl
        # inputs = [global_config,
        #           task_config,
        #           phase,
        #           preprocess,
        #           root_dir,
        #           data_list,
        #           dataset_name,
        #           pseudo_labels_path,
        #           worker,
        #           shuffle,
        #           shuffle_seed,
        #           sql_filter,
        #           ratio,
        #           camera_name,
        #           w,
        #           h,
        #           point_sigma,
        #           line_sigma,
        #           line_pad,
        #           fast_buffer_path]
        # pkl.dump(inputs, open("inputs.pkl", 'wb'))
        # exit(1)

        DATASETS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT")
        LOCAL_DATASETS_ROOT = os.getenv(
            "ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT")

        root_dir = os.path.join(DATASETS_ROOT, root_dir)

        self.root_dir = root_dir
        self.data_list = os.path.join(self.root_dir, data_list)
        super().__init__(global_config=global_config,
                         task_config=task_config,
                         preprocess=preprocess,
                         dataset_name=dataset_name,
                         phase=phase,
                         camera_name=camera_name,
                         root_dir=root_dir,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed,
                         sql_filter=sql_filter,
                         ratio=ratio,
                         worker=worker,
                         pseudo_labels_path=pseudo_labels_path,
                         fast_buffer_path="" if fast_buffer_path == "" else os.path.join(
                             LOCAL_DATASETS_ROOT, fast_buffer_path, f"{task_config.name}_buf")
                         )

        self.w, self.h = w, h
        self.point_sigma = point_sigma
        self.line_sigma = line_sigma
        self.line_pad = line_pad
        self.assigner = PLAssigner(
            self.h, self.w, self.point_sigma, self.line_sigma, self.line_pad)

        self.transforms = None
        self.img_transforms = None
        self.task = task_config.name
        self.camera_names = camera_name

    def _build_world_data_list(self):
        try:
            rank_curr = distributed.get_rank()
            self.global_rank = rank_curr
            self.rank_local = distributed.get_rank() % 8
        except (RuntimeError, AssertionError):
            rank_curr = 0
            self.rank_local = 0

        with open(self.data_list, 'r') as f:
            self.world_data_list = f.readlines()
            self.world_data_list = [line.rstrip(
                '\n') for line in self.world_data_list]

            if self.phase == const.PHASE_TRAINING:
                self.world_data_list = [os.path.join(
                    self.root_dir, line) for line in self.world_data_list]

                self.world_data_list = [(line, line.replace(".json", ".jpg").replace(
                    "label", "avm")) for line in self.world_data_list]
            else:
                self.world_data_list = [[os.path.join(
                    self.root_dir, line.split(' ')[0]), os.path.join(
                    self.root_dir, line.split(' ')[1])] for line in self.world_data_list]
            # self.world_data_list = self.world_data_list[:10]

    def __len__(self):
        return len(self.dataset)

    def save_all_heatmap(self, savePath):
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        for idx, data in enumerate(self.dataset):
            anno_f, image_f = data
            if idx > 10:
                break
            image = self.pull_img(image_f)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            anno = self.pull_anno(anno_f)
            gtmap = self.assigner.assign(anno)

            # kmap = np.zeros((self.h, self.w, 1), np.uint8)
            # lmap = np.zeros((self.h, self.w, 1), np.uint8)
            # for j in range(self.h):
            #     for i in range(self.w):
            #         kmap[j][i] = int(gtmap[0][j][i] * 255)
            #         lmap[j][i] = int(gtmap[1][j][i] * 255)
            # print("kmap value ", kmap[j][i])

            res_img = overlay_heatmap(image, gtmap[0], point_radius=3)
            line_img = overlay_heatmap(image, gtmap[1], point_radius=3)
            cv2.imwrite(savePath + '/' + str(idx) + '_pt.jpg', res_img)
            cv2.imwrite(savePath + '/' + str(idx) + '_line.jpg', line_img)

    def pull_img(self, image_f):
        # print("image_f:", image_f)
        image = cv2.imread(image_f)
        if image is None:
            print("!!!!image is None = ", image_f)
        origin_shape = image.shape
        image = cv2.resize(image, (self.w, self.h))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, origin_shape

    def pull_anno(self, anno_f):
        json_path = anno_f
        assert os.path.exists(json_path), "%s isn't existed." % json_path
        anno = self.json_load(json_path)
        return anno

    def json_load(self, json_path):
        anno = {}
        annos = json.load(open(json_path))
        objs = annos["annotation"]["object"]
        raw_img_w = annos["annotation"]["imgsize"]["width"]
        raw_img_h = annos["annotation"]["imgsize"]["height"]
        anno['w'] = int(float(raw_img_w))
        anno['h'] = int(float(raw_img_h))
        coex = self.w * 1.0 / anno['w']
        coey = self.h * 1.0 / anno['h']
        slotpt_list = []
        slotline_list = []

        for obj in objs:
            if obj["name"] == "keypoint":
                slot_pt = obj["pt"][0]
                x = float(slot_pt['x']) * coex
                y = float(slot_pt['y']) * coey
                slotpt_list.append([x, y])
            if obj["name"] == "line":
                slot_line = obj["pt"]
                pt0 = slot_line[0]
                pt1 = slot_line[1]
                x0 = float(pt0['x']) * coex
                y0 = float(pt0['y']) * coey

                x1 = float(pt1['x']) * coex
                y1 = float(pt1['y']) * coey
                slotline_list.append([x0, y0, x1, y1])
        anno['points'] = slotpt_list
        anno['lines'] = slotline_list

        return anno

    def getImageSizeScale(self, img_w, img_h, model_w, model_h):
        # sw = 240.0 / img_w
        # sh = 288.0 / img_h
        sw = float(model_w) / img_w
        sh = float(model_h) / img_h
        return sw, sh

    @TimeProf
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        anno_f, image_f = self.dataset[idx]

        image, origin_shape = self.pull_img(image_f)
        rawh, raww, _ = origin_shape
        if self.phase == const.PHASE_TRAINING:
            slot_anno = self.pull_anno(anno_f)
            slot_maps = self.assigner.assign(slot_anno)
            anno = slot_maps
            # slot_maps = torch.from_numpy(slot_maps.astype(np.float32))
            slot_gt = np.zeros((2, self.h, self.w), dtype=np.float32)  # ch h w
            if self.transforms is not None:
                # image = self.img_transforms(image)
                trans_1 = self.transforms(
                    image=image, mask1=anno[0], mask2=anno[1])
                image_trans = trans_1['image']
                point_gt = trans_1['mask1']
                line_gt = trans_1['mask2']

                slot_gt[0] = point_gt
                slot_gt[1] = line_gt
                image = image_trans
            else:
                slot_gt[0] = anno[0]
                slot_gt[1] = anno[1]

            slot_maps = slot_gt.astype(np.float32)
            gt = slot_maps
        else:
            model_h = self.h
            model_w = self.w
            sw, sh = self.getImageSizeScale(raww, rawh, model_w, model_h)

            labelInstance = TXTLabelLoader(sw, sh)
            annotations = labelInstance.decodePointLineLabel(anno_f)
            gt = annotations
            # print("annotations\n", annotations)

        image_gt = self.img_transforms(
            image) if self.img_transforms is not None else image

        image_gt = image_gt.astype(np.float32).transpose(2, 0, 1) / 255.0
        data_dict = {'label': gt, "image": image_gt, "meta": {}}

        data_dict['meta']['last_img_path'] = image_f
        data_dict['meta']['task_name'] = self.task
        data_dict['meta']["camera_name"] = self.camera_names
        data_dict['meta']['frame_num'] = str(self.rank_local) + '_' + str(idx)
        data_dict['meta']['sw_sh'] = [sw, sh]

        return data_dict


def Get(dataset_temp, i, j):
    for k in range(i, j):
        print(k, dataset_temp[k]["dataloader_time"])


if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("inputs.pkl", 'rb'))

    random.seed(555)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    distributed.init_process_group(backend='nccl')
    train_dataset = PARKING_IPM_STADataset(*inputs)

    print(len(train_dataset))

    d = train_dataset[0]
    # print(d.keys())

    # print(d)

    from tools_scripts.data_format_cvt import ShowDataStruct
    from tools_scripts.vis_2d import Vis2D

    print(ShowDataStruct("image_gt", d["image"]))
    print(ShowDataStruct("slot_maps", d["label"]))

    train_dataset.save_all_heatmap('experiments/data_visual')
