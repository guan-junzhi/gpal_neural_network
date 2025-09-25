
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
from gpal_nn.tasks.driving_bev_dyn.datasets.loader_utils import InitJsonFile, read_camera_yaml_to_dict
from gpal_nn.tasks.driving_bev_dyn.datasets.data_utils import aug_image
from gpal_nn.tasks.driving_bev_dyn.utils import common_utils

from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.datasets.data_processor import DataProcessor
from pyquaternion import Quaternion
import torch.nn.functional as F
import scipy
from torchvision import transforms as T


def read_img(files_img, image_resize=[360, 640, 3]):
    # try:
    if True:
        if 'jac4' in files_img and 'img_back' in files_img:  # 4号车无后视
            bin_data = np.zeros(image_resize).astype(np.uint8)
            return bin_data
        bin_data = cv2.imread(files_img)
        if bin_data is None:
            print(files_img, bin_data)
            bin_data = np.zeros(image_resize).astype(np.uint8)
    # except:
    #     bin_data = np.zeros(image_resize).astype(np.uint8)
    #     print(f"{files_img} img_data error")
    return bin_data


@DATASETS.register_module()
class DRIVING_BEV_DYNDataset(ImageBaseDataset):
    def __init__(self,
                 global_config: GlobalConfig,
                 task_config,
                 preprocess,
                 dataset_name: str,
                 phase: str,
                 camera_name=['img_front_120'],
                 root_dir='',
                 shuffle: bool = True,
                 shuffle_seed: int = 0,
                 sql_filter: str = "",
                 ratio: float = 0.0,
                 worker: int = 0,
                 pseudo_labels_path: Union[str, list] = None,
                 fast_buffer_path="",
                 data_list=[],
                 is_manual_label=False,
                 have_prev_label=False,
                 image_dir="",
                 json_dir="",
                 middle_json_str=""
                 ):

        # import pickle as pkl
        # inputs = [global_config,
        #           task_config,
        #           preprocess,
        #           dataset_name,
        #           phase,
        #           camera_name,
        #           root_dir,
        #           shuffle,
        #           shuffle_seed,
        #           sql_filter,
        #           ratio,
        #           worker,
        #           pseudo_labels_path,
        #           fast_buffer_path,
        #           data_list,
        #           is_manual_label,
        #           have_prev_label,
        #           image_dir,
        #           json_dir,
        #           middle_json_str]
        # pkl.dump(inputs, open("inputs.pkl", 'wb'))
        # exit(1)

        DATASETS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT")
        LOCAL_DATASETS_ROOT = os.getenv(
            "ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT")

        WORKDIRS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT")
        DATA_COLLECT_ROOT = os.getenv(
            "ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT")

        self.root_dir = os.path.join(DATASETS_ROOT, root_dir)
        self.json_dir = os.path.join(WORKDIRS_ROOT, json_dir)
        self.image_dir = os.path.join(
            WORKDIRS_ROOT if is_manual_label else DATA_COLLECT_ROOT, image_dir)
        # self.image_dir = "/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_od_pcdet/data/2025-09-23_16-40-52-312.4"
        self.middle_json_str = middle_json_str

        self.id_to_type = task_config.class_dict
        self.have_prev_label = have_prev_label
        self.camera_names = camera_name
        self.task = task_config.name

        print(self.id_to_type)
        self.type_to_id = {
            "-".join(self.id_to_type[k]): k for k in self.id_to_type}
        print(self.type_to_id)

        self.class_names = self.type_to_id.keys()
        print(self.class_names)
        # exit(1)

        # self.type_to_id = {}
        # for i, name in enumerate(class_names):
        #     self.type_to_id[name] = i + 1

        # self.use_occ = False
        # self.use_track = False

        # if training:
        #     self.logger.info('Total samples for track: %s' % (self.use_track))
        #     self.logger.info('Total samples for occ: %s' % (self.use_occ))
        # print(preprocess)
        # self.json_data = InitJsonFile(
        #     self.class_names, self.dataset_cfg.POINT_CLOUD_RANGE)
        self.image_view = camera_name

        # self.fusion_infos = []
        self.data_list = [os.path.join(WORKDIRS_ROOT, ele)
                          for ele in data_list]

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
                             LOCAL_DATASETS_ROOT, fast_buffer_path, f"{task_config.name}_buf_by_slice")
                         )

        self.img_crop_size = self.task_config.image_crop_config['IMAGE_CROP_SIZE']
        self.image_resize = self.task_config.image_crop_config['IMAGE_RESIZE']
        self.img_h_len = self.task_config.image_crop_config['IMAGE_CROP_H_LEN']
        self.img_crop_dict = self.task_config.image_crop_config
        self.img_crop_start = self.task_config.image_crop_config['CROP_HeSai_ID4']['CROP_START']

        self.json_data = InitJsonFile(
            self.class_names, self.task_config.od_range)
        OD_HEATMAP_VOXEL_SIZE = [0.64, 0.64, 0.5]
        OCC_RANGE = [-40.96, -25.6, -1.0, 81.92, 25.6, 5.0]
        RADAR_PREPROCESS_VOXEL_SIZE = [0.32, 0.32, 0.5]
        OD_HEATMAP_OUT_HW = [
            int((self.task_config.od_range[4] -
                self.task_config.od_range[1])/OD_HEATMAP_VOXEL_SIZE[1]),
            int((self.task_config.od_range[3] -
                self.task_config.od_range[0])/OD_HEATMAP_VOXEL_SIZE[0]),
        ]  # [96, 240]  # YX
        CLASS_NAMES_LIST = self.class_names
        MAX_OBJ_NUMS = 256

        DATA_PROCESSOR = [
            # dict(
            #     NAME='makeBEVMap',
            #     POINT_CLOUD_RANGE=OD_RANGE,
            #     OCC_RANGE=OCC_RANGE,
            #     VOXEL_SIZE=BEV_MAP_VOXEL_SIZE,
            #     BEV_ENABLED=True
            # ),
            dict(
                NAME='mask_points_and_boxes_outside_range',
                OCC_RANGE=OCC_RANGE,
                REMOVE_OUTSIDE_BOXES=True
            ),
            # dict(
            #     NAME='shuffle_points',
            #     SHUFFLE_ENABLED=dict(train=True, test=False)
            # ),
            # dict(
            #     NAME='transform_points_to_voxels',
            #     VOXEL_SIZE=RADAR_PREPROCESS_VOXEL_SIZE,
            #     MAX_POINTS_PER_VOXEL=32,
            #     VOXEL_FEATURES=8,
            #     MAX_NUMBER_OF_VOXELS=dict(train=100000, test=100000)
            # ),
            # dict(
            #     NAME='build_od_gt_targets',
            #     hm_size=OD_HEATMAP_OUT_HW,
            #     num_classes=len(CLASS_NAMES_LIST),
            #     max_objects=MAX_OBJ_NUMS,
            #     TARGET_ENABLED=dict(train=True, test=False)
            # ),
            dict(
                NAME='build_targets_track',
                hm_size=OD_HEATMAP_OUT_HW,
                num_classes=len(CLASS_NAMES_LIST),
                max_objects=MAX_OBJ_NUMS,
                TARGET_ENABLED=dict(train=True, test=False)
            ),
            # dict(
            #     NAME='build_targets_former',
            #     hm_size=OD_HEATMAP_OUT_HW,
            #     num_classes=len(CLASS_NAMES_LIST),
            #     max_objects=MAX_OBJ_NUMS,
            #     TARGET_ENABLED=dict(train=True, test=False)
            # ),
            # dict(
            #     NAME='build_occ',
            #     voxel_size=OCC_OUT_VOXEL_SIZE,
            #     pcr=OCC_RANGE,
            #     OCC_ENABLED=dict(train=True, test=False)
            # ),
            # dict(
            #     NAME='build_2d_targets',
            #     sample_size=[40, 96],
            #     max_objects=MAX_OBJ_NUMS,
            #     TARGET_ENABLED=dict(train=True, test=False)
            # )
        ]

        POINT_FEATURE_ENCODING = dict(
            encoding_type="absolute_coordinates_encoding",
            used_feature_list=["x", "y", "z",
                               "vr", "cv_ground", "power", "snr"],
            src_feature_list=["x", "y", "z", "vr",
                              "cv_ground", "power", "snr"],
        )
        self.data_processor = DataProcessor(
            DATA_PROCESSOR, point_cloud_range=np.array(
                self.task_config.od_range),
            training=phase == const.PHASE_TRAINING, num_point_features=0
        )

        # self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # self.root_split_path = self.dataset_cfg.DATA_PATH_LIST
        # self.class_names = class_names
        # self.logger = logger

        # if self.dataset_cfg.USE_CAMERA_YAML:

        cam_calib_dir = "camera_0811" if phase == const.PHASE_TRAINING else "camera"
        if True:
            intrinsic = []
            distort_coeff = []
            r_mat = []
            t_vec = []
            # breakpoint()
            for curr_view in self.image_view:
                curr_view_yaml_file = f"{WORKDIRS_ROOT}/gpal_neural_network_group/sikong/temp_dir_for_od/{cam_calib_dir}/{curr_view.replace('img_', '')}.yaml"
                # curr_view_yaml_file = f"/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_od_pcdet/tools_own/read_update_cam_yaml_and_save_grid_valid/calibration-dev@1ac5e4038a8/JX_C5_1/vehicle_config/calibration/camera/{curr_view.replace('img_', '')}.yaml"
                yaml_dict = read_camera_yaml_to_dict(curr_view_yaml_file)
                intrinsic.append(yaml_dict['camera_matrix'].reshape(-1, 3, 3))
                distort_coeff.append(
                    yaml_dict['distortion_coefficients'].reshape(-1, 1, 5))
                r_mat.append(yaml_dict['r_mat'].reshape(-1, 3, 3))
                t_vec.append(yaml_dict['t_vec'].reshape(-1, 3, 1))

            intrinsic_np = np.concatenate(intrinsic, axis=0)
            distort_coeff_np = np.concatenate(distort_coeff, axis=0)
            r_mat_np = np.concatenate(r_mat, axis=0)
            t_vec_np = np.concatenate(t_vec, axis=0)

            self.intrinsic = intrinsic_np
            self.cam_dist = distort_coeff_np
            self.r_mat_np = r_mat_np
            self.t_vec_np = t_vec_np

        self.jitter = T.ColorJitter([0.2, 1.2], 0.3, 0.3, 0.2)

        self.ClearFastBufCnt()

    def include_fusion_data(self, phase):

        print('Loading HeSai dataset ...')

        fusion_infos = []
        for info_path in self.data_list:
            # info_path = self.root_path / info_path
            if not os.path.exists(info_path):
                print(info_path, f' is not exists')
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                # infos = infos[:100]
                fusion_infos.extend(infos)

        print('Total samples for HeSai dataset: %d' %
              (len(fusion_infos)))

        skip_subday_list = [
            '2025-07-10_13-44-15-069',
            '2025-07-10_13-52-15-068',
            '2025-07-10_13-50-15-068',
            '2025-07-10_13-43-15-071',
            '2025-07-10_13-57-15-069',
            '2025-07-10_13-53-15-071',
            '2025-07-10_13-54-15-072',
            '2025-07-10_13-58-15-068',
            '2025-07-10_13-45-15-071',
            '2025-07-10_13-41-15-068',
            '2025-07-10_13-48-15-068',
            '2025-07-10_13-42-15-069',
            '2025-07-10_13-56-15-068',
            '2025-07-10_10-35-52-674',
            '2025-07-10_10-32-52-674',
            '2025-07-10_11-36-52-674',
            '2025-07-10_10-41-52-675',
            '2025-07-10_11-03-52-676',
            '2025-07-10_11-49-52-674',
            '2025-07-10_11-46-52-674',
            '2025-07-10_10-52-52-674',
            '2025-07-10_10-56-52-676',
            '2025-07-10_11-52-52-674',
            '2025-07-10_11-34-52-675',
            '2025-07-10_11-02-52-675',
            '2025-07-10_10-25-52-675',
            '2025-07-10_11-38-52-674',
            '2025-07-10_10-46-52-674',
            '2025-07-10_10-55-52-676',
            '2025-07-10_11-33-52-674',
            '2025-07-10_11-12-52-674',
            '2025-07-10_10-49-52-675',
        ]

        fusion_infos = [i for i in fusion_infos if i['sequence_name'].split(
            '/')[-1] not in skip_subday_list]

        if phase != const.PHASE_TRAINING:
            skip_subday_list = [
                "EKART_ID4001_2025-07-01-13-18-12",
                "EKART_ID4001_2025-07-01-15-45-23",
                "EKART_ID4001_2025-07-01-17-13-05",
                "EKART_ID4001_2025-07-05-12-56-38",
                "EKART_ID4001_2025-07-06-13-19-04",
                "EKART_ID4001_2025-07-06-13-48-04",
                "EKART_ID4001_2025-07-10-10-20-59/2025-07-10_11-36-52-674",
                "EKART_ID4001_2025-07-10-10-20-59/2025-07-10_11-46-52-674",
                "EKART_ID4001_2025-07-10-10-20-59/2025-07-10_11-52-52-674",
                "EKART_ID4001_2025-07-10-10-20-59/2025-07-10_11-38-52-674",
                "EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-46-14-102",
            ]
            fusion_infos_new = []
            for info in fusion_infos:
                flag_name_0 = info['sequence_name'].split('/')[0]
                flag_name_1 = info['sequence_name']
                # if flag_name_1 == 'EKART_ID4001_2025-07-10-09-30-00/2025-07-10_09-46-14-102':
                #     breakpoint()
                if flag_name_0 in skip_subday_list:
                    continue
                if flag_name_1 in skip_subday_list:
                    continue
                fusion_infos_new.append(info)

            fusion_infos = fusion_infos_new
            self.fusion_infos = []
        # fusion_infos = [fusion_infos[100]] * 6
        # if phase == const.PHASE_TRAINING:
        #     fusion_infos_ext = []
        #     for ele in fusion_infos:
        #         fusion_infos_ext.append(copy.deepcopy(ele))
        #         ele["curr_index"], ele["next_index"] = ele["next_index"], ele["curr_index"]
        #         ele["curr_timestamp"], ele["next_timestamp"] = ele["next_timestamp"], ele["curr_timestamp"]
        #         ele["time_stamp"] = ele["time_stamp"].split('/')[1] + '/' + ele["time_stamp"].split('/')[0]
        #         fusion_infos_ext.append(copy.deepcopy(ele))
        #     fusion_infos = fusion_infos_ext
        print('Total samples for HeSai dataset: %d' %
              (len(fusion_infos)))
        return fusion_infos

    def _build_world_data_list(self):
        try:
            rank_curr = distributed.get_rank()
            self.global_rank = rank_curr
            self.rank_local = distributed.get_rank() % 8
        except (RuntimeError, AssertionError):
            rank_curr = 0
            self.rank_local = 0

        self.world_data_list = self.include_fusion_data(self.phase)

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

    def get_camera_parameters(self, cam_infos):
        # === 内外参 === #
        json_data_dict = {}
        intrinsic = []
        cam_dist = []
        extrinsic = []
        camera_sizes = []

        for cur_view in self.image_view:
            assert cur_view == self.json_data.cameras[cur_view].name
            intrinsic.append(
                self.json_data.cameras[cur_view].intrinsic.to_matrix())
            cam_dist.append(
                self.json_data.cameras[cur_view].intrinsic.get_distortion_coeffs())
            extrinsic.append(
                self.json_data.cameras[cur_view].extrinsic.to_matrix())
            camera_sizes.append(self.json_data.cameras[cur_view].image_size)

        # json_data_dict[cur_view] = {
        #     'image_size': camera_sizes,
        #     'camera_names': self.image_view,
        # }

        intrinsic = np.array(intrinsic).reshape(7, 3, 3)
        cam_dist = np.array(cam_dist).reshape(7, 1, 5)
        extrinsic = np.array(extrinsic).reshape(7, 4, 4)  # 4*4

        return intrinsic, cam_dist, extrinsic, camera_sizes

    def get_box(self, bounding_boxes):
        obj_list = []
        gt_names_list = []
        is_visible_list = []

        for i, bbox in enumerate(bounding_boxes):
            x, y, z = bbox.position
            l, w, h = bbox.size
            heading = bbox.rotation[2]
            object_type = bbox.object_type
            trackid = bbox.track_id
            vx, vy = 0.0, 0.0
            is_visible = bbox.image_visible

            obj = [x, y, z, l, w, h, heading, 0.0, vx, vy]
            obj_list.append(obj)
            gt_names_list.append(object_type)
            is_visible_list.append(is_visible)

        gt_boxes = np.array(obj_list)
        gt_names = np.array(gt_names_list)
        is_visible = np.array(is_visible_list).astype(np.bool_)
        gt_boxes = gt_boxes[is_visible]
        gt_names = gt_names[is_visible]

        return gt_boxes, gt_names

    def get_image(self, filepath, view_idx):
        """统一处理不同视角的图像"""
        img_file = filepath
        self.fast_buf_try_cnt += 1
        database_key = "_".join(img_file.split('/')[-4:])
        image, hw_origin = self._image_buffer_access(database_key)
        if image is None:
            image = read_img(str(img_file), self.image_resize + [3])
            image = cv2.resize(image, self.image_resize[::-1])
            image = image[self.img_crop_start[view_idx]:self.img_crop_start[view_idx] + self.img_h_len]
            self._image_cache(
                database_key, image, pre_resize=(image.shape[1], image.shape[0]), quality=100)
        else:
            self.fast_buf_sec_cnt += 1

        return image

    def get_image_by_slice(self, filepath, slice_timestamp, view_key, view_idx):
        """统一处理不同视角的图像"""
        img_file = filepath
        self.fast_buf_try_cnt += 1
        database_slice_key = "_".join(img_file.split('/')[-4:-2]+[slice_timestamp])
        image, hw_origin = self._slice_image_buffer_access(
            database_slice_key, view_key)
        if image is None:
            image = read_img(str(img_file), self.image_resize + [3])
            image = cv2.resize(image, self.image_resize[::-1])
            image = image[self.img_crop_start[view_idx]:self.img_crop_start[view_idx] + self.img_h_len]
            self._slice_image_cache(
                database_slice_key, view_key, image, pre_resize=(image.shape[1], image.shape[0]), quality=95)
        else:
            self.fast_buf_sec_cnt += 1

        return image

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.phase == const.PHASE_TRAINING:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            '''data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )'''

        # print(data_dict['gt_names'], self.class_names)
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict['gt_names'], self.class_names)

            not_selected = [i for i in range(
                len(data_dict['gt_names'])) if i not in selected]

            if not_selected != []:
                print(len(data_dict['gt_names'][selected]),
                      data_dict['gt_names'][not_selected])
                # exit(1)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.type_to_id[n]
                                  for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'].reshape(-1, 10),
                                       gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        if data_dict.get('gt_boxes_former', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict['gt_names_former'], self.class_names)
            data_dict['gt_boxes_former'] = data_dict['gt_boxes_former'][selected]
            data_dict['gt_names_former'] = data_dict['gt_names_former'][selected]
            gt_classes = np.array(
                [self.type_to_id[n] for n in data_dict['gt_names_former']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes_former'].reshape(-1, 10),
                                       gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes_former'] = gt_boxes

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        # if (self.phase == const.PHASE_TRAINING) and (len(data_dict['gt_boxes']) == 0):
        #     new_index = np.random.randint(self.__len__())
        #     print(f"resample trig {new_index}")
        #     return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        if data_dict.get('gt_names_former', None) is not None:
            data_dict.pop('gt_names_former', None)
        return data_dict

    def ClearFastBufCnt(self):
        self.fast_buf_try_cnt = 0
        self.fast_buf_sec_cnt = 0


    def img_aug_cuda(self, img_tensor, trans_cv, rots_cv, intrin, device = "cuda:0"):
        noise_rot_mat = None
        # if self.task_config.ext_aug_conf and random.random() < 0.25:
        if random.random() < 0.5:
            trans_cv, rots_cv, noise_rot_mat = self.ext_augmentation(trans_cv, rots_cv)
            # print(trans_cv, rots_cv, noise_rot_mat)
        if noise_rot_mat is not None:
            img_tensor = self.remap_rotate_aug2_cuda(img_tensor, noise_rot_mat, intrin, device)
        if self.jitter and random.random() < 0.7:
            for i in range(img_tensor.shape[0]):
                img_tensor[i] = self.jitter(img_tensor[i]/255.0) * 255.0

        return img_tensor, trans_cv, rots_cv

    def ext_augmentation(self, trans_cv, rots_cv):
        max_noise_angle = [3, 3, 3]
        select = list(np.linspace(-max_noise_angle[0], max_noise_angle[0], 11))
        noise_angle = np.array([random.sample(select, 1)[0],
                                random.sample(select, 1)[0],
                                random.sample(select, 1)[0]])
        noise_angle = noise_angle * (np.pi / 180.)
        cos_noise_angle = np.cos(noise_angle)
        sin_noise_angle = np.sin(noise_angle)
        noise_rot_mat = np.array([1.0, 0.0, 0.0,
                                  0.0, cos_noise_angle[0], -sin_noise_angle[0],
                                  0.0, sin_noise_angle[0], cos_noise_angle[0]]).reshape(3, 3) @ \
                        np.array([cos_noise_angle[1], 0.0, -sin_noise_angle[1],
                                  0.0, 1.0, 0.0,
                                  sin_noise_angle[1], 0.0, cos_noise_angle[1]]).reshape(3, 3) @ \
                        np.array([cos_noise_angle[2], -sin_noise_angle[2], 0.0,
                                  sin_noise_angle[2], cos_noise_angle[2], 0.0,
                                  0.0, 0.0, 1.0]).reshape(3, 3)
        # print(rots_cv)
        # print(noise_rot_mat)
        rots_cv = rots_cv * Quaternion._from_matrix(noise_rot_mat)
        # print(rots_cv)
        # No noise for translation for now
        return trans_cv, rots_cv, noise_rot_mat
    
    def generate_homo_grid(self, homo, size, device = "cuda:0"):
        #assert type(size) == torch.Size
        N, C, H, W = size

        base_grid = homo.new(1, H, W, 3).to(device)
        linear_points = torch.linspace(-1, 1, W, device=device) if W > 1 else torch.Tensor([-1], device=device)
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H, device=device), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H, device=device) if H > 1 else torch.Tensor([-1], device=device)
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W, device=device)).expand_as(base_grid[:, :, :, 1])
        base_grid[:, :, :, 2] = 1
        grid = torch.bmm(base_grid.view(1, H * W, 3), homo.transpose(1, 2))
        grid = grid.view(1, H, W, 3)
        grid[:, :, :, 0] = grid[:, :, :, 0] / grid[:, :, :, 2]
        grid[:, :, :, 1] = grid[:, :, :, 1] / grid[:, :, :, 2]

        grid = grid[:, :, :, :2].float()
        return grid.repeat(N, 1, 1, 1)

    def remap_rotate_aug2_cuda(self, img, noise_rot_mat, intrin, device = "cuda:0"):
        N, C, H, W = img.shape

        transformation_matrix = np.dot(noise_rot_mat, np.linalg.inv(intrin))
        pts_src = np.array([[0, 0], [0, H-1], [W-1, 0], [W-1, H-1]])
        x_flat = pts_src[:,0]
        y_flat = pts_src[:,1]
        ones = np.ones_like(x_flat)
        camera_coords = np.dot(intrin, transformation_matrix) @ np.vstack((x_flat, y_flat, ones))    
        pts_dst = np.round(camera_coords[:2] / camera_coords[2]).T

        pts_dst[:, 0] = pts_dst[:, 0]  / (W-1) * 2.0 - 1.0
        pts_src[:, 0] = pts_src[:, 0]  / (W-1) * 2.0 - 1.0

        pts_dst[:, 1] = pts_dst[:, 1]  / (H-1) * 2.0 - 1.0
        pts_src[:, 1] = pts_src[:, 1]  / (H-1) * 2.0 - 1.0
        h, status = cv2.findHomography(pts_src, pts_dst)
        
        homo = torch.from_numpy(h).unsqueeze(0).to(device)
        homo_grid = self.generate_homo_grid(homo, img.shape, device)
        out = F.grid_sample(img, homo_grid).float()

        return out 
    @TimeProf
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        self.ClearFastBufCnt()

        time_dp = DetailProf()
        time_dp.Tic("begin")

        try:
            info = copy.deepcopy(self.dataset[idx])

            # print(f"__getitem__ {idx}")
            # print(info)
            input_dict = {}

            # 总起
            sequence_name = info['sequence_name']
            curr_time_stamp, prev_time_stamp = info['time_stamp'].split('/')

            # === 当前帧 格式必须统一
            curr_json_file = f'{self.json_dir}/{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json'
            # curr_json_file = "/data/ai_group/workdirs/od_occ_group/huiquyang/data/Obstacle_3DModelResult_/EKART_ID4001_2025-08-15-18-20-39/2025-08-15_18-34-44-232/3d_detection_json/1755254118.200182.json"
            curr_json_data = self.json_data.load(curr_json_file)
            re_curr_infos = self.json_data.parse_json(curr_json_data)
            meta_info, cameras, bounding_boxes, special_labels = re_curr_infos

            gt_boxes, gt_names = self.get_box(bounding_boxes=bounding_boxes)
            intrinsic, cam_dist, extrinsic, camera_sizes = self.get_camera_parameters(
                cam_infos=cameras)

            input_dict['gt_names'] = gt_names
            input_dict['gt_boxes'] = gt_boxes

            time_dp.Duration("cur_json", "begin")

            time_dp.Duration("prev_json", "cur_json")

            # === 共同信息
            input_dict['frame_id'] = info['time_stamp']

            # if self.dataset_cfg.USE_CAMERA_YAML:
            if True:
                intrinsic = self.intrinsic
                cam_dist = self.cam_dist
                temp = np.stack([np.eye(4) for i in range(7)], axis=0)
                temp[:, :3:, :3] = self.r_mat_np
                temp[:, :3:, [3]] = self.t_vec_np
                extrinsic = temp

            input_dict['intrinsic'] = copy.deepcopy(intrinsic)  # np.stack([intrinsic, intrinsic])
            input_dict['cam_dist'] = copy.deepcopy(cam_dist)  # np.stack([cam_dist, cam_dist])
            input_dict['extrinsic'] = copy.deepcopy(extrinsic)  # np.stack([extrinsic, extrinsic])
            input_dict['camera_names'] = copy.deepcopy(self.image_view)
            input_dict['camera_sizes'] = copy.deepcopy(camera_sizes)

            img_path = {}
            for view_idx, camera_view in enumerate(self.image_view):
                image_file = f'{self.image_dir}/{sequence_name}/{camera_view}/{curr_time_stamp}.jpg'
                img_path[camera_view] = image_file
                # current_img = self.get_image(image_file, view_idx)  # cv2: BGR
                current_img = self.get_image_by_slice(image_file, curr_time_stamp, camera_view, view_idx)

                calib_intrin = copy.deepcopy(input_dict['intrinsic'][view_idx])
                calib_extrin = copy.deepcopy(input_dict["extrinsic"][view_idx])
                calib_dist = copy.deepcopy(input_dict["cam_dist"][view_idx])
                img_crop_dict = copy.deepcopy(self.img_crop_dict)
                
                calib_intrin[:2, :] /= float(img_crop_dict['CROP_HeSai_ID4']['SCALE'][view_idx])
                calib_intrin[1, 2] -= float(img_crop_dict['CROP_HeSai_ID4']['CROP_START'][view_idx])

                current_img = cv2.undistort(
                    current_img, calib_intrin, calib_dist, calib_intrin)

                if self.phase == const.PHASE_TRAINING:
                    current_img = torch.from_numpy(current_img).unsqueeze(
                        0).to("cpu").permute(0, 3, 1, 2).float()
                    cam_to_vehicle = np.linalg.inv(calib_extrin)
                    rot_temp = scipy.spatial.transform.Rotation.from_matrix(
                        cam_to_vehicle[:3, :3]).as_quat()
                    rot_temp = Quaternion(rot_temp[3], rot_temp[0], rot_temp[1], rot_temp[2])
                    current_img, trans_cv, rots_cv = self.img_aug_cuda(
                        current_img, None, rot_temp, calib_intrin, device="cpu")
                    cam_to_vehicle[:3, :3] = rots_cv.rotation_matrix

                    input_dict["extrinsic"][view_idx] = np.linalg.inv(cam_to_vehicle)
                    current_img = current_img.squeeze(
                        0).permute(1, 2, 0).cpu().numpy()
    
                input_dict[f'images_input{view_idx}'] = current_img.astype(
                    np.float32) / 255.0

            time_dp.Duration("image", "prev_json")

            data_dict = self.prepare_data(data_dict=input_dict)
            time_dp.Duration("prepare_data", "image")

            data_dict_ret = {
                "meta": {"frame_id": data_dict["frame_id"]}, 'image': {}, "label": {}, "calib": {}}
            for i in range(len(data_dict["camera_names"])):
                data_dict_ret['image'][data_dict["camera_names"]
                                    [i]] = data_dict[f"images_input{i}"].transpose(2, 0, 1)
                # data_dict_ret['image'][data_dict["camera_names"]
                #                     [i]+"_pre"] = data_dict[f"images_input_former{i}"].transpose(2, 0, 1)

            for key in data_dict:
                if "gt_curr_" in key:
                    data_dict_ret["label"][key] = data_dict[key]
                if "gt_prev_" in key:
                    data_dict_ret["label"][key] = data_dict[key]
            data_dict_ret["label"]["gt_boxes"] = data_dict["gt_boxes"]

            for key in ["intrinsic", "cam_dist", "extrinsic"]:
                data_dict_ret["calib"][key] = data_dict[key]

            data_dict_ret["calib"]["cam_dist"] *= 0.0
            
            data_dict_ret["calib"]["img_crop_dict"] = self.img_crop_dict
            data_dict_ret['calib']["img_shapes"] = np.stack(
                [np.array(list(img.shape)) for img in data_dict_ret["image"].values()], axis=0)
            data_dict_ret['calib']["bev_real2aug"] = np.eye(4, dtype=np.float32)

            intrinsics = copy.deepcopy(data_dict_ret['calib']["intrinsic"])
            for i in range(intrinsics.shape[0]):
                intrinsics[i, :2] /= self.img_crop_dict["CROP_HeSai_ID4"]['SCALE'][i]
                intrinsics[i, 1,
                           2] -= self.img_crop_dict["CROP_HeSai_ID4"]["CROP_START"][i]
            
            data_dict_ret['calib']["ego2imgs"] = np.stack(
                [i@e for e, i in zip(data_dict_ret['calib']['extrinsic'][:,:3], intrinsics)], axis=0)
            data_dict_ret['calib']["ego2imgs"] = np.stack([np.concatenate([ele, np.array(
                [[0, 0, 0, 1]])], axis=0) for ele in data_dict_ret['calib']["ego2imgs"]], axis=0)


            data_dict_ret['meta']['camera_name'] = self.camera_names
            data_dict_ret['meta']['task_name'] = self.task
            data_dict_ret['meta']['img_path'] = img_path
            frame_path = info['sequence_name'] + "/" + str(info['curr_index'])
            data_dict_ret['meta']['clip_id'] = '_'.join(frame_path.split('/')[:2])
            data_dict_ret['meta']['frame_num'] = str(self.rank_local) + '_' + str(idx)
            data_dict_ret['fast_buf_try_cnt'] = self.fast_buf_try_cnt
            data_dict_ret['fast_buf_sec_cnt'] = self.fast_buf_sec_cnt

        except Exception as e:

            if self.phase == const.PHASE_TRAINING:
                new_index = np.random.randint(self.__len__())
                print(f"PHASE_TRAINING {idx} load faild {e}, resample trig {new_index}")
                return self.__getitem__(new_index)
            else:
                print(f"PHASE_TRAINING {idx} load faild {e}, faild exit(1)")
                exit(1)

        time_dp.Duration("move_data", "prepare_data")

        time_dp.Duration("dataset.getitem", "begin")
        # time_dp.Print()
        return data_dict_ret


def Get(dataset_temp, i, j):
    for k in range(i, j):
        print(k, dataset_temp[k]["dataloader_time"])


if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("ssd/inputs.pkl", 'rb'))

    random.seed(555)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    os.environ["ENV_GPAL_NEURAL_NETWORK_DATASETS_ROOT"] = '/data/ai_group/datasets/'
    os.environ["ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT"] ='/data1/'
    os.environ["ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT"] = '/data/ai_group/workdirs/'
    os.environ["ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT"] = '/data/dp_group/process-prod-bucket/data_collect/'


    distributed.init_process_group(backend='nccl')
    print(1, len(inputs))
    train_dataset = DRIVING_BEV_DYNDataset(*inputs)

    print(len(train_dataset))

    d = train_dataset[0]
    print(ShowDataStruct("d", d))

    import time
    t1 = time.time()
    for i in range(0, 10):
        print(i)
        d = train_dataset[i]
        print(d["fast_buf_sec_cnt"], d["fast_buf_try_cnt"] )
    t2 = time.time()
    d = train_dataset[0]
    print(t2-t1)
    # 无缓存 7.560542583465576
    # 帧缓存 5.709890842437744
    exit(1)

    from tools_scripts.data_format_cvt import ShowDataStruct
    from tools_scripts.vis_2d import Vis2D

    print(ShowDataStruct("image_gt", d["image"]))
    print(ShowDataStruct("slot_maps", d["label"]))

    train_dataset.save_all_heatmap('experiments/data_visual')
