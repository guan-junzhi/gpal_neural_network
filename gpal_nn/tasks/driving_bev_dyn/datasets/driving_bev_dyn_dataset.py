
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
        self.data_list = data_list

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
            dict(
                NAME='build_targets_former',
                hm_size=OD_HEATMAP_OUT_HW,
                num_classes=len(CLASS_NAMES_LIST),
                max_objects=MAX_OBJ_NUMS,
                TARGET_ENABLED=dict(train=True, test=False)
            ),
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
        if True:
            intrinsic = []
            distort_coeff = []
            r_mat = []
            t_vec = []
            # breakpoint()
            for curr_view in self.image_view:
                curr_view_yaml_file = f"{WORKDIRS_ROOT}/gpal_neural_network_group/sikong/temp_dir_for_od/camera/{curr_view.replace('img_', '')}.yaml"
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
        image = read_img(str(img_file), self.image_resize + [3])
        image = cv2.resize(image, self.image_resize[::-1])
        image = image[self.img_crop_start[view_idx]:self.img_crop_start[view_idx] + self.img_h_len]
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
                exit(1)
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

        if (self.phase == const.PHASE_TRAINING) and (len(data_dict['gt_boxes']) == 0):
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        if data_dict.get('gt_names_former', None) is not None:
            data_dict.pop('gt_names_former', None)
        return data_dict

    @TimeProf
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        info = copy.deepcopy(self.dataset[idx])

        # print(f"__getitem__ {idx}")
        # print(info)
        # exit(1)
        # print(self.image_dir)
        # print(self.json_dir)
        # print(self.middle_json_str)

        input_dict = {}

        # 总起
        sequence_name = info['sequence_name']
        curr_time_stamp, prev_time_stamp = info['time_stamp'].split('/')

        # === 当前帧 格式必须统一
        curr_json_file = f'{self.json_dir}/{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json'
        curr_json_data = self.json_data.load(curr_json_file)
        re_curr_infos = self.json_data.parse_json(curr_json_data)
        meta_info, cameras, bounding_boxes, special_labels = re_curr_infos

        # print(ShowDataStruct("meta_info", meta_info))
        # print(ShowDataStruct("cameras", cameras))
        # print(ShowDataStruct("bounding_boxes", bounding_boxes))
        # print(ShowDataStruct("special_labels", special_labels))

        gt_boxes, gt_names = self.get_box(bounding_boxes=bounding_boxes)
        intrinsic, cam_dist, extrinsic, camera_sizes = self.get_camera_parameters(
            cam_infos=cameras)

        input_dict['gt_names'] = gt_names
        input_dict['gt_boxes'] = gt_boxes

        # print(f"gt_boxes = {gt_boxes[0]}")
        # print(f"input_dict['gt_boxes'] = {input_dict['gt_boxes'][0]}")

        # === 前一帧
        if self.have_prev_label:
            prev_json_file = f'{json_dir}/{sequence_name}/{middle_json_str}/{prev_time_stamp}.json'
            prev_json_data = self.json_data.load(prev_json_file)
            
            try:
                prev_json_data = self.json_data.load(prev_json_file)
            except:
                print(f'error json file: {prev_json_file}')
                
            re_prev_infos  = self.json_data.parse_json(prev_json_data)  # 上一帧不需要真值(但模型输出的有(是连续的))
            meta_info, cameras, bounding_boxes, special_labels = re_prev_infos

            gt_boxes_, gt_names_ = self.get_box(bounding_boxes=bounding_boxes)
            intrinsic, cam_dist, extrinsic, camera_sizes = self.get_camera_parameters(cam_infos=cameras)
            _, _, _, camera_sizes = self.get_camera_parameters(cam_infos=cameras)

            input_dict['gt_names_former'] = gt_names_
            input_dict['gt_boxes_former'] = gt_boxes_
        else:
            input_dict['gt_names_former'] = gt_names
            input_dict['gt_boxes_former'] = gt_boxes

        input_dict['gt_names_former'] = gt_names
        input_dict['gt_boxes_former'] = gt_boxes

        # print(f"gt_boxes = {gt_boxes[0]}")
        # print(f"input_dict['gt_boxes'] = {input_dict['gt_boxes'][0]}")
        # print(
        #     f"input_dict['gt_boxes_former'] = {input_dict['gt_boxes_former'][0]}")

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

        input_dict['intrinsic'] = intrinsic  # np.stack([intrinsic, intrinsic])
        input_dict['cam_dist'] = cam_dist  # np.stack([cam_dist, cam_dist])
        input_dict['extrinsic'] = extrinsic  # np.stack([extrinsic, extrinsic])
        input_dict['camera_names'] = self.image_view
        input_dict['camera_sizes'] = camera_sizes

        if self.image_dir == '/opt/airflow/process-prod-bucket/data_collect/./':
            current_image_save_path = f'/opt/airflow/local_datasets/tmp_train/od/{sequence_name}/{curr_time_stamp}.npy'
            previous_image_save_path = f'/opt/airflow/local_datasets/tmp_train/od/{sequence_name}/{prev_time_stamp}.npy'
            if not os.path.exists(current_image_save_path) or not os.path.exists(previous_image_save_path):
                os.makedirs(os.path.dirname(
                    current_image_save_path), exist_ok=True)
                current_images = np.zeros(
                    (7, self.img_crop_size[0], self.img_crop_size[1], 3), dtype=np.uint8)
                previous_images = np.zeros(
                    (7, self.img_crop_size[0], self.img_crop_size[1], 3), dtype=np.uint8)
                for view_idx, camera_view in enumerate(self.image_view):
                    image_file = f'{self.image_dir}/{sequence_name}/{camera_view}/{curr_time_stamp}.jpg'
                    current_img = self.get_image(
                        image_file, view_idx)  # cv2: BGR

                    current_images[view_idx] = current_img

                    if self.phase == const.PHASE_TRAINING:
                        # current_img = aug_image(current_img)
                        pass
                    input_dict[f'images_input{view_idx}'] = current_img.astype(
                        np.float32) / 255.0

                    image_file = f'{self.image_dir}/{sequence_name}/{camera_view}/{prev_time_stamp}.jpg'
                    previous_img = self.get_image(image_file, view_idx)
                    previous_images[view_idx] = previous_img
                    if self.phase == const.PHASE_TRAINING:
                        # previous_img = aug_image(previous_img)
                        pass
                    input_dict[f'images_input_former{view_idx}'] = previous_img.astype(
                        np.float32) / 255.0
                if not os.path.exists(current_image_save_path):
                    np.save(current_image_save_path, current_images)
                if not os.path.exists(previous_image_save_path):
                    np.save(previous_image_save_path, previous_images)
            else:
                current_images = np.load(current_image_save_path)
                previous_images = np.load(previous_image_save_path)
                for view_idx, camera_view in enumerate(self.image_view):
                    current_img = current_images[view_idx]
                    previous_img = previous_images[view_idx]
                    if self.training:
                        # current_img = aug_image(current_img)
                        # previous_img = aug_image(previous_img)
                        pass
                    input_dict[f'images_input{view_idx}'] = current_img.astype(
                        np.float32) / 255.0
                    input_dict[f'images_input_former{view_idx}'] = previous_img.astype(
                        np.float32) / 255.0
        else:
            for i, cur_view in enumerate(self.image_view):
                image_file = f'{self.image_dir}/{sequence_name}/{cur_view}/{curr_time_stamp}.jpg'
                img = self.get_image(image_file, i)  # -> BGR

                if self.phase == const.PHASE_TRAINING:
                    # img = aug_image(img)
                    pass
                input_dict[f'images_input{i}'] = img.astype(np.float32) / 255.0

                image_file = f'{self.image_dir}/{sequence_name}/{cur_view}/{prev_time_stamp}.jpg'
                img = self.get_image(image_file, i)
                if self.phase == const.PHASE_TRAINING:
                    # img = aug_image(img)
                    pass
                input_dict[f'images_input_former{i}'] = img.astype(
                    np.float32) / 255.0

        # 训练策略
        # if self.mode == "train":
        #     num_views = len(self.image_view)
        #     mask = np.ones(num_views, dtype=np.float32)  # 初始全1
        #     mask_nums = np.random.randint(0, num_views+1)  # 几个视角丢失？
        #     discard_indices = np.random.choice(num_views, size=mask_nums, replace=False)  # 无重复采样
        #     mask[discard_indices] = 0  # 对应位置置0
        #     for i in range(num_views):
        #         input_dict[f'images_input_{i}'] *= mask[i]
        data_dict = self.prepare_data(data_dict=input_dict)

        # print(data_dict["gt_curr_indices_center"])
        # exit(1)
        # print(ShowDataStruct("data_dict", data_dict))
        # exit(1)

        data_dict_ret = {
            "meta": {"frame_id": data_dict["frame_id"]}, 'image': {}, "label": {}, "calib": {}}
        for i in range(len(data_dict["camera_names"])):
            data_dict_ret['image'][data_dict["camera_names"]
                                   [i]] = data_dict[f"images_input{i}"].transpose(2, 0, 1)
            data_dict_ret['image'][data_dict["camera_names"]
                                   [i]+"_pre"] = data_dict[f"images_input_former{i}"].transpose(2, 0, 1)

        for key in data_dict:
            if "gt_curr_" in key:
                data_dict_ret["label"][key] = data_dict[key]
            if "gt_prev_" in key:
                data_dict_ret["label"][key] = data_dict[key]
        data_dict_ret["label"]["gt_boxes"] = data_dict["gt_boxes"]

        for key in ["intrinsic", "cam_dist", "extrinsic"]:
            data_dict_ret["calib"][key] = data_dict[key]

        data_dict_ret["calib"]["img_crop_dict"] = self.img_crop_dict

        data_dict_ret['meta']['camera_name'] = self.camera_names
        data_dict_ret['meta']['task_name'] = self.task
        frame_path = info['sequence_name'] + "/" + str(info['curr_index'])
        data_dict_ret['meta']['clip_id'] = '_'.join(frame_path.split('/')[:2])
        data_dict_ret['meta']['frame_num'] = str(self.rank_local) + '_' + str(idx)

        return data_dict_ret


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
    print(1, len(inputs))
    train_dataset = DRIVING_BEV_DYNDataset(*inputs)

    print(len(train_dataset))

    d = train_dataset[0]
    print(d.keys())
    print(d["label"]["gt_boxes"])
    exit(1)

    for d in train_dataset:
        # print(d["frame_id"])
        pass

    exit(1)
    # print(d)

    from tools_scripts.data_format_cvt import ShowDataStruct
    from tools_scripts.vis_2d import Vis2D

    print(ShowDataStruct("image_gt", d["image"]))
    print(ShowDataStruct("slot_maps", d["label"]))

    train_dataset.save_all_heatmap('experiments/data_visual')
