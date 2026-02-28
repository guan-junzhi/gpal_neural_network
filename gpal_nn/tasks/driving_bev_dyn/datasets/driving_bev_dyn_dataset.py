import copy
from multiprocessing import Pool
import pickle as pkl
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
from gpal_nn.tasks.driving_bev_dyn.utils import common_utils

from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.datasets.data_processor import DataProcessor
from pyquaternion import Quaternion
import torch.nn.functional as F
import scipy
from torchvision import transforms as T
from gpal_lightning.utils.deploy_utils import DistGridMap
from gpal_lightning.data.dataloader_helpers.clip_sampler import DatalistByclip

from gpal_nn.tasks.driving_bev_dyn.datasets.utils import read_pbtxt_file, create_extrinsic_matrix,create_intrinsic_matrix,parse_file

# 文件格式常量
FILE_FORMAT_PCD = '.pcd'
FILE_FORMAT_JPG = '.jpg'
FILE_FORMAT_TXT = '.txt'
FILE_FORMAT_JSON = '.json'


def read_img(files_img, image_resize=[360, 640, 3]):
    # try:
    if True:
        bin_data = cv2.imread(files_img)
        if bin_data is None:
            print(files_img, bin_data)
            bin_data = np.zeros(image_resize).astype(np.uint8)
            return bin_data, False
        else:
            return bin_data, True

def read_radar_point_cloud_from_pcd(radar_path):
    """支持ASCII格式的PCD文件读取函数"""
    if not os.path.exists(radar_path):
        raise FileNotFoundError(f"文件不存在: {radar_path}")
    try:
        with open(radar_path, 'r') as f:
            # 读取并解析头部
            header = {}
            while True:
                line = f.readline().strip()
                
                if line.startswith('DATA'):
                    header['data_type'] = line.split()[1]
                    # 记录数据开始位置
                    data_start = f.tell()
                    break
                
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        value = parts[1:] if len(parts) > 2 else parts[1]
                        header[key] = value
            
            # 解析字段信息
            fields = header['FIELDS'].split() if isinstance(header['FIELDS'], str) else header['FIELDS']
            sizes = [int(s) for s in header['SIZE'].split()] if isinstance(header['SIZE'], str) else [int(s) for s in header['SIZE']]
            types = header['TYPE'].split() if isinstance(header['TYPE'], str) else header['TYPE']
            points_count = int(header['POINTS'][0] if isinstance(header['POINTS'], list) else header['POINTS'])
            
            # 检查数据格式
            if header['data_type'] != 'ascii':
                raise ValueError(f"不支持的数据格式: {header['data_type']}，仅支持ascii格式")
            
            # 读取ASCII数据
            f.seek(data_start)
            lines = f.readlines()
            
            # 解析ASCII数据
            points_data = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 分割数据行
                    values = line.split()
                    if len(values) == len(fields):
                        # 转换为对应的数据类型
                        point = []
                        for i, (field, size, type_char) in enumerate(zip(fields, sizes, types)):
                            try:
                                if type_char == 'F':  # Float类型
                                    point.append(float(values[i]))
                                elif type_char == 'U':  # Unsigned整数类型
                                    point.append(int(values[i]))
                                elif type_char == 'I':  # Signed整数类型
                                    point.append(int(values[i]))
                                else:
                                    # 默认按浮点数处理
                                    point.append(float(values[i]))
                            except (ValueError, IndexError):
                                point.append(0.0)  # 转换失败时使用默认值
                        points_data.append(point)
            
# 转换为numpy数组
            if points_data:
                points_array = np.array(points_data, dtype=np.float32)
                
                # 过滤包含NaN的点
                valid_mask = ~np.isnan(points_array).any(axis=1)
                points_array = points_array[valid_mask]
                
                return points_array
            else:
                print("警告: 未读取到有效数据")
                return np.array([])
            
    except Exception as e:
        print(f"读取点云文件 {radar_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])

def parse_sensor_timestamps(file_path: str, max_lines: int = 10964) -> Dict[str, Dict[str, float]]:
    """解析传感器时间戳日志文件为字典结构

    Args:
        file_path: 日志文件路径
        max_lines: 最大读取行数

    Returns:
        Dict[str, Dict[str, float]]: 传感器名称为键，时间戳映射为值的嵌套字典
    """
    sensor_data: Dict[str, Dict[str, float]] = {}
    current_sensor: Optional[str] = None
    line_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                # if line_count > max_lines:
                #     break

                line = line.strip()
                if not line:
                    continue

                # 检测传感器名称行（以冒号结尾）
                if line.endswith(':'):
                    current_sensor = line[:-1].strip()
                    sensor_data[current_sensor] = {}  # 初始化传感器数据字典
                    continue

                # 处理时间戳行
                if current_sensor:
                    timestamp_data = _parse_timestamp_line(line)
                    if timestamp_data:
                        timestamp_ori, timestamp_now, time_diff = timestamp_data
                        sensor_data[current_sensor][f'{timestamp_now:.6f}'] = timestamp_ori

        return sensor_data

    except Exception as e:
        raise RuntimeError(f"解析传感器时间戳失败: {str(e)}") from e

def _parse_timestamp_line(line: str) -> Optional[Tuple[float, float, float]]:
    """解析单行时间戳数据

    Args:
        line: 包含时间戳的日志行

    Returns:
        Tuple[float, float, float] or None: 原始时间戳、当前时间戳和时间差，如果解析成功
    """
    # 检查文件格式
    if FILE_FORMAT_PCD in line:
        ext = FILE_FORMAT_PCD
    elif FILE_FORMAT_JPG in line:
        ext = FILE_FORMAT_JPG
    elif FILE_FORMAT_TXT in line:
        ext = FILE_FORMAT_TXT
    else:
        return None

    # 分割行并提取时间戳
    parts = line.split()
    if len(parts) < 3:
        return None

    try:
        timestamp_ori = float(parts[0].replace(ext, ''))
        timestamp_now = float(parts[2].replace(ext, ''))
        return timestamp_ori, timestamp_now, timestamp_ori - timestamp_now
    except (ValueError, IndexError):
        return None


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
        LOCAL_DATASETS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_LOCAL_DATASETS_ROOT")

        WORKDIRS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT")
        DATA_COLLECT_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT")
        self.has_label = global_config.has_label

        self.root_dir = os.path.join(DATASETS_ROOT, root_dir)
        self.json_dir = os.path.join(WORKDIRS_ROOT, json_dir)
        if self.has_label:
            self.image_dir = os.path.join(DATA_COLLECT_ROOT, image_dir)
        else:
            self.image_dir = "/data/ai_group/workdirs/od_occ_group/huiquyang/codes/gpal_neural_network/.vscode/data"
        # self.image_dir = "/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_od_pcdet/data/2025-09-23_16-40-52-312.4"
        self.middle_json_str = middle_json_str

        self.id_to_type = task_config.class_dict
        self.have_prev_label = have_prev_label
        self.camera_names = camera_name
        self.task = task_config.name

        print(self.id_to_type)
        self.type_to_id = {"-".join(self.id_to_type[k]): k for k in self.id_to_type}
        print(self.type_to_id)

        self.class_names = self.type_to_id.keys()
        print(self.class_names)
        # exit(1)

        self.image_view = camera_name

        # self.fusion_infos = []
        self.data_list = [{**ele, 'pkl_path': os.path.join(WORKDIRS_ROOT, ele['pkl_path'])}
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
        self.camera_raw_size = self.task_config.image_crop_config['CAMERA_RAW_SIZE']

        self.json_data = InitJsonFile(self.class_names, self.task_config.od_range)
        
        OD_HEATMAP_VOXEL_SIZE = self.task_config.build_gt["od_heatmap_voxel_size"]
        OD_HEATMAP_OUT_HW = [
            int(round((self.task_config.od_range[4] - self.task_config.od_range[1]) / OD_HEATMAP_VOXEL_SIZE[1],2)),
            int(round((self.task_config.od_range[3] - self.task_config.od_range[0]) / OD_HEATMAP_VOXEL_SIZE[0],2)),
        ]  # [96, 240]  # YX

        DATA_PROCESSOR = [
            dict(
                NAME='mask_points_and_boxes_outside_range',
                OCC_RANGE=self.task_config.od_range,
                REMOVE_OUTSIDE_BOXES=True
            ),
            dict(
                NAME='build_targets_track',
                hm_size=OD_HEATMAP_OUT_HW,
                num_classes=len(self.class_names),
                max_objects=self.task_config.build_gt["max_objects"],
            ),
        ]

        self.data_processor = DataProcessor(
            DATA_PROCESSOR, 
            point_cloud_range=np.array(self.task_config.od_range), 
            # placeholder
            training= phase == const.PHASE_TRAINING, 
            num_point_features=None
        )
        
        # 初始化共享内存管理器用于epoch同步
        self._shared_current_epoch = None
        self._shared_memory_manager = None
        self._setup_shared_epoch()
        
        self.current_epoch = self.get_current_epoch()
        # self.sequence_name_dict = self.get_current_epoch()
        

        # if self.dataset_cfg.USE_CAMERA_YAML:
        cam_calib_dir = ".vscode/calib/camera"
        cam_real = {
            "img_front_30": "camera_front_long",
            "img_front_120": "camera_front_wide",
            "img_front_left": "camera_front_left",
            "img_front_right": "camera_front_right",
            "img_rear_left": "camera_back_left",
            "img_rear_right": "camera_back_right",
            "img_back": "camera_back",
        }

        if not self.has_label:
            intrinsic = []
            distort_coeff = []
            r_mat = []
            t_vec = []
            # breakpoint()
            for curr_view in self.image_view:
                extrinsics_path = f"{cam_calib_dir}/{cam_real[curr_view]}_extrinsics.pb.txt"
                # curr_view_yaml_file = f"/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_od_pcdet/tools_own/read_update_cam_yaml_and_save_grid_valid/calibration-dev@1ac5e4038a8/JX_C5_1/vehicle_config/calibration/camera/{curr_view.replace('img_', '')}.yaml"
                # yaml_dict = read_camera_yaml_to_dict(curr_view_yaml_file)
                extrinsics_data = read_pbtxt_file(extrinsics_path)
                _, extrinsic_matrix_inv = create_extrinsic_matrix(extrinsics_data)
                intrinsics_path = f"{cam_calib_dir}/{cam_real[curr_view]}_intrinsics.pb.txt"
                intrinsics_data = read_pbtxt_file(intrinsics_path)
                intrinsic_matrix, distortion_coeffs = create_intrinsic_matrix(intrinsics_data)


                intrinsic.append(intrinsic_matrix.reshape(-1, 3, 3))
                distort_coeff.append(
                    distortion_coeffs.reshape(-1, 1, 5))
                r_mat.append(extrinsic_matrix_inv[ :3, :3].reshape(-1, 3, 3))
                t_vec.append(extrinsic_matrix_inv[ :3, 3].reshape(-1, 3, 1))

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

        self.deploy_eval = (phase != const.PHASE_TRAINING) and (global_config.onnx_path != None)
        
        self.subtask_name = self.global_config.Tasks['DRIVING_BEV_DYN']['SWITCH_SUBTASK']

    def _setup_shared_epoch(self):
        """设置共享内存用于epoch同步"""
        try:
            # 使用multiprocessing.Manager创建共享变量
            from multiprocessing import Manager
            self._shared_memory_manager = Manager()
            self._shared_current_epoch = self._shared_memory_manager.Value('i', 0)
            # 添加共享的sequence_name_dict
            self._shared_sequence_name_dict = self._shared_memory_manager.dict()
        except Exception as e:
            print(f"Warning: Failed to setup shared memory manager: {e}")
            # 如果共享内存失败，使用普通变量作为fallback
            self._shared_current_epoch = 0
            self._shared_sequence_name_dict = {}

    def set_current_epoch(self, epoch):
        """设置当前epoch，并同步到共享内存"""
        if hasattr(self._shared_current_epoch, 'value'):
            # 共享内存版本
            self._shared_current_epoch.value = epoch
        else:
            # 普通变量版本
            self._shared_current_epoch = epoch
        self.current_epoch = epoch
        
        # 随机设置sequence_name_dict中10%的关键字为False，其他为True
        # 使用共享内存确保所有工作进程状态一致
        if self.sequence_name_dict:
            import random
            
            # 设置确定性随机种子，基于epoch确保所有工作进程结果一致
            seed = 42 + self.current_epoch * 1000  # 固定种子，不依赖worker_id
            random.seed(seed)
            
            keys = list(self.sequence_name_dict.keys())
            # 对关键字进行确定性排序，确保所有工作进程顺序一致
            keys.sort()
            
            num_keys = len(keys)
            num_false = max(1, int(num_keys * 0.15))  # 至少设置1个为False
            
            # 使用确定性随机选择，确保所有工作进程选择相同的10%
            false_keys = random.sample(keys, num_false)
            
            # 更新共享内存中的sequence_name_dict
            if hasattr(self._shared_sequence_name_dict, 'update'):
                # 共享内存版本
                shared_dict = {}
                for key in keys:
                    shared_dict[key] = True
                for key in false_keys:
                    shared_dict[key] = False
                self._shared_sequence_name_dict.update(shared_dict)
            else:
                # 普通变量版本
                for key in keys:
                    self._shared_sequence_name_dict[key] = True
                for key in false_keys:
                    self._shared_sequence_name_dict[key] = False
            
            # 同步本地副本
            self.sequence_name_dict.update(self._shared_sequence_name_dict)
            
            print(f'current_epoch: {self.current_epoch}, sequence_name_dict updated: {num_false}/{num_keys} keys set to False')
        else:
            print(f'current_epoch: {self.current_epoch}, sequence_name_dict is empty')

    def get_current_epoch(self):
        """获取当前epoch，优先从共享内存读取"""
        if hasattr(self._shared_current_epoch, 'value'):
            # 共享内存版本
            return self._shared_current_epoch.value
        else:
            # 普通变量版本
            return self._shared_current_epoch
    def get_shared_sequence_name_dict(self):
        """获取共享的sequence_name_dict"""
        if hasattr(self, '_shared_sequence_name_dict'):
            if hasattr(self._shared_sequence_name_dict, 'copy'):
                # 共享内存版本
                return dict(self._shared_sequence_name_dict)
            else:
                # 普通变量版本
                return self._shared_sequence_name_dict.copy()
        else:
            return self.sequence_name_dict.copy()


    def include_fusion_data(self, phase):

        self.sequence_name_dict = {}
        print('Loading Mixed dataset ...')

        fusion_infos = []
        for info_path in self.data_list:
            if not os.path.exists(info_path['pkl_path']):
                print(info_path['pkl_path'], f' is not exists')
                continue
            with open(info_path['pkl_path'], 'rb') as f:
                infos = pickle.load(f)
                print(f'description: {info_path["description"]},', 
                      f' use_ratio: {info_path["use_ratio"]},', 
                      f' has {len(infos)} samples')
                fusion_infos.extend(infos)

        print('Total samples for Mixed dataset [原始数据]: %d' %(len(fusion_infos)))

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

        fusion_infos = [i for i in fusion_infos if i['sequence_name'].split('/')[-1] not in skip_subday_list]

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
            # self.fusion_infos = []
        else:
            for info in fusion_infos:
                self.sequence_name_dict.setdefault(info['sequence_name'], True)
        # fusion_infos = [fusion_infos[100]] * 6 
        # fusion_infos = fusion_infos[:1000]
        # if phase == const.PHASE_TRAINING:
        #     fusion_infos_ext = []
        #     for ele in fusion_infos:
        #         fusion_infos_ext.append(copy.deepcopy(ele))
        #         ele["curr_index"], ele["next_index"] = ele["next_index"], ele["curr_index"]
        #         ele["curr_timestamp"], ele["next_timestamp"] = ele["next_timestamp"], ele["curr_timestamp"]
        #         ele["time_stamp"] = ele["time_stamp"].split('/')[1] + '/' + ele["time_stamp"].split('/')[0]
        #         fusion_infos_ext.append(copy.deepcopy(ele))
        #     fusion_infos = fusion_infos_ext
        print('Total samples for Mixed dataset [指定日期过滤]: %d' %(len(fusion_infos)))
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

    def DistributeByClip(self, datalist, world_size, length_lim=15, rank_curr=0):
        epoch_len = len(datalist) // world_size
        datalist_by_clip = DatalistByclip(datalist, "scene")
        clip_key_list = [k for k in datalist_by_clip if len(datalist_by_clip[k]) > length_lim]
        res_clip_n_1 = []
        while len(res_clip_n_1) < (world_size - 1):
            res_clip_n_1 += clip_key_list[:world_size - 1 - len(res_clip_n_1)]
        clip_key_list += res_clip_n_1

        clip_keys_per_rank = len(clip_key_list) // world_size
        
        start_index = rank_curr * clip_keys_per_rank
        end_index = start_index + clip_keys_per_rank
        clip_keys_rank = pkl.loads(pkl.dumps(
            clip_key_list[rank_curr::world_size][:clip_keys_per_rank]))
        from tqdm import tqdm
        dataset = [ele for ele in tqdm(datalist, desc=f'初筛数据[补全rank] {world_size}-{rank_curr}') 
                   if ele["scene"] in clip_keys_rank]
        return dataset

    def _preconstruct_test_stream_indices(self, datalist, batch_size, key="sequence_name"):
        
        def GroupByclip(datalist, key="sequence_name", ret_idx=False, add_clip_idx=False):
            datalist_by_clip = {}
            for ele_i, ele in enumerate(datalist):
                clip_key = ele[key]
                if clip_key not in datalist_by_clip:
                    datalist_by_clip[clip_key] = []
                if ret_idx:
                    datalist_by_clip[clip_key].append(ele_i)
                else:
                    datalist_by_clip[clip_key].append(ele)
            
            if add_clip_idx:
                keys = list(datalist_by_clip.keys())
                new_datalist = []
                for clip_idx, clip_key in enumerate(keys):
                    clip_data = datalist_by_clip[clip_key]
                    for ele_idx, ele in enumerate(clip_data):
                        new_ele = ele.copy() if isinstance(ele, dict) else ele
                        if isinstance(new_ele, dict):
                            new_ele['clip_idx'] = clip_idx
                            new_ele['frame_idx'] = ele_idx
                        new_datalist.append(new_ele)
                return new_datalist
            return datalist_by_clip

        NEW_datalist = GroupByclip(datalist, key, ret_idx=False, add_clip_idx=True)  # 添加clip索引和frame索引用于debug
        datalist_by_clip = GroupByclip(NEW_datalist, key, ret_idx=True, add_clip_idx=False)
        
        from collections import Counter
        len_clip = Counter([len(ele) for ele in datalist_by_clip.values()])
        
        if len(len_clip) > 1 or len(NEW_datalist) % (list(len_clip.items())[0][0] * batch_size) != 0:
            for _ in range(5):
                print(f'Warning: 不同clip的帧数不同, {len_clip}')
                print(f'Warning: len % (max_length * batch_size * X) == 0 才可以时序, tot:{len(NEW_datalist)}, ')
                print(f'max_length:{list(len_clip.items())[0][0]} batch_size:{batch_size}')
                print(f'不再进行时序推理')
                print('--------------\n')
                
            re_range_idx = [i for i in range(len(NEW_datalist))]
            """
            cnt = 0
            for i in range(0, len(re_range_idx), batch_size):
                idx_in_batch = re_range_idx[i:i+batch_size]
                data_info = [f"batch {i//batch_size:05d} flatten_i {i:05d}: {NEW_datalist[idx]['clip_idx']:04d}^{NEW_datalist[idx]['time_stamp']}^{NEW_datalist[idx]['frame_idx']:04d}" for idx in idx_in_batch]
                cnt += len(idx_in_batch)
                print(f"  数据: {data_info}")
            print(f"总样本数: {cnt}")
            print(f"总clip数: {len(NEW_datalist)}")
            print(f"最大batch长度(从0开始): {len(NEW_datalist) // batch_size} 最大整数 faltten 索引 {len(NEW_datalist) // batch_size * batch_size}")
            """

            return [NEW_datalist[i] for i in re_range_idx]
    
        print(f'进行时序推理: {len(NEW_datalist)} 个样本, {len(datalist_by_clip)} 个clip')
        clip_key_list = list(datalist_by_clip.keys())
        epoch_len = sum(len(frames) for frames in datalist_by_clip.values())
        
        current_clip_idx = 0
        flatten_idxs = np.zeros([epoch_len, 2], dtype = np.int32) - 1
        for i in range(epoch_len):
            if (flatten_idxs[i, 0] < 0) or (flatten_idxs[i, 1] < 0):
                
                if current_clip_idx >= len(clip_key_list):
                    break

                clip_idx = current_clip_idx
                current_clip_idx += 1

                clip_key = clip_key_list[clip_idx]
                frames_in_clip = datalist_by_clip[clip_key]
                
                frame_start_idx = 0
                frame_end_idx = len(frames_in_clip)
                
                for j in range(frame_end_idx - frame_start_idx):
                    if (i + j * batch_size) >= epoch_len:
                        break
                    flatten_idxs[i + j * batch_size, 0] = clip_idx
                    flatten_idxs[i + j * batch_size, 1] = frame_start_idx + j  # 记录帧索引

        NEW_flatten_idxs = [datalist_by_clip[clip_key_list[ele[0]]][ele[1]]
                            for ele in flatten_idxs]
        new_datalist = [NEW_datalist[i] for i in NEW_flatten_idxs]
        
        """
        验证信息
        
        with open("test_flatten_idxs.log", "w") as f:
            cnt = 0
            for i in range(0, len(flatten_idxs), batch_size):
                idx_in_batch = NEW_flatten_idxs[i:i+batch_size]
                data_info = [f"batch {i//batch_size:05d} flatten_i {i:05d}: {NEW_datalist[idx]['clip_idx']:04d}^{NEW_datalist[idx]['time_stamp']}^{NEW_datalist[idx]['frame_idx']:04d}" for idx in idx_in_batch]
                cnt += len(idx_in_batch)
                # if i < 256:
                print(f"  数据: {data_info}", file=f)

            print(f"总样本数: {cnt}", file=f)
            print(f"总clip数: {len(clip_key_list)}", file=f)
            print(f"最大batch长度(从0开始): {len(NEW_datalist) // batch_size}, 最大整数faltten索引 {len(NEW_datalist) // batch_size * batch_size}", file=f)
        """
        
        return new_datalist

    def _distribute_data(self):
        try:
            rank_curr = distributed.get_rank()
            world_size = distributed.get_world_size()
        except (RuntimeError, AssertionError):
            rank_curr = 0
            world_size = 1
        
        if self.phase == const.PHASE_TRAINING:
            cut_data_list = self.DistributeByClip(self.world_data_list, world_size=world_size, length_lim=15, rank_curr=rank_curr)
        elif self.phase == const.PHASE_VALIDATION:
            """
            模拟训练时ClipSampler的行为, 但不考虑rank行为, 单卡测试
            """
            cut_data_list = cut_and_resample_sorted_data_list = self._preconstruct_test_stream_indices(
                self.world_data_list, 
                batch_size=self.global_config.image_per_gpu
            )
        else:
            raise NotImplementedError
        
        return cut_data_list

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

        # for cur_view in self.image_view:
        actual_views = list(self.json_data.cameras.keys())
        
        for cur_ref_view in self.image_view:
            ref_view_short = cur_ref_view.replace('img_', '')
            
            if ref_view_short in actual_views:
                cur_view = ref_view_short
            else:
                cur_view = cur_ref_view
                
            # assert cur_view == self.json_data.cameras[cur_view].name
            intrinsic.append(
                self.json_data.cameras[cur_view].intrinsic.to_matrix())
            cam_dist.append(
                self.json_data.cameras[cur_view].intrinsic.get_distortion_coeffs())
            extrinsic.append(
                self.json_data.cameras[cur_view].extrinsic.to_matrix())
            # camera_sizes.append(self.json_data.cameras[cur_view].image_size)
            camera_sizes.append(self.camera_raw_size[self.image_view.index(cur_ref_view)])

        # json_data_dict[cur_view] = {
        #     'image_size': camera_sizes,
        #     'camera_names': self.image_view,
        # }
        V = len(self.image_view)
        intrinsic = np.array(intrinsic).reshape(V, 3, 3)
        cam_dist = np.array(cam_dist).reshape(V, 1, 5)
        extrinsic = np.array(extrinsic).reshape(V, 4, 4)  # 4*4

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
            # vx, vy = 0.0, 0.0
            vx, vy = bbox.velocity[0], bbox.velocity[1]
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

    def get_image_by_slice(self, filepath, slice_timestamp, view_key, view_idx, crop_start, calib_intrin, calib_dist):
        """统一处理不同视角的图像"""
        img_file = filepath
        self.fast_buf_try_cnt += 1
        database_slice_key = "_".join(img_file.split('/')[-4:-2]+[slice_timestamp])
        image, hw_origin = self._slice_image_buffer_access(
            database_slice_key, view_key
        )
        if image is None:
            image, is_valid = read_img(str(img_file), self.image_resize + [3])
            if self.phase == const.PHASE_TRAINING:
                if is_valid:
                    if self.subtask_name in ['DRIVING_BEV_DYN_FISHEYE']:
                        pass
                    else:
                        image = cv2.undistort(image, calib_intrin, calib_dist, calib_intrin)  # 原图去畸变

                    image = cv2.resize(image, self.image_resize[::-1])
                image = image[crop_start:crop_start + self.img_h_len]
            self._slice_image_cache(
                database_slice_key, view_key, image, pre_resize=(image.shape[1], image.shape[0]), quality=100)
        else:
            self.fast_buf_sec_cnt += 1

        return image
    
    # def get_radar_by_slice(self, filepath, slice_timestamp, view_key):
    #     """统一处理不同视角的图像"""
    #     img_file = filepath
    #     self.fast_buf_try_cnt += 1
    #     database_slice_key = "_".join(img_file.split('/')[-4:-2]+[slice_timestamp])
    #     image, hw_origin = self._slice_image_buffer_access(
    #         database_slice_key, view_key
    #     )
    #     if image is None:
    #         image, is_valid = read_img(str(img_file), self.image_resize + [3])
    #         if self.phase == const.PHASE_TRAINING:
    #             if is_valid:
    #                 if self.subtask_name in ['DRIVING_BEV_DYN_FISHEYE']:
    #                     pass
    #                 else:
    #                     image = cv2.undistort(image, calib_intrin, calib_dist, calib_intrin)  # 原图去畸变

    #                 image = cv2.resize(image, self.image_resize[::-1])
    #             image = image[crop_start:crop_start + self.img_h_len]
    #         self._slice_image_cache(
    #             database_slice_key, view_key, image, pre_resize=(image.shape[1], image.shape[0]), quality=100)
    #     else:
    #         self.fast_buf_sec_cnt += 1

    #     return image
    
    # def _slice_radar_cache(self, slice_key, view_key, img, pre_resize, quality):
    #     if (self.buffer is None):
    #         return False

    #     if slice_key not in self.buffer_slice_write_cache:
    #         for k, v in self.buffer_slice_write_cache.items():
    #             ret = self.buffer.Cache(k, pickle.dumps(v))
    #             if not ret:
    #                 print(f"self.buffer.Cache {k} faild")

    #         self.buffer_slice_write_cache = {slice_key: {}}
    #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    #     h = int(img.shape[0]).to_bytes(4, byteorder='little', signed=False)
    #     w = int(img.shape[1]).to_bytes(4, byteorder='little', signed=False)
    #     img = cv2.resize(img, pre_resize)
    #     result, encimg = cv2.imencode('.jpg', img, encode_param)
    #     self.buffer_slice_write_cache[slice_key][view_key] = h + w + encimg.tobytes()
    #     return True

    # def _slice_radar_buffer_access(self, slice_key, view_key):
    #     if (self.buffer is None):
    #         return None, None
    #     try:
    #     # if True:
    #         if slice_key not in self.buffer_slice_read_cache:
    #             self.buffer_slice_read_cache = {slice_key: pickle.loads(self.buffer[slice_key])}
           
    #         x = self.buffer_slice_read_cache[slice_key][view_key]
    #         h = int().from_bytes(x[:4], byteorder='little', signed=False)
    #         w = int().from_bytes(x[4:8], byteorder='little', signed=False)
    #         return cv2.imdecode(np.frombuffer(x[8:], dtype=np.uint8), cv2.IMREAD_COLOR), [h, w]
    #     except Exception as e:
    #         # print(f"_slice_image_buffer_access failed {e}")
    #         return None, None

    def prepare_data(self, data_dict):
        if self.phase == const.PHASE_TRAINING:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'

        # print(data_dict['gt_names'], self.class_names)
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)

            not_selected = [i for i in range(len(data_dict['gt_names'])) if i not in selected]

            if not_selected != []:
                print(len(data_dict['gt_names'][selected]), data_dict['gt_names'][not_selected])
                # exit(1)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.type_to_id[n] for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'].reshape(-1, 10), 
                                       gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        data_dict.pop('gt_names', None)
        
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

    def getitem_driving_bev_dyn_subtask(self, idx):
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

            input_dict = {}

            sequence_name = info['sequence_name']
            json_dir = info["json_dir"]
            curr_time_stamp, prev_time_stamp = info['time_stamp'].split('/')
            # curr_time_stamp = f"{curr_time_stamp}0"

            # 无论预刷还是指标测试的数据格式/相对路径必须一致, f'{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json'
            curr_json_file = f'{self.json_dir}/{json_dir}/{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json'
            vcu_file =  f'{self.image_dir}/{sequence_name}/vcu/{curr_time_stamp}.txt'
            
            if 'SKYWELL' in sequence_name:
                WORKDIRS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT")
                curr_json_file = os.path.join(WORKDIRS_ROOT,f'od_occ_group/huiquyang/data/Obstacle_3DModelResult_L4/{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json')
                vcu_file       = f'{self.image_dir}/{sequence_name}/vcu/{curr_time_stamp}.txt'
            
            # seq,time_meas,time_pub,motion_info.vehicle_speed,motion_info.yaw_rate,motion_info.longitudinal_acceleration,motion_info.lateral_acceleration,motion_info.drive_mode,actuator_info.is_left_direction_light_on,actuator_info.is_right_direction_light_on,actuator_info.is_main_beam_on,actuator_info.is_dipped_beam_on,actuator_info.is_wiper_on,actuator_info.is_horn_on,actuator_info.is_left_direction_light_switch_on,actuator_info.is_right_direction_light_switch_on,actuator_info.front_left_door_status,actuator_info.front_right_door_status,actuator_info.rear_left_door_status,actuator_info.rear_right_door_status,actuator_info.rear_hatch_status,actuator_info.driver_safety_belt_status,actuator_info.is_brake_light_on,actuator_info.is_dangerous_warning_light_on,actuator_info.is_front_frog_light_on,actuator_info.is_rear_frog_light_on,actuator_info.is_reverse_direction_light_on,actuator_info.is_width_lamp_on,actuator_info.wiper_speed,actuator_info.is_washer_on,actuator_info.is_autodrive_active,axle_info[0].axis_id,axle_info[0].left_wheel_tire_pressure,axle_info[0].right_wheel_tire_pressure,axle_info[0].left_wheel_speed,axle_info[0].right_wheel_speed,axle_info[0].left_wheel_angle,axle_info[0].right_wheel_angle,axle_info[0].left_wheel_pulse,axle_info[0].right_wheel_pulse,axle_info[0].left_wheel_pulse_direction,axle_info[0].right_wheel_pulse_direction,axle_info[1].left_wheel_tire_pressure,axle_info[1].right_wheel_tire_pressure,axle_info[1].left_wheel_speed,axle_info[1].right_wheel_speed,axle_info[1].left_wheel_angle,axle_info[1].right_wheel_angle,axle_info[1].left_wheel_pulse,axle_info[1].right_wheel_pulse,axle_info[1].left_wheel_pulse_direction,axle_info[1].right_wheel_pulse_direction,powertrain_info.motor_speed,powertrain_info.motor_reference_torque,powertrain_info.motor_torque_change_rate,powertrain_info.battery_charge,powertrain_info.transmission_current_gear_level,powertrain_info.transmission_current_gear_position,powertrain_info.motor_torque_response,powertrain_info.throttle_percentage,powertrain_info.is_accelerator_pedal_override,powertrain_info.controlled_state_of_longitudinal_dynamic_system,powertrain_info.torque_request,powertrain_info.torque_feedback,powertrain_info.mcu_longitudinal_control_state_feedback,powertrain_info.mcu_driving_mode_feedback,steering_system_info.steering_wheel_angle,steering_system_info.steering_wheel_angle_speed,steering_system_info.steering_motor_torque,steering_system_info.steer_hands_on_status,steering_system_info.steer_angle_calibration_status,steering_system_info.received_steering_angle_request,steering_system_info.received_steering_torque_request,steering_system_info.eps_control_status,steering_system_info.eps_failure_reason,steering_system_info.steering_wheel_angle_control_failure_reason,steering_system_info.torque_control_failure_reason,steering_system_info.steering_wheel_angle_control_state,steering_system_info.torque_control_state,steering_system_info.mcu_lateral_control_state_feedback,steering_system_info.mcu_gear_control_state_feedback,brake_system_info.is_break_pedal_pressed,steering_system_info.is_abs_active,steering_system_info.is_epb_on,steering_system_info.brake_system_acceleration_response,steering_system_info.break_pedal_position,steering_system_info.is_brake_pedal_override,steering_system_info.is_vehicle_stand_still,steering_system_info.is_vehicle_park_stand_still,steering_system_info.braking_system_control_state,steering_system_info.mcu_brake_system_control_state_feedback,steering_system_info.epb_state
            
            
            # curr_json_file = "/data/ai_group/workdirs/od_occ_group/huiquyang/data/Obstacle_3DModelResult_/EKART_ID4001_2025-08-15-18-20-39/2025-08-15_18-34-44-232/3d_detection_json/1755254118.200182.json"
            if self.has_label:
                with open(vcu_file, 'r') as vcu_reader:
                    vcu = vcu_reader.readline().split('\t')
                ego_speed = float(vcu[3])
                ego_yaw_rate = float(vcu[4])
                curr_json_data = self.json_data.load(curr_json_file)
                re_curr_infos = self.json_data.parse_json(curr_json_data)
                meta_info, cameras, bounding_boxes, special_labels = re_curr_infos

                gt_boxes, gt_names = self.get_box(bounding_boxes=bounding_boxes)
                intrinsic, cam_dist, extrinsic, camera_sizes = self.get_camera_parameters(cam_infos=cameras)
            else:
                vcu_file = f'{self.image_dir}/{sequence_name}/vcu_slice/{curr_time_stamp}.pb.txt'
                gt_boxes = np.zeros((1, 10))
                gt_names = np.array(["vehicle_car"])
                vcu = parse_file(vcu_file)
                ego_speed = float( vcu['motion_info']['vehicle_speed']       )
                ego_yaw_rate = float(vcu['motion_info']['yaw_rate'])
                intrinsic = self.intrinsic
                cam_dist = self.cam_dist
                temp = np.stack([np.eye(4) for i in range(7)], axis=0)
                temp[:, :3:, :3] = self.r_mat_np
                temp[:, :3:, [3]] = self.t_vec_np
                extrinsic = temp

            input_dict['gt_names'] = gt_names
            input_dict['gt_boxes'] = gt_boxes

            radar_point_path = f'{self.image_dir}/{sequence_name}/pcd/{curr_time_stamp}.pcd'
            log_path = f'{self.image_dir}/{sequence_name}/logs/synced_files_log.txt'
            # sensor_timestamps = parse_sensor_timestamps(log_path)
            # img_real_timestamp = sensor_timestamps['img_front_120'][curr_time_stamp]
            radar_point = read_radar_point_cloud_from_pcd(radar_point_path)
            radar_point = radar_point[:,[0,1,2,4,5,10]]
            # radar_point[:,5] = radar_point[:,5]-float(img_real_timestamp)
            radar_point[:,5] = radar_point[:,5]-radar_point[:,5]
            #TODO 点云数据数据增强  随机屏蔽传感器数据  点云buffer  图像随机丢弃图像（目前有）

            time_dp.Duration("cur_json", "begin")

            # time_dp.Duration("prev_json", "cur_json")

            # === 共同信息
            input_dict['frame_id'] = info['time_stamp']


            input_dict['intrinsic'] = copy.deepcopy(intrinsic)  # np.stack([intrinsic, intrinsic])
            input_dict['cam_dist'] = copy.deepcopy(cam_dist)  # np.stack([cam_dist, cam_dist])
            input_dict['extrinsic'] = copy.deepcopy(extrinsic)  # np.stack([extrinsic, extrinsic])
            input_dict['camera_names'] = copy.deepcopy(self.image_view)
            # input_dict['camera_sizes'] = copy.deepcopy(camera_sizes)

            img_path = {}
            img_crop_dict = copy.deepcopy(self.img_crop_dict)
            for view_idx, camera_view in enumerate(self.image_view):
                image_file = f'{self.image_dir}/{sequence_name}/{camera_view}/{curr_time_stamp}.jpg'
                img_path[camera_view] = image_file

                calib_intrin = copy.deepcopy(input_dict['intrinsic'][view_idx])
                calib_extrin = copy.deepcopy(input_dict["extrinsic"][view_idx])
                calib_dist = copy.deepcopy(input_dict["cam_dist"][view_idx])
                if 'SKYWELL' in sequence_name:
                    img_crop_dict['CROP_HeSai_ID4']['CROP_START'][view_idx] = img_crop_dict['CROP_HeSai_ID4']['CROP_START_SKYWELL'][view_idx]
                crop_start = img_crop_dict['CROP_HeSai_ID4']['CROP_START'][view_idx]
                
                current_img = self.get_image_by_slice(image_file, curr_time_stamp, camera_view, view_idx, crop_start,
                                                      calib_intrin, calib_dist)

                if self.phase != const.PHASE_TRAINING:
                    input_dict[f'origin_images_input{view_idx}'] = current_img.astype(np.float32).copy()
                    
                    current_img2 = cv2.undistort(current_img, calib_intrin, calib_dist, calib_intrin)
                    current_img2 = cv2.resize(current_img2, self.image_resize[::-1])
                    current_img = current_img2[crop_start:crop_start + self.img_h_len]
                    # img_grid = DistGridMap(
                    #     current_img.shape[1],
                    #     current_img.shape[0],
                    #     calib_dist,
                    #     calib_intrin,
                    #     int(img_crop_dict["IMAGE_RESIZE"][1]),
                    #     int(img_crop_dict["IMAGE_RESIZE"][0]),
                    #     int(img_crop_dict["IMAGE_CROP_H_LEN"]),
                    #     int(img_crop_dict["CROP_HeSai_ID4"]["CROP_START"][view_idx]),
                    #     norm=False
                    # ).astype(np.float32)
                    
                    # current_img = cv2.remap(
                    #     current_img,
                    #     img_grid[...,0], 
                    #     img_grid[...,1],
                    #     interpolation=cv2.INTER_NEAREST
                    # )
                    calib_intrin[:2, :] /= float(img_crop_dict['CROP_HeSai_ID4']['SCALE'][view_idx])
                    calib_intrin[1, 2] -= float(img_crop_dict['CROP_HeSai_ID4']['CROP_START'][view_idx])
                else:
                    calib_intrin[:2, :] /= float(img_crop_dict['CROP_HeSai_ID4']['SCALE'][view_idx])
                    calib_intrin[1, 2] -= float(img_crop_dict['CROP_HeSai_ID4']['CROP_START'][view_idx])
                    # current_img = cv2.undistort(current_img, calib_intrin, calib_dist, calib_intrin)

                # image augmentation
                if self.phase == const.PHASE_TRAINING:
                    current_img = torch.from_numpy(current_img).unsqueeze(0).to("cpu").permute(0, 3, 1, 2).float()
                    cam_to_vehicle = np.linalg.inv(calib_extrin)
                    rot_temp = scipy.spatial.transform.Rotation.from_matrix(cam_to_vehicle[:3, :3]).as_quat()
                    rot_temp = Quaternion(rot_temp[3], rot_temp[0], rot_temp[1], rot_temp[2])
                    current_img, trans_cv, rots_cv = self.img_aug_cuda(
                        img_tensor=current_img, 
                        trans_cv=None, 
                        rots_cv=rot_temp, 
                        intrin=calib_intrin, 
                        device="cpu"
                    )
                    cam_to_vehicle[:3, :3] = rots_cv.rotation_matrix

                    input_dict["extrinsic"][view_idx] = np.linalg.inv(cam_to_vehicle)
                    current_img = current_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

                input_dict[f'images_input{view_idx}'] = current_img.astype(np.float32) / 255.0

            time_dp.Duration("image", "cur_json")

            data_dict = self.prepare_data(data_dict=input_dict)
            time_dp.Duration("prepare_data", "image")

            data_dict_ret = {
                "meta": {"frame_id": data_dict["frame_id"]}, 'image': {}, "label": {}, "calib": {}}
            for i in range(len(data_dict["camera_names"])):
                if not self.deploy_eval:
                    data_dict_ret['image'][data_dict["camera_names"]
                                        [i]] = data_dict[f"images_input{i}"].transpose(2, 0, 1)
                else:
                    data_dict_ret['image'][data_dict["camera_names"]
                                           [i]] = data_dict[f"origin_images_input{i}"]
                    
                    # 7V区分 3.0/4.0 主要是onnx的推理的图像输入尺寸
                    if hasattr(self.task_config, 'DEPLOY_CFG'):
                        if self.task_config.DEPLOY_CFG['mode'] == "gpal30_in_model_with_small_image":
                            data_dict_ret['image'][data_dict["camera_names"][i]] = input_dict[f"images_input{i}"] * 255.0

            for key in data_dict:
                if "gt_curr_" in key:
                    data_dict_ret["label"][key] = data_dict[key]
                if "gt_prev_" in key:
                    data_dict_ret["label"][key] = data_dict[key]
            data_dict_ret["label"]["gt_boxes"] = data_dict["gt_boxes"]

            for key in ["intrinsic", "cam_dist", "extrinsic"]:
                data_dict_ret["calib"][key] = data_dict[key]
            # data_dict_ret["calib"]["img_crop_dict"] = self.img_crop_dict
            data_dict_ret['calib']["img_shapes"] = np.stack(
                [np.array(list(img.shape)) for img in data_dict_ret["image"].values()], axis=0)
            data_dict_ret['calib']["bev_real2aug"] = np.eye(4, dtype=np.float32)

            intrinsics = copy.deepcopy(data_dict_ret['calib']["intrinsic"])
            for i in range(intrinsics.shape[0]):
                intrinsics[i, :2] /= img_crop_dict["CROP_HeSai_ID4"]['SCALE'][i]
                intrinsics[i, 1, 2] -= img_crop_dict["CROP_HeSai_ID4"]["CROP_START"][i]
            
            data_dict_ret['calib']["ego2imgs"] = np.stack(
                [i@e for e, i in zip(data_dict_ret['calib']['extrinsic'][:,:3], intrinsics)], axis=0)
            data_dict_ret['calib']["ego2imgs"] = np.stack([np.concatenate([ele, np.array(
                [[0, 0, 0, 1]])], axis=0) for ele in data_dict_ret['calib']["ego2imgs"]], axis=0)

            data_dict_ret['meta']['is_key'] = info.get('is_key', True)
            data_dict_ret['meta']['camera_name'] = self.camera_names
            data_dict_ret['meta']['task_name'] = self.task
            # data_dict_ret['meta']['img_path'] = img_path
            frame_path = info['sequence_name'] + "/" + str(info['curr_index'])
            data_dict_ret['meta']['clip_id'] = '^'.join(frame_path.split('/')[:2])
            data_dict_ret['meta']['timestamp'] = curr_time_stamp
            data_dict_ret['meta']['ego_speed'] = ego_speed
            data_dict_ret['meta']['ego_yaw_rate'] = ego_yaw_rate
            data_dict_ret['meta']['frame_num'] = str(self.rank_local) + '_' + str(idx)
            data_dict_ret['fast_buf_try_cnt'] = self.fast_buf_try_cnt
            data_dict_ret['fast_buf_sec_cnt'] = self.fast_buf_sec_cnt
            
            data_dict_ret['meta']['crop'] = np.array(img_crop_dict['CROP_HeSai_ID4']['CROP_START'])
            data_dict_ret['meta']['scale'] = np.array(img_crop_dict['CROP_HeSai_ID4']['SCALE'])
            if self.phase == const.PHASE_TRAINING:
                if not self._shared_sequence_name_dict.get(sequence_name, True):
                    data_dict_ret.update({"points": np.zeros_like(radar_point)})
                elif np.random.rand() < 0.2:
                    data_dict_ret.update({"points": np.zeros_like(radar_point)})
                else:
                    data_dict_ret.update({"points": radar_point.astype(np.float32)})
            else:
                data_dict_ret.update({"points": radar_point.astype(np.float32)})          
            # data_dict_ret.update({"points": radar_point.astype(np.float32)})       

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
        
    def getitem_subtask_driving_bev_byn_fisheye(self, idx):
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

            input_dict = {}

            sequence_name = info['sequence_name']
            curr_time_stamp, prev_time_stamp = info['time_stamp'].split('/')

            # === gtbox info ===
            curr_json_file = f'{self.json_dir}/{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json'
            vcu_file       = f'{self.image_dir}/{sequence_name}/vcu/{curr_time_stamp}.txt'
            
            if 'SKYWELL' in sequence_name:
                WORKDIRS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT")
                curr_json_file = os.path.join(WORKDIRS_ROOT,f'od_occ_group/huiquyang/data/Obstacle_3DModelResult_L4/{sequence_name}/{self.middle_json_str}/{curr_time_stamp}.json')
                vcu_file       = f'{self.image_dir}/{sequence_name}/vcu/{curr_time_stamp}.txt'
            
            with open(vcu_file, 'r') as vcu_reader:
                vcu = vcu_reader.readline().split('\t')
            
            # curr_json_file = "/data/ai_group/workdirs/od_occ_group/huiquyang/data/Obstacle_3DModelResult_/EKART_ID4001_2025-08-15-18-20-39/2025-08-15_18-34-44-232/3d_detection_json/1755254118.200182.json"
            curr_json_data = self.json_data.load(curr_json_file)
            ret_curr_infos = self.json_data.parse_json(curr_json_data)
            meta_info, cameras, bounding_boxes, special_labels = ret_curr_infos

            gt_boxes, gt_names = self.get_box(bounding_boxes=bounding_boxes)
            intrinsic, cam_dist, extrinsic, camera_sizes = self.get_camera_parameters(cam_infos=cameras)

            input_dict['gt_names'] = gt_names
            input_dict['gt_boxes'] = gt_boxes

            time_dp.Duration("cur_json", "begin")

            # time_dp.Duration("prev_json", "cur_json")

            # === common info ===
            input_dict['frame_id'] = info['time_stamp']

            # if self.dataset_cfg.USE_CAMERA_YAML:
            # if self.phase == const.PHASE_VALIDATION:
            #     intrinsic = self.intrinsic
            #     cam_dist = self.cam_dist
            #     temp = np.stack([np.eye(4) for i in range(7)], axis=0)
            #     temp[:, :3:, :3] = self.r_mat_np
            #     temp[:, :3:, [3]] = self.t_vec_np
            #     extrinsic = temp

            # TODO
            if 'SKYWELL' in sequence_name:
                intrinsic[:, 0, 0] *= 1.15363
                intrinsic[:, 1, 1] *= 1.15363
                intrinsic[:, 0, 2] = intrinsic[:, 0, 2] * 1.15363 - 127.0
                intrinsic[:, 1, 2] = intrinsic[:, 1, 2] * 1.15363 - 314.0
            
            input_dict['intrinsic'] = copy.deepcopy(intrinsic)  # np.stack([intrinsic, intrinsic])
            input_dict['cam_dist'] = copy.deepcopy(cam_dist)  # np.stack([cam_dist, cam_dist])
            input_dict['extrinsic'] = copy.deepcopy(extrinsic)  # np.stack([extrinsic, extrinsic])
            input_dict['camera_names'] = copy.deepcopy(self.image_view)
            # input_dict['camera_sizes'] = copy.deepcopy(camera_sizes)

            # === image info ===
            # img_path = {}
            for view_idx, camera_view in enumerate(self.image_view):
                if 'SKYWELL' in sequence_name:
                    WORKDIRS_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_WORKDIRS_ROOT")
                    image_file = os.path.join(WORKDIRS_ROOT, f'od_occ_group/huiquyang/data/Obstacle_3DModelResult_odom_undis_l4_mutli_fisheye_eq_image_data/{sequence_name}/{camera_view}/{curr_time_stamp}.jpg')
                    if self.deploy_eval:
                        image_file = f'{self.image_dir}/{sequence_name}/{camera_view}/{curr_time_stamp}.jpg'
                        camera_view = camera_view + "_deploy_raw"
                    assert os.path.exists(image_file), f"image_file {image_file} not exists"
                else:
                    image_file = f'{self.image_dir}/{sequence_name}/{camera_view}/{curr_time_stamp}.jpg'
                # img_path[camera_view] = image_file
                # current_img = self.get_image(image_file, view_idx)  # cv2: BGR
                crop_start = self.img_crop_start[view_idx]
                current_img = self.get_image_by_slice(image_file, curr_time_stamp, camera_view, view_idx, crop_start,
                                                      None, None)

                input_dict[f'origin_images_input{view_idx}'] = current_img.astype(np.float32).copy()

                if self.phase == const.PHASE_TRAINING:
                    current_img = torch.from_numpy(current_img).unsqueeze(0).to("cpu").permute(0, 3, 1, 2).float()
                    # current_img, trans_cv, rots_cv = self.img_aug_cuda(
                    #     img_tensor=current_img,
                    #     trans_cv=None,
                    #     rots_cv=None,
                    #     intrin=None,
                    #     device="cpu"
                    # )
                    if self.jitter and random.random() < 0.7:
                        for i in range(current_img.shape[0]):
                            current_img[i] = self.jitter(current_img[i]/255.0) * 255.0
                    current_img = current_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # B C H W -> B H W C
                            
                #     current_img = current_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if self.phase != const.PHASE_TRAINING:
                    current_img = cv2.resize(current_img, self.image_resize[::-1])
                    current_img = current_img[self.img_crop_start[view_idx]:self.img_crop_start[view_idx] + self.img_h_len]
                
                input_dict[f'images_input{view_idx}'] = current_img.astype(np.float32) / 255.0
                
            time_dp.Duration("image", "cur_json")
            data_dict = self.prepare_data(data_dict=input_dict)
            time_dp.Duration("prepare_data", "image")

            data_dict_ret = {
                "meta": {"frame_id": data_dict["frame_id"]}, 
                'image': {}, 
                "label": {}, 
                "calib": {},
            }
            
            for i in range(len(data_dict["camera_names"])):
                if not self.deploy_eval:
                    data_dict_ret['image'][data_dict["camera_names"][i]] = data_dict[f"images_input{i}"].transpose(2, 0, 1)
                else:
                    data_dict_ret['image'][data_dict["camera_names"][i]] = data_dict[f"origin_images_input{i}"]

            for key in data_dict:
                if "gt_curr_" in key:
                    data_dict_ret["label"][key] = data_dict[key]

            data_dict_ret["label"]["gt_boxes"] = data_dict["gt_boxes"]

            for key in ["intrinsic", "cam_dist", "extrinsic"]:
                data_dict_ret["calib"][key] = data_dict[key]

            data_dict_ret['calib']["img_shapes"] = np.stack([np.array(list(img.shape)) 
                                                             for img in data_dict_ret["image"].values()], axis=0)
            data_dict_ret['calib']["bev_real2aug"] = np.eye(4, dtype=np.float32)

            data_dict_ret['meta']['is_key'] = info.get('is_key', True)
            data_dict_ret['meta']['camera_name'] = self.camera_names
            data_dict_ret['meta']['task_name'] = self.task
            # data_dict_ret['meta']['img_path'] = img_path
            frame_path = info['sequence_name'] + "/" + str(info['curr_index'])
            data_dict_ret['meta']['clip_id'] = '^'.join(frame_path.split('/')[:2])
            data_dict_ret['meta']['timestamp'] = curr_time_stamp
            data_dict_ret['meta']['ego_speed'] = float(vcu[3])
            data_dict_ret['meta']['ego_yaw_rate'] = float(vcu[4])
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

    @TimeProf
    def __getitem__(self, idx):
        
        if self.subtask_name in ['DRIVING_BEV_DYN_FISHEYE']:
            data_dict = self.getitem_subtask_driving_bev_byn_fisheye(idx)
            return data_dict
        elif self.subtask_name in ['DRIVING_BEV_DYN']:
            data_dict = self.getitem_driving_bev_dyn_subtask(idx)
        else:
            raise NotImplementedError(f"subtask_name {self.subtask_name} not support")
        
        return data_dict


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