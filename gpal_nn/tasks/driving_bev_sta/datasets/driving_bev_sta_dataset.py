
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
from gpal_lightning.neural_network.tasks.base.datasets.slice_base_dataset import SliceBaseDataset
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


polyline_class2id = {name: i for i, name in enumerate(map_classes_line)}
polyline_shape2id = {name: i for i, name in enumerate(shape_type)}
polyline_color2id = {name: i for i, name in enumerate(color_type)}
polyline_stop2id = {name: i for i, name in enumerate(stop_type)}

edge_class2id = {name: i for i, name in enumerate(map_classes_edge)}

polygon_class2id = {name: i for i, name in enumerate(map_classes_polygon)}
polygon_arrow2id = {name: i for i, name in enumerate(arrow_type)}


@DATASETS.register_module()
class DRIVING_BEV_STADataset(SliceBaseDataset):
    def __init__(self,
                 global_config: GlobalConfig,
                 task_config,
                 phase: str,
                 preprocess,
                 root_dir,
                 pkl_root,
                 pkl_infos,
                 in_shape,
                 dataset_name: str,
                 pseudo_labels_path: Union[str, list] = None,
                 worker: int = 0,
                 shuffle: bool = True,
                 shuffle_seed: int = 0,
                 sql_filter: str = "",
                 ratio: float = 0.0,
                 camera_name=['img_front_120'],
                 transforms=None,
                 test_mode=False,
                 gt_range=None,
                 inverse_int=True,
                 pts_per_vector=20,
                 is_random_scale_and_translate=False,
                 fast_buffer_path=""
                 ):
        '''
        :param root_dict:
        :param pkl_root:
        :param pkl_infos:
        :param in_shape:
        :param camera_names:
        :param transforms:
        :param test_mode:
        :param gt_range:
        :param inverse_int:
        :param pts_per_vector:
        :param using_loss_fn:
        :param pc_range:
        :param is_random_scale_and_translate:
        '''

        # import pickle as pkl
        # inputs = [global_config,
        #           task_config,
        #           phase,
        #           preprocess,
        #           root_dir,
        #           pkl_root,
        #           pkl_infos,
        #           in_shape,
        #           dataset_name,
        #           pseudo_labels_path,
        #           worker,
        #           shuffle,
        #           shuffle_seed,
        #           sql_filter,
        #           ratio,
        #           camera_name,
        #           transforms,
        #           test_mode,
        #           gt_range,
        #           inverse_int,
        #           pts_per_vector,
        #           is_random_scale_and_translate,
        #           fast_buffer_path]
        # pkl.dump(inputs, open("inputs.pkl", 'wb'))
        # exit(1)
        self.root_dir = root_dir
        self.in_shape = in_shape
        self.camera_names = camera_name
        assert gt_range is not None
        self.gt_range = gt_range
        self.inverse_int = inverse_int
        self.pts_per_vector = pts_per_vector
        self.is_random_scale_and_translate = is_random_scale_and_translate

        self.pkl_root = pkl_root
        self.pkl_infos = pkl_infos
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
                         fast_buffer_path="" if fast_buffer_path == "" else f"{fast_buffer_path}/{task_config.name}_buf"
                         )
        cut_start_h = 112
        mean = (0., 0., 0.)
        std = (255., 255., 255.)
        if phase == const.PHASE_TRAINING:
            self.transforms = [
                CutImageUpper(cut_start_h),
                MultiViewRandomCutOut(0.65, 6, [[40, 40]]),
                MultiViewPhotoMetricDistortion(),
                Normalize(mean=mean, std=std),
            ]
        else:
            self.transforms = [
                CutImageUpper(cut_start_h),
                Normalize(mean=mean, std=std),
            ]
        self.task = task_config.name

    def _build_world_data_list(self):
        try:
            rank_curr = distributed.get_rank()
            self.global_rank = rank_curr
            self.rank_local = distributed.get_rank() % 8
        except (RuntimeError, AssertionError):
            rank_curr = 0
            self.rank_local = 0

        self.world_data_list = self.load_data_infos(
            self.pkl_root, self.pkl_infos)

    def load_data_infos(self, pkl_root, pkl_infos):
        pkl_path = os.path.join(pkl_root, pkl_infos + '.pkl')
        data_infos = pickle.load(open(pkl_path, 'rb'))
        return data_infos

    def __len__(self):
        return len(self.dataset)

    def prepare_data(self, idx):
        cv2.setNumThreads(1)
        data_dict = {}
        data_dict['meta'] = {}
        data_dict['label'] = {}

        time_dp = DetailProf()
        time_dp.Tic("begin")

        data_info = copy.deepcopy(self.dataset[idx])
        ret, sub_prof = self.read_frame(data_info, data_dict)
        time_dp.AddSubProf("read_frame_sub", sub_prof)

        time_dp.Duration("read_frame", "begin")
        if ret is None:
            return None, sub_prof

        self.parse_annotations(data_info, data_dict['label'])
        time_dp.Duration("parse_annotations", "read_frame")

        time_dp.Duration("prepare_data_all", "begin")
        # if idx % 10 == 0:
        #     time_dp.Print()
        return data_dict, time_dp

    def Ego2Img(self, extrin, intrin):
        return intrin @ extrin

    def read_frame(self, data_info, data_dict):
        imgs = {}
        ego2cam = []
        cam2ego = []
        ists = []
        ists_norm = []
        dists = []
        ori_shape = []
        time_dp = DetailProf()
        time_dp.Tic("begin")
        for camera_name in self.camera_names:
            img_src, img, K, norm_K, ext, dist, ori_img_h, ori_img_w, img_path, time_dp_sub = self.read_single_camera(
                data_info['sensor'], camera_name)
            time_dp.AddSubProf(
                f"{camera_name}_read_single_camera", time_dp_sub)
            if img is None:
                return None, time_dp
            imgs[camera_name] = img
            ego2cam.append(ext)
            cam2ego.append(np.linalg.inv(np.concatenate(
                [ext, np.array([[0, 0, 0, 1]])], axis=0)))
            ists.append(K)
            ists_norm.append(norm_K)
            dists.append(dist)
            ori_shape.append([ori_img_h, ori_img_w])

        data_dict['image'] = imgs
        data_dict['calib'] = {'exts': np.stack(ego2cam), 'cam2egos': np.stack(
            cam2ego), 'ists': np.stack(ists), 'ists_norm': np.stack(ists_norm), 'dists': np.stack(dists)}
        data_dict['meta']['ori_shape'] = np.stack(ori_shape)
        data_dict['meta']['last_img_path'] = img_path
        data_dict['meta']['camera_name'] = self.camera_names
        data_dict['meta']['task_name'] = self.task
        data_dict['meta']['clip_id'] = '_'.join(img_path.split('/')[:2])

        return data_dict, time_dp

    def read_single_camera(self, sensor, camera_name):
        '''
        :param sensor:
        :param camera_name:
        :return: image, resize_image, K, norm_K, ext[:3, :], dist, ori_img_h, ori_img_w, img_name
        '''

        camera = sensor[camera_name]
        img_name = camera['img_path']
        rot = camera['extr']['rot']
        T = camera['extr']['T']
        K = camera['intr']['K']
        dist = camera['intr']['dist']

        # read img
        root_path = self.root_dir
        img_path = os.path.join(root_path, img_name)
        time_dp = DetailProf()
        time_dp.Tic("begin")
        try:
            image, hw_origin = self._image_buffer_access(
                img_path)
            if image is None:
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                image = cv2.undistort(image, K, dist)
                self._image_cache(
                    img_path, image, pre_resize=(960, 540), quality=100)
                # print("cache")
            else:
                # print("fast load")
                K[0, :] *= image.shape[1]/hw_origin[1]
                K[1, :] *= image.shape[0]/hw_origin[0]

            time_dp.Duration("imread", "begin")
            ori_img_h, ori_img_w, _ = image.shape
            resize_image, K = letterbox_image(image, self.in_shape, K=K)
            if self.is_random_scale_and_translate:
                resize_image, K = random_scale_and_translate(resize_image, self.in_shape, K=K, scale=0.1,
                                                             offset=0.1)
            time_dp.Duration("random_scale_and_translate", "imread")
        except Exception as e:
            print(e)
            print('Got None from : ', img_path)
            time_dp.Duration("Exception", "begin")
            return None, None, None, None, None, None, None, None, '', time_dp

        resize_img_h, resize_img_w, _ = resize_image.shape
        norm_K = np.array([
            [2. / resize_img_w, 0., -1.],
            [0., 2. / resize_img_h, -1.],
            [0., 0., 1.],
        ], dtype=K.dtype) @ K

        # parameter
        K = K.astype(np.float32)
        dist = dist.reshape(1, 5).astype(np.float32)
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = rot
        ext[:3, 3:] = T.reshape(3, 1)

        return image, resize_image, K, norm_K, ext[:3, :], dist, ori_img_h, ori_img_w, img_name, time_dp

    def parse_annotations(self, data_info, data_dict):
        annot = data_info['annotation']

        self.process_polylines(annot['polylines'], data_dict)
        self.process_edges(annot['edges'], data_dict)
        self.process_polygons_arrow(annot['polygons'], data_dict)

    def reorder_points(self, points):
        # turn it into far to near
        if points[0, 0] < points[-1, 0]:
            points = points[::-1]

        return points

    def process_polylines(self, polylines, data_dict):
        data_dict['polylines'] = {}
        lanes = []

        line_mask = np.zeros(len(polylines['points']), dtype=bool)
        for idx, lane in enumerate(polylines['points']):
            # print(idx)
            # TODO: move range filter to pipeline
            lane = _fix_pts_interpolate(
                lane, max(int(LineString(lane).length / 0.2), self.pts_per_vector))
            # print(lane)
            try:
                mask = lane[..., 0] <= self.gt_range[0]
                mask *= lane[..., 0] >= self.gt_range[3]
                mask *= lane[..., 1] <= self.gt_range[1]
                mask *= lane[..., 1] >= self.gt_range[4]
                lane = lane[mask]
            except:
                exit(1)

            if len(lane) <= 1:

                # print("continue", self.gt_range)
                continue

            line_mask[idx] = True
            lane = self.reorder_points(lane)
            lanes.append(lane)

        if len(lanes) > 0:
            points = np.array([_fix_pts_interpolate(
                item, self.pts_per_vector) for item in lanes])
            assert points.shape[
                1] == self.pts_per_vector, f'gp_points shape:{points.shape} is not {self.pts_per_vector}!'
            data_dict['polylines']['points'] = points

            self.get_polyline_cat_ids(polylines, data_dict, line_mask)

    def get_polyline_cat_ids(self, polylines, data_dict, masks):
        polyline_class_ids = []
        polyline_shape_ids = []
        polyline_color_ids = []
        polyline_stop_ids = []

        for mask, name, shape, color, stop in zip(masks, polylines['classes'],
                                                  polylines['shape_type'],
                                                  polylines['color_type'],
                                                  polylines['stop_type']):
            if mask == False:
                continue
            if name in map_classes_line:
                polyline_class_ids.append(polyline_class2id[name])
            if shape in shape_type:
                polyline_shape_ids.append(polyline_shape2id[shape])
            if color in color_type:
                polyline_color_ids.append(polyline_color2id[color])
            if stop in stop_type:
                polyline_stop_ids.append(polyline_stop2id[stop])

        data_dict['polylines']['classes'] = polyline_class_ids
        data_dict['polylines']['shape_type'] = polyline_shape_ids
        data_dict['polylines']['color_type'] = polyline_color_ids
        data_dict['polylines']['stop_type'] = polyline_stop_ids

    def process_edges(self, edges, data_dict):
        data_dict['edges'] = {}
        edges_line = []

        edge_mask = np.zeros(len(edges['points']), dtype=bool)
        for idx, edge in enumerate(edges['points']):
            # TODO: move range filter to pipeline
            edge = _fix_pts_interpolate(
                edge, max(int(LineString(edge).length / 0.2), self.pts_per_vector))
            mask = edge[..., 0] <= self.gt_range[0]
            mask *= edge[..., 0] >= self.gt_range[3]
            mask *= edge[..., 1] <= self.gt_range[1]
            mask *= edge[..., 1] >= self.gt_range[4]
            edge = edge[mask]
            if len(edge) <= 1:
                continue

            edge_mask[idx] = True
            edge = self.reorder_points(edge)
            edges_line.append(edge)

        if len(edges_line) > 0:
            points = np.array([_fix_pts_interpolate(
                item, self.pts_per_vector) for item in edges_line])
            assert points.shape[
                1] == self.pts_per_vector, f'gp_points shape:{points.shape} is not {self.pts_per_vector}!'
            data_dict['edges']['points'] = points

            self.get_edge_cat_ids(edges, data_dict, edge_mask)

    def get_edge_cat_ids(self, edges, data_dict, masks):
        edge_class_ids = []

        for mask, name in zip(masks, edges['classes']):
            if mask == False:
                continue
            if name in map_classes_edge:
                edge_class_ids.append(edge_class2id[name])

        data_dict['edges']['classes'] = edge_class_ids

    def process_polygons_arrow(self, polygons, data_dict):
        data_dict['polygon_arrows'] = {}
        polygons_points = []

        polygon_mask = np.zeros(len(polygons['points']), dtype=bool)
        for idx, polygon in enumerate(polygons['points']):
            # TODO: move range filter to pipeline
            mask = polygon[..., 0] <= self.gt_range[0]
            mask *= polygon[..., 0] >= self.gt_range[3]
            mask *= polygon[..., 1] <= self.gt_range[1]
            mask *= polygon[..., 1] >= self.gt_range[4]
            filter_polygon = polygon[mask]
            if len(filter_polygon) != len(polygon) or len(polygon) != 5:
                continue

            polygon_mask[idx] = True
            polygons_points.append(polygon)

        # process labels
        if len(polygons_points) > 0:
            data_dict['polygon_arrows']['points'] = polygons_points
            self.get_polygon_arrow_cat_ids(polygons, data_dict, polygon_mask)

    def get_polygon_arrow_cat_ids(self, polygons, data_dict, masks):
        polygon_class_ids = []
        arrow_type_ids = []

        for mask, name, arrow_type in zip(masks, polygons['classes'], polygons['arrow_type']):
            if mask == False:
                continue
            if name in map_classes_polygon:
                polygon_class_ids.append(polygon_class2id[name])
                arrow_type_ids.append(polygon_arrow2id[arrow_type])
            else:
                polygon_class_ids.append(-999)
                arrow_type_ids.append(-999)

        polygon_class_ids = np.array(polygon_class_ids)
        arrow_type_ids = np.array(arrow_type_ids)
        arrow_idx = map_classes_polygon.index('arrow')
        arrow_mask = (arrow_idx == polygon_class_ids)

        data_dict['polygon_arrows']['classes'] = arrow_type_ids[arrow_mask]

        arrow_array = []
        for arrow, mask in zip(data_dict['polygon_arrows']['points'], arrow_mask):
            if mask == True:
                arrow_array.append(arrow[1:])

        data_dict['polygon_arrows']['points'] = np.array(
            arrow_array, dtype=np.float32)

    @TimeProf
    def __getitem__(self, idx):
        # idx = 0

        while True:
            t1 = time.time()
            time_dp = DetailProf()
            time_dp.Tic("begin")

            data, sub_prof = self.prepare_data(idx)

            time_dp.AddSubProf("prepare_data_sub", sub_prof)
            time_dp.Duration("prepare_data", "begin")

            if data is None:
                print("_rand_another")
                idx = random.randint(0, len(self))
                continue

            if self.transforms is not None:
                for transform in self.transforms:
                    data = transform(data)

            time_dp.Duration("transform", "prepare_data")

            data['image'] = {k: data['image'][k].transpose(
                2, 0, 1) for k in data['image']}
            data['calib']["ego2imgs"] = np.stack(
                [i@e for e, i in zip(data['calib']['exts'], data['calib']['ists'])], axis=0)
            data['calib']["ego2imgs"] = np.stack([np.concatenate([ele, np.array(
                [[0, 0, 0, 1]])], axis=0) for ele in data['calib']["ego2imgs"]], axis=0)

            ists_wt = copy.deepcopy(data['calib']['ists'])
            ists_wt[:, 1, 2] += 112
            data['calib']["ego2imgs_wt"] = np.stack(
                [i@e for e, i in zip(data['calib']['exts'], ists_wt)], axis=0)
            data['calib']["ego2imgs_wt"] = np.stack([np.concatenate([ele, np.array(
                [[0, 0, 0, 1]])], axis=0) for ele in data['calib']["ego2imgs_wt"]], axis=0)

            data['calib']["img_shapes"] = np.stack(
                [np.array(list(img.shape)) for img in data["image"].values()], axis=0)

            time_dp.Duration("tail", "transform")
            time_dp.Duration("dataset_all", "begin")

            data['meta']['frame_num'] = str(self.rank_local) + '_' + str(idx)

            t2 = time.time()
            # if t2-t1 > 1.0:
            #     time_dp.Print()

            return data


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
    train_dataset = DRIVING_BEV_STADataset(*inputs)

    print(len(train_dataset))

    d = train_dataset[0]
    # print(d.keys())

    # print(d)
    # exit(1)

    from tools_scripts.data_format_cvt import ShowDataStruct
    from tools_scripts.vis_2d import Vis2D

    print(ShowDataStruct("d", d))

    # pool = Pool(processes=4)

    # for i in range(0, len(train_dataset), 5000):
    #     print(i)
    #     pool.apply_async(Get, (copy.deepcopy(train_dataset), i, i + 5000))
    #     # Get(train_dataset, i)
    # pool.close()
    # pool.join()

    # exit(1)

    for di, d in enumerate(train_dataset):
        # try:
        vis = Vis2D([-30, 100], [-40, 40], 0.02)
        for l in d["label"]["polylines"]["points"]:
            vis.DrawPolyline(l[:, :2], (255, 255, 255), 2)
        for l in d["label"]["edges"]["points"]:
            vis.DrawPolyline(l[:, :2], (0, 255, 255), 2)
        lanes = d["label"]["polylines"]["points"]
        lanes = np.concatenate([lanes, np.ones_like(lanes[:, :, :1])], axis=-1)
        curb = d["label"]["edges"]["points"]
        curb = np.concatenate([curb, np.ones_like(curb[:, :, :1])], axis=-1)
        imgs = []
        for img, extrin, intrin, intrin_norm, disr in zip(d["image"].values(), d["calib"]["exts"], d["calib"]["ists"], d["calib"]["ists_norm"], d["calib"]["dists"]):
            #  = calib["exts"], calib["ists"], calib["ists_norm"], calib["dists"]
            img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
            extrin = np.concatenate([extrin, np.array([[0, 0, 0, 1]])], axis=0)

            print(img.shape, intrin.shape, disr.shape, None, intrin.shape)
            img = cv2.undistort(img, intrin, disr, None, intrin)
            for l in np.concatenate([lanes, curb]):
                l_cam = extrin.dot(l.transpose()).transpose()
                img_u = (l_cam[:, 0] / l_cam[:, 2] * float(intrin[0, 0]
                                                           ) + float(intrin[0, 2]) + 0.5).astype(np.int32)
                img_v = (l_cam[:, 1] / l_cam[:, 2] * float(intrin[1, 1]
                                                           ) + float(intrin[1, 2]) + 0.5).astype(np.int32)
                pc_img_z = l_cam[:, 2]

                mask = ((img_u > 0) * (img_v > 0) * (img_u <
                        img.shape[1]) * (img_v < img.shape[0]) * (pc_img_z > 0.1)) == 1
                img_pts = np.stack([img_u, img_v], axis=-1)
                img_pts = img_pts[mask]

                cv2.polylines(img, [img_pts], False, (0, 255, 255), 4)
            imgs.append(img)

        imgs = np.concatenate(imgs, axis=0)

        img_bev = vis.Draw()
        preview_img_f = f"temp/bev_lane_debug_viz_{di}.jpg"
        cv2.imwrite(preview_img_f, img_bev)

        preview_img_f = f"temp/img_lane_debug_viz_{di}.jpg"
        print(preview_img_f)
        cv2.imwrite(preview_img_f, imgs)
