from functools import partial

import numpy as np
# from skimage import transform
from gpal_nn.tasks.driving_bev_dyn.utils.fusion_utils import compute_radius, gen_hm_radius  # !!!!
from gpal_nn.tasks.driving_bev_dyn.utils import box_utils, common_utils
import math

# from ...ops.roiaware_pool3d.roiaware_pool3d_utils import (points_in_boxes_cpu,
#                                                           points_in_boxes_gpu)
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

camera_names_ = [
    ["img_front_120",
     "img_left_front",
     "img_left_back",
     "img_right_front",
     "img_right_back",
     "img_back",
     "img_front_30",],
    ["img_left",
     "img_right",
     "img_front_left",
     "img_front_right",
     "img_back_left",
     "img_back_right",
     "pass",
     ],
]

camera_shape_ = [
    [(3840, 2160),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (3840, 2160),],
    [(1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),
     (1920, 1080),]
]


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(
                tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(
                data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        # and self.training:
        if data_dict.get('gt_boxes', None) is not None and config['REMOVE_OUTSIDE_BOXES']:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get(
                    'min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        # OCC-Boxes 无论 traing or test 都作范围限制
        # mask outside range of occ
        if data_dict.get('points_occ', None) is not None:
            points_occ = data_dict['points_occ']
            # OCC_RANGE = np.array(config.OCC_RANGE, dtype=np.float32)
            minX, minY, minZ, maxX, maxY, maxZ = config.OCC_RANGE
            mask = (points_occ[:, 0] > minX) & (points_occ[:, 0] < maxX) & \
                   (points_occ[:, 1] > minY) & (points_occ[:, 1] < maxY) & \
                   (points_occ[:, 2] > minZ) & (points_occ[:, 2] < maxZ)
            points_occ = points_occ[mask]  # 就是mask
            # mask = common_utils.mask_points_by_range(data_dict['points_occ'], OCC_RANGE)
            # 绝对移除, 偶尔起作用，主要还是真值别出问题，不然拼接后各种意想不到的情况出现!
            if True:
                pc_range_ego = [-2.5, -1.4, -1.5, 2.5, 1.4, 2.5]
                masks = (points_occ[:, 0] > pc_range_ego[0]) & (points_occ[:, 0] < pc_range_ego[3]) & \
                        (points_occ[:, 1] > pc_range_ego[1]) & (points_occ[:, 1] < pc_range_ego[4]) & \
                        (points_occ[:, 2] > pc_range_ego[2]) & (points_occ[:, 2] < pc_range_ego[5]) & \
                        (points_occ[:, 3] == 5)
                points_occ = points_occ[~masks]  # 非自车区域内
                # # 周围地面
                pc_range_ego = [-6, -4, -0.1, 6, 4, 0.1]
                masks = (points_occ[:, 0] > pc_range_ego[0]) & (points_occ[:, 0] < pc_range_ego[3]) & \
                        (points_occ[:, 1] > pc_range_ego[1]) & (points_occ[:, 1] < pc_range_ego[4]) & \
                        (points_occ[:, 2] > pc_range_ego[2]) & (points_occ[:, 2] < pc_range_ego[5]) & \
                        (points_occ[:, 3] == 5)  # 背景残留
                points_occ = points_occ[~masks]  # 非地面点
            data_dict['points_occ'] = points_occ

        if data_dict.get('occ_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES:
            OCC_RANGE = np.array(config.OCC_RANGE, dtype=np.float32)
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['occ_boxes'], OCC_RANGE, min_num_corners=config.get(
                    'min_num_corners', 1)
            )
            data_dict['occ_boxes'] = data_dict['occ_boxes'][mask]

        if data_dict.get('points_former', None) is not None:
            mask = common_utils.mask_points_by_range(
                data_dict['points_former'], self.point_cloud_range)
            data_dict['points_former'] = data_dict['points_former'][mask]

        # and self.training:
        if data_dict.get('gt_boxes_former', None) is not None and config['REMOVE_OUTSIDE_BOXES']:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes_former'], self.point_cloud_range, min_num_corners=config.get(
                    'min_num_corners', 1)
            )
            data_dict['gt_boxes_former'] = data_dict['gt_boxes_former'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            if data_dict.get('points', None) is not None:
                points = data_dict['points']
                shuffle_idx = np.random.permutation(points.shape[0])
                points = points[shuffle_idx]
                data_dict['points'] = points

        return data_dict
    # OCC-单帧

    def remove_point(self, points_occ, for_move_boxes):

        points_occ_cuda = torch.from_numpy(
            points_occ[:, :3][np.newaxis, :, :].astype(np.float32)).cuda()
        for_move_boxes_cuda = torch.from_numpy(
            for_move_boxes[:, :7][np.newaxis, :, :].astype(np.float32)).cuda()
        pts_inRemoveBox_mask = points_in_boxes_gpu(
            points_occ_cuda,
            for_move_boxes_cuda,
        )  # 总的查找表
        del points_occ_cuda
        del for_move_boxes_cuda
        return pts_inRemoveBox_mask.squeeze(0).cpu().numpy()

    def build_occ(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_occ, config=config)

        if data_dict.get('points_occ', None) is not None:
            minX, minY, minZ, maxX, maxY, maxZ = config.pcr
            voxel_l, voxel_w, voxel_h = config.voxel_size
            hm_l, hm_w, hm_h = int(
                (maxX - minX) / voxel_l), int((maxY - minY) / voxel_w), int((maxZ - minZ) / voxel_h)
            # ----------------  clear/lidar-label, occ-data for filter invisible objects ----------------
            # 由于双range, gt和occ分别过滤出的框数量是不一致,对范围大的det-range作occ-range的约束 det > occ
            occ_range = np.array(config.pcr, dtype=np.float32)
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], occ_range, min_num_corners=config.get(
                    'min_num_corners', 1)
            )
            gt_boxes = data_dict['gt_boxes'][mask]  # less
            occ_boxes = data_dict['occ_boxes']  # more

            # new-repo
            gt_trackids = gt_boxes[:, 7]
            oc_trackids = occ_boxes[:, 7]

            gt_trackids_l = np.around(gt_trackids).tolist()
            oc_trackids_l = np.around(oc_trackids).tolist()

            points_occ = data_dict['points_occ']

            mask_inter = np.array(
                [row in gt_trackids_l for row in oc_trackids_l])
            # mask_inter = np.array([gt_trackids_l.index(row) if row in gt_trackids_l else -1 for row in oc_trackids_l ])
            if (len(oc_trackids_l) - sum(mask_inter)) > 0:
                for_move_boxes = occ_boxes[~mask_inter]
                for_move_boxes[:, 3:6] = for_move_boxes[:,
                                                        3:6] * 1.1  # lwh 扣除干净些
                for_move_boxes[:, 5:6] = for_move_boxes[:,
                                                        5:6] + 0.7  # h 扣除干净些
                # GPU-no
                # pts_inRemoveBox_mask = self.remove_point(points_occ, for_move_boxes)
                # pts_inRemoveBox_mask = points_in_boxes_gpu(points_occ, gt_class_boxes)  # 总的查找表
                # pts_inRemoveBox_mask = points_in_boxes_gpu(
                #     torch.from_numpy(points_occ[:, :3][np.newaxis, :, :].astype(np.float32)).cuda(),
                #     torch.from_numpy(for_move_boxes[:, :7][np.newaxis, :, :].astype(np.float32)).cuda(),
                #     )  # 总的查找表
                # pts_inRemoveBox_mask = pts_inRemoveBox_mask.squeeze(0).cpu().numpy()
                # points_occ = points_occ[pts_inRemoveBox_mask == -1]

                # CPU-default
                pts_inRemoveBox_mask = points_in_boxes_cpu(
                    points_occ[:, :3], for_move_boxes[:, :7])
                pts_inRemoveBox_mask = np.logical_or.reduce(
                    pts_inRemoveBox_mask, axis=0)  # N, Pts_num
                points_occ = points_occ[~pts_inRemoveBox_mask]  # 反向保留

            # 偏移矩阵， radar(减去)lidar
            # xyz_inter = gt_boxes[...,:3] - occ_boxes[mask_inter!=-1]  # Mx3
            # # 全部occboxes
            # occ_pts_inOccBox_mask = points_in_boxes_cpu(points_occ[:, :3], occ_boxes[:, :7])  # N, Pts_num
            # occ_pts_inOccBox_mask_outInter = occ_pts_inOccBox_mask[~mask_inter]  # 选取clear_label之外的
            # occ_pts_inOccBox_mask_outInter = np.logical_or.reduce(occ_pts_inOccBox_mask_outInter, axis=0)  # Pts_num
            # points_occ[:, :3] = points_occ[:,:3] + xyz_inter
            # points_occ = points_occ[~pts_inRemoveBox_mask]
            # box_mask = np.logical_or.reduce(occ_pts_inOccBox_mask, axis=1)  # N

            # ----------------  clear/occ-label, occ-data for filter invisible objects ----------------
            points_occ[:, 0] = (points_occ[:, 0] - minX) / voxel_l
            points_occ[:, 1] = (points_occ[:, 1] - minY) / voxel_w
            points_occ[:, 2] = (points_occ[:, 2] - minZ) / voxel_h
            points_occ = np.floor(points_occ).astype(np.int_)  # -> int-index
            hm_occ = np.zeros((hm_l, hm_h, hm_w), dtype=np.float32)
            # 在赋值前添加掩码
            # 由于浮点数精度和取整方式导致的边界计算
            valid_mask = (points_occ[:, 0] >= 0) & (points_occ[:, 0] < hm_l) & \
                         (points_occ[:, 1] >= 0) & (points_occ[:, 1] < hm_w) & \
                         (points_occ[:, 2] >= 0) & (points_occ[:, 2] < hm_h)
            points_occ = points_occ[valid_mask]
            hm_occ[points_occ[:, 0], points_occ[:, 2],
                   points_occ[:, 1]] = points_occ[:, 3]  # -> lhw

            data_dict['hm_occ'] = hm_occ  # 网格类别 cls_id:1~5  # for test

            # --------------------------- 在这里可做一些正样本权重初始化 ---------------------------
            # 利用 clear-label 中box生成 heatmap
            if config.OCC_ENABLED[self.mode]:
                bound_size_x, bound_size_y = maxX - minX, maxY - minY
                hm_main_center = np.zeros((5, 1, hm_w, hm_l), dtype=np.float32)
                for k in range(len(gt_boxes)):
                    # old-repo
                    if data_dict.get('oc_trackids') is not None:
                        x, y, z, l, w, h, yaw, v_car, v_target, cls_id = gt_boxes[k]
                    else:
                        x, y, z, l, w, h, yaw, track_id, v_car, v_target, cls_id = gt_boxes[k]
                    cls_id = int(cls_id)
                    if cls_id == 4:  # P | BCCP gtbox only 4-cls 1,2,3,4
                        continue
                    if not ((minX < x < maxX) and (minY < y < maxY)):
                        continue
                    if (w < 0) or (l < 0):
                        continue
                    bbox_l = l / bound_size_x * hm_l
                    bbox_w = w / bound_size_y * hm_w
                    radius = compute_radius(
                        (math.ceil(bbox_l), math.ceil(bbox_w)))
                    radius = max(0, int(radius))
                    # x --> y (invert to 2D image space)
                    center_x = (x - minX) / bound_size_x * hm_l
                    center_y = (y - minY) / bound_size_y * hm_w  # y --> x
                    center = np.array([center_x, center_y], dtype=np.float32)
                    center_int = center.astype(np.int32)
                    # hm_main_center 0~4 | gtbox: 1-4
                    gen_hm_radius(
                        hm_main_center[cls_id - 1, 0], center, radius)

                # --------------------------- 可以在这里做一些正样本权重的初始化 ---------------------------
                # 0 1 2 3 4 -> BCCPG 各类别  值区间为0~1
                hm_main_center[4] += 0.7  # Gkd   0.7 0.6|  因为第五类(背景)为0
                hm_main_center[3] += 1.4  # Ped   0.8 0.5|
                hm_main_center[2] += 0.8  # Cyc   0.3 0.5|
                hm_main_center[1] += 0.45  # Car   0.3|
                hm_main_center[0] += 0.4  # Bus   0.3|

                data_dict['pos_weight'] = hm_main_center
                # data_dict['pos_weight'] = 1
            # else:
            #     hm_occ_weight = (hm_occ != 0)
            #     hm_occ_weight = hm_occ_weight.astype(np.float32)
            #     data_dict['pos_weight'] = hm_occ_weight
            # ---------------------------

        return data_dict

    def build_2d_targets(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_2d_targets, config=config)
        if config.TARGET_ENABLED[self.mode]:
            hm_l, hm_w = config.sample_size
            hm_w, hm_l = int(hm_l), int(hm_w)

            mode = data_dict['vision_mode']
            if mode == 1:
                camera_names = camera_names_[1]
                camera_shape = camera_shape_[1]
            else:
                camera_shape = camera_shape_[0]
                camera_names = camera_names_[0]

            gt_boxes_list = data_dict['label_2d']
            hm_main_center = np.zeros((7, hm_w, hm_l), dtype=np.float32)

            cen_offset = np.zeros((7, 32, 2), dtype=np.float32)
            dimension = np.zeros((7, 32, 2), dtype=np.float32)
            indices_center = np.zeros((7, 32), dtype=np.int64)
            obj_mask = np.zeros((7, 32), dtype=np.int64)

            for i, camera_name in enumerate(camera_names):
                if camera_name == 'pass':
                    continue
                img_shape = camera_shape[i]

                gt_boxes = gt_boxes_list[i]

                if len(gt_boxes) == 0:
                    continue
                for j in range(len(gt_boxes)):
                    x, y,  w, h = gt_boxes[j]
                    bbox_h = h / img_shape[1] * hm_w
                    bbox_w = w / img_shape[0] * hm_l
                    radius = compute_radius(
                        (math.ceil(bbox_h), math.ceil(bbox_w)))
                    radius = max(0, int(radius))
                    center_x = x / img_shape[0] * hm_l
                    center_y = y / img_shape[1] * hm_w
                    center = np.array([center_x, center_y], dtype=np.float32)
                    center_int = center.astype(np.int32)
                    gen_hm_radius(hm_main_center[i], center, radius)
                    indices_center[i, j] = center_int[1] * hm_l + center_int[0]
                    cen_offset[i, j] = center - center_int
                    dimension[i, j, 0] = np.log(w)
                    dimension[i, j, 1] = np.log(h)
                    obj_mask[i, j] = 1
            if 'points_former' in data_dict:
                gt_boxes_list_former = data_dict['label_2d_former']
                hm_main_center_former = np.zeros(
                    (7, hm_w, hm_l), dtype=np.float32)

                cen_offset_former = np.zeros((7, 32, 2), dtype=np.float32)
                dimension_former = np.zeros((7, 32, 2), dtype=np.float32)
                indices_center_former = np.zeros((7, 32), dtype=np.int64)
                obj_mask_former = np.zeros((7, 32), dtype=np.int64)

                for i, camera_name in enumerate(camera_names):
                    if camera_name == 'pass':
                        continue
                    img_shape = camera_shape[i]

                    gt_boxes_former = gt_boxes_list_former[i]

                    if len(gt_boxes_former) == 0:
                        continue
                    for j in range(len(gt_boxes_former)):
                        x, y,  w, h = gt_boxes_former[j]
                        bbox_h = h / img_shape[1] * hm_w
                        bbox_w = w / img_shape[0] * hm_l
                        radius = compute_radius(
                            (math.ceil(bbox_h), math.ceil(bbox_w)))
                        radius = max(0, int(radius))
                        center_x = x / img_shape[0] * hm_l
                        center_y = y / img_shape[1] * hm_w
                        center = np.array(
                            [center_x, center_y], dtype=np.float32)
                        center_int = center.astype(np.int32)
                        gen_hm_radius(hm_main_center_former[i], center, radius)
                        indices_center_former[i,
                                              j] = center_int[1] * hm_l + center_int[0]
                        cen_offset_former[i, j] = center - center_int
                        dimension_former[i, j, 0] = np.log(w)
                        dimension_former[i, j, 1] = np.log(h)
                        obj_mask_former[i, j] = 1
                hm_main_center = np.stack(
                    (hm_main_center, hm_main_center_former), axis=0)
                cen_offset = np.stack((cen_offset, cen_offset_former), axis=0)
                dimension = np.stack((dimension, dimension_former), axis=0)
                indices_center = np.stack(
                    (indices_center, indices_center_former), axis=0)
                obj_mask = np.stack((obj_mask, obj_mask_former), axis=0)

            data_dict['2d_hm_cen'] = hm_main_center
            data_dict['2d_cen_offset'] = cen_offset
            data_dict['2d_dim'] = dimension

            data_dict['2d_indices_center'] = indices_center
            data_dict['2d_obj_mask'] = obj_mask
        return data_dict

    def build_det_targets(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_det_targets, config=config)

        if config.TARGET_ENABLED[self.mode]:
            det_range = self.point_cloud_range  # det-range 从dataset传过来的
            minX, maxX, minY, maxY, minZ, maxZ = det_range[0], det_range[3], det_range[1], \
                det_range[4], det_range[2], det_range[5]
            gt_boxes = data_dict['gt_boxes']  # 在getlabel那里需要规则需要设计清除

            num_objects = min(gt_boxes.shape[0], config['max_objects'])

            hm_w, hm_l = config.hm_size  # 小, 大  Y X
            hm_w, hm_l = int(hm_w), int(hm_l)  # img space
            bound_size_x, bound_size_y = maxX - minX, maxY - minY  # lidar space
            hm_main_center = np.zeros(
                (config.num_classes, hm_w, hm_l), dtype=np.float32)
            cen_offset = np.zeros((config['max_objects'], 2), dtype=np.float32)
            direction = np.zeros((config['max_objects'], 2), dtype=np.float32)
            z_coor = np.zeros((config['max_objects'], 1), dtype=np.float32)
            vel = np.zeros((config['max_objects'], 2), dtype=np.float32)
            dimension = np.zeros((config['max_objects'], 3), dtype=np.float32)
            indices_center = np.zeros(
                (config['max_objects']),    dtype=np.int64)
            obj_mask = np.zeros((config['max_objects']),    dtype=np.uint8)

            idd = []
            for k in range(num_objects):
                # old-repo
                if data_dict.get('oc_trackids') is not None:
                    x, y, z, l, w, h, yaw, v_car, v_target, cls_id = gt_boxes[k]
                else:
                    x, y, z, l, w, h, yaw, track_id, v_car, v_target, cls_id = gt_boxes[k]
                cls_id = int(cls_id)
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                bbox_l = l / bound_size_x * hm_l  # lidar space
                bbox_w = w / bound_size_y * hm_w  # lidar space
                #                if cls_id>1:
                #                    bbox_l = 2 / bound_size_x * hm_l
                #                    bbox_w = 2 / bound_size_y * hm_w
                # lidar space <=> image space
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                # x --> y (invert to 2D image space)
                center_x = (x - minX) / bound_size_x * hm_l
                center_y = (y - minY) / bound_size_y * hm_w  # y --> x
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)
                gen_hm_radius(hm_main_center[cls_id - 1], center, radius)
                indices_center[k] = center_int[1] * hm_l + center_int[0]

                cen_offset[k] = center - center_int
                dimension[k, 0] = l  # h
                dimension[k, 1] = w  # w
                dimension[k, 2] = h  # l
                direction[k, 0] = math.sin(float(yaw))  # im
                direction[k, 1] = math.cos(float(yaw))  # re
                z_coor[k, 0] = z
                obj_mask[k] = 1
                vel[k, 0] = np.clip(v_car / 40, 0, 1)
                vel[k, 1] = np.clip(v_target / 40, 0, 1)

            data_dict['hm_cen'] = hm_main_center  # C W L
            data_dict['cen_offset'] = cen_offset
            data_dict['direction'] = direction
            data_dict['z_coor'] = z_coor
            data_dict['dim'] = dimension
            data_dict['indices_center'] = indices_center
            data_dict['obj_mask'] = obj_mask
            data_dict['vel'] = vel

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (
                self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config['VOXEL_SIZE'])
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config['VOXEL_SIZE']
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config['VOXEL_SIZE'],
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config['MAX_POINTS_PER_VOXEL'],
                max_num_voxels=config['MAX_NUMBER_OF_VOXELS'][self.mode],
            )
        if data_dict.get('points', None) is not None:
            points = data_dict['points']
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output
            coordinates = np.concatenate(
                [coordinates, np.zeros((len(coordinates), 1))], axis=1)
            if data_dict.get('points_former', None) is not None:
                points1 = data_dict['points_former']
                voxel_output1 = self.voxel_generator.generate(points1)
                voxels1, coordinates1, num_points1 = voxel_output1
                voxels = np.concatenate([voxels, voxels1], axis=0)
                coordinates1 = np.concatenate(
                    [coordinates1, np.ones((len(coordinates1), 1))], axis=1)
                coordinates = np.concatenate(
                    [coordinates, coordinates1], axis=0)
                num_points = np.concatenate([num_points, num_points1], axis=0)

            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def makeBEVMap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.makeBEVMap, config=config)
        if config.BEV_ENABLED:
            PointCloud_ = data_dict['points']
            point_cloud_range = config.POINT_CLOUD_RANGE
            mask = np.where((PointCloud_[:, 0] >= point_cloud_range[0]) & (PointCloud_[:, 0] <= point_cloud_range[3]) & (PointCloud_[:, 1] >= point_cloud_range[1]) & (
                PointCloud_[:, 1] <= point_cloud_range[4]))
            PointCloud_ = PointCloud_[mask]
            PointCloud = np.copy(PointCloud_)
            Discretization_x, Discretization_y, _ = config.VOXEL_SIZE
            Width = int(
                (point_cloud_range[4]-point_cloud_range[1])/Discretization_y)
            Height = int(
                (point_cloud_range[3]-point_cloud_range[0])/Discretization_x)
            PointCloud[:, 0] = np.int_(
                np.floor((PointCloud[:, 0]-point_cloud_range[0]) / Discretization_x))
            PointCloud[:, 1] = np.int_(
                np.floor((PointCloud[:, 1]-point_cloud_range[1]) / Discretization_y))
            indices = np.lexsort(
                (-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
            PointCloud = PointCloud[indices]
            rMap = np.ones((Height, Width))
            gMap = np.ones((Height, Width))
            bMap = np.ones((Height, Width))
            _, indices = np.unique(
                PointCloud[:, 0:2], axis=0, return_index=True)
            PointCloud_frac = PointCloud[indices]
            intensity = np.abs(PointCloud_frac[:, 4]) / 10
            mask = intensity > 1
            intensity[mask] = intensity[mask] * 0 + 1
            wavelength = intensity * (780 - 450) + 450
            rlength = intensity * (780 - 450) + 450
            glength = intensity * (780 - 450) + 450
            blength = intensity * (780 - 450) + 450
            mask1 = (wavelength >= 380) & (wavelength < 440)
            rlength[mask1] = (-(rlength[mask1] - 440) / (440 - 380)) * 255
            glength[mask1] = 0 * glength[mask1]
            blength[mask1] = 255 + 0 * blength[mask1]
            mask3 = (wavelength >= 490) & (wavelength < 510)
            rlength[mask3] = 0 * rlength[mask3]
            glength[mask3] = 0 * glength[mask3] + 255
            blength[mask3] = (-(blength[mask3] - 510) / (510 - 490)) * 255*2
            mask4 = (wavelength >= 510) & (wavelength < 580)
            rlength[mask4] = ((rlength[mask4] - 510) / (580 - 510)) * 255
            glength[mask4] = 0 * glength[mask4] + 255
            blength[mask4] = 0 * blength[mask4]
            mask5 = (wavelength >= 580) & (wavelength < 645)
            rlength[mask5] = 0 * rlength[mask5] + 255
            glength[mask5] = (-(glength[mask5] - 781) / (781 - 645)) * 255
            blength[mask5] = 0 * blength[mask5]
            mask6 = (wavelength >= 645) & (wavelength < 781)
            rlength[mask6] = 255 + 0 * rlength[mask6]
            glength[mask6] = 0 * glength[mask6]
            blength[mask6] = 0 * blength[mask6]
            mask2 = (wavelength >= 440) & (wavelength < 490)
            rlength[mask2] = 0 * rlength[mask2]
            glength[mask2] = ((glength[mask2] - 440) / (490 - 440)) * 255*4
            blength[mask2] = 0 * blength[mask2] + 255
            rMap[np.int_(PointCloud_frac[:, 0])-1,
                 np.int_(PointCloud_frac[:, 1])-1] = rlength
            gMap[np.int_(PointCloud_frac[:, 0])-1,
                 np.int_(PointCloud_frac[:, 1])-1] = glength
            bMap[np.int_(PointCloud_frac[:, 0])-1,
                 np.int_(PointCloud_frac[:, 1])-1] = blength
            RGB_Map = np.ones((3, Width - 1, Height - 1)) * 255
            RGB_Map[2, :, :] = rMap[:(Height-1), :(Width-1)].T  # r_map
            RGB_Map[1, :, :] = gMap[:(Height-1), :(Width-1)].T  # g_map
            RGB_Map[0, :, :] = bMap[:(Height-1), :(Width-1)].T  # b_map
            data_dict['rgb_bev'] = RGB_Map/255
        return data_dict

    def build_targets(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_targets, config=config)
        if config.TARGET_ENABLED[self.mode]:
            point_cloud_range = self.point_cloud_range
            minX, maxX, minY, maxY, minZ, maxZ = point_cloud_range[0], point_cloud_range[
                3], point_cloud_range[1], point_cloud_range[4], point_cloud_range[2], point_cloud_range[5]
            gt_boxes = data_dict['gt_boxes']

            num_objects = min(gt_boxes.shape[0], config['max_objects'])
            hm_l, hm_w = config.hm_size
            hm_w, hm_l = int(hm_l), int(hm_w)
            bound_size_x, bound_size_y = maxX-minX, maxY-minY
            hm_main_center = np.zeros(
                (config.num_classes, hm_w, hm_l), dtype=np.float32)
            cen_offset = np.zeros((config['max_objects'], 2), dtype=np.float32)
            direction = np.zeros((config['max_objects'], 2), dtype=np.float32)
            z_coor = np.zeros((config['max_objects'], 1), dtype=np.float32)
            vel = np.zeros((config['max_objects'], 2), dtype=np.float32)
            dimension = np.zeros((config['max_objects'], 3), dtype=np.float32)
            indices_center = np.zeros((config['max_objects']), dtype=np.int64)
            obj_mask = np.zeros((config['max_objects']), dtype=np.uint8)

            idd = []

            for k in range(num_objects):
                x, y, z, l, w, h, yaw, v_car, v_target, cls_id = gt_boxes[k]
                cls_id = int(cls_id)
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                bbox_l = l / bound_size_x * hm_l
                bbox_w = w / bound_size_y * hm_w
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                # x --> y (invert to 2D image space)
                center_x = (x - minX) / bound_size_x * hm_l
                center_y = (y - minY) / bound_size_y * hm_w  # y --> x
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)
                gen_hm_radius(hm_main_center[cls_id-1], center, radius)
                indices_center[k] = center_int[1] * hm_l + center_int[0]
                cen_offset[k] = center - center_int
                dimension[k, 0] = l  # h
                dimension[k, 1] = w  # w
                dimension[k, 2] = h  # l
                direction[k, 0] = math.sin(float(yaw))  # im
                direction[k, 1] = math.cos(float(yaw))  # re
                z_coor[k, 0] = z
                obj_mask[k] = 1
                vel[k, 0] = np.clip(v_car/40, 0, 1)
                vel[k, 1] = np.clip(v_target/40, 0, 1)

            data_dict['hm_cen'] = hm_main_center
            data_dict['cen_offset'] = cen_offset
            data_dict['direction'] = direction
            data_dict['z_coor'] = z_coor
            data_dict['dim'] = dimension
            data_dict['indices_center'] = indices_center
            data_dict['obj_mask'] = obj_mask
            data_dict['vel'] = vel
        return data_dict

    def build_targets_former(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_targets_former, config=config)

        # if config['TARGET_ENABLED'][self.mode]:
        if True:
            point_cloud_range = self.point_cloud_range
            # minX,maxX, minY,maxY,minZ,maxZ=point_cloud_range[0],point_cloud_range[3],point_cloud_range[1],point_cloud_range[4],point_cloud_range[2],point_cloud_range[5]
            minX, minY, minZ, maxX, maxY, maxZ = point_cloud_range
            gt_boxes = data_dict['gt_boxes_former']
            num_objects = min(gt_boxes.shape[0], config['max_objects'])
            hm_l, hm_w = config['hm_size']
            hm_w, hm_l = int(hm_l), int(hm_w)
            bound_size_x, bound_size_y = maxX - minX, maxY - minY
            indices_track = np.zeros((config['max_objects']), dtype=np.int64)-1
            hm_main_center = np.zeros(
                (config['num_classes'], hm_w, hm_l), dtype=np.float32)
            indices_center = np.zeros((config['max_objects']), dtype=np.int64)
            obj_mask = np.zeros((config['max_objects']), dtype=np.uint8)
            former_xy = np.zeros((config['max_objects'], 2), dtype=np.uint8)

            for k in range(num_objects):
                x, y, z, l, w, h, yaw, track_id, car_v, target_v, cls_id = gt_boxes[k]
                target_v = np.min([target_v, 40])/40
                car_v = np.min([car_v, 40])/40
                cls_id = int(cls_id)
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                bbox_l = l / bound_size_x * hm_l
                bbox_w = w / bound_size_y * hm_w
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                center_x = (x - minX) / bound_size_x * hm_l
                center_y = (y - minY) / bound_size_y * hm_w
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)
                gen_hm_radius(hm_main_center[cls_id-1], center_int, radius)
                # indices_center[k] = center_int[1] * hm_l + center_int[0]
                '''if cls_id == 1:
                    w,l = max(2,w),max(3,l)'''
                # obj_mask[k] = 1
                # indices_track[k] = track_id

            # data_dict['former_hm_cen'] = hm_main_center
            data_dict['gt_prev_hm_cen'] = hm_main_center
            # data_dict['former_indices_center'] = indices_center
            # data_dict['former_obj_mask'] = obj_mask
            # data_dict['former_indice_track'] = indices_track

        return data_dict

    def build_targets_track(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_targets_track, config=config)

        # if config['TARGET_ENABLED'][self.mode]:
        if True:
            point_cloud_range = self.point_cloud_range
            minX, minY, minZ, maxX, maxY, maxZ = point_cloud_range
            gt_boxes = data_dict['gt_boxes']
            num_objects = min(gt_boxes.shape[0], config['max_objects'])
            hm_l, hm_w = config['hm_size']
            hm_w, hm_l = int(hm_l), int(hm_w)
            bound_size_x, bound_size_y = maxX - minX, maxY - minY
            hm_main_center = np.zeros(
                (config['num_classes'], hm_w, hm_l), dtype=np.float32)
            cen_offset = np.zeros((config['max_objects'], 2), dtype=np.float32)
            direction = np.zeros((config['max_objects'], 2), dtype=np.float32)
            z_coor = np.zeros((config['max_objects'], 1), dtype=np.float32)
            dimension = np.zeros((config['max_objects'], 3), dtype=np.float32)
            vel = np.zeros((config['max_objects'], 2), dtype=np.float32)
            indices_center = np.zeros((config['max_objects']), dtype=np.int64)
            obj_mask = np.zeros((config['max_objects']), dtype=np.uint8)
            indices_track = np.zeros(
                (config['max_objects']), dtype=np.int64) - 1

            idd = []
            for k in range(num_objects):
                x, y, z, l, w, h, yaw, track_id, car_v, target_v, cls_id = gt_boxes[k]
                target_v = np.min([target_v, 40])/40
                car_v = np.min([car_v, 40])/40
                cls_id = int(cls_id)
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                bbox_l = l / bound_size_x * hm_l
                bbox_w = w / bound_size_y * hm_w
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                # x --> y (invert to 2D image space)
                center_x = (x - minX) / bound_size_x * hm_l
                center_y = (y - minY) / bound_size_y * hm_w  # y --> x
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)
                gen_hm_radius(hm_main_center[cls_id-1], center_int, radius)
                indices_center[k] = center_int[1] * hm_l + center_int[0]
                cen_offset[k] = np.array([x, y])
                # cen_offset[k] = center - center_int
                '''if cls_id == 1:
                    w,l = max(2,w),max(3,l)'''
                dimension[k, 0] = l
                dimension[k, 1] = w
                dimension[k, 2] = h
                direction[k, 0] = math.sin(float(yaw))  # rad -> -1~1
                direction[k, 1] = math.cos(float(yaw))  # rad -> -1~1
                z_coor[k, 0] = z
                vel[k, 0] = car_v
                vel[k, 1] = target_v
                obj_mask[k] = 1
                indices_track[k] = track_id

            data_dict['gt_curr_vel'] = vel
            data_dict['gt_curr_hm_cen'] = hm_main_center
            data_dict['gt_curr_cen_offset'] = cen_offset
            data_dict['gt_curr_direction'] = direction
            data_dict['gt_curr_z_coor'] = z_coor
            data_dict['gt_curr_dim'] = dimension
            data_dict['gt_curr_indices_center'] = indices_center
            data_dict['gt_curr_obj_mask'] = obj_mask
            # data_dict['gt_curr_indice_track']   = indices_track

        return data_dict

    def build_od_gt_targets(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_od_gt_targets, config=config)

        # if config.TARGET_ENABLED[self.mode]:
        if True:
            minX, minY, minZ, maxX, maxY, maxZ = self.point_cloud_range
            gt_boxes = data_dict['gt_boxes']
            num_objects = min(gt_boxes.shape[0], config['max_objects'])
            hm_y, hm_x = config.hm_size
            hm_y, hm_x = int(hm_y), int(hm_x)
            bound_size_x, bound_size_y = maxX - minX, maxY - minY
            hm_main_center = np.zeros(
                (config.num_classes, hm_y, hm_x), dtype=np.float32)
            cen_offset = np.zeros((config['max_objects'], 2), dtype=np.float32)
            direction = np.zeros((config['max_objects'], 2), dtype=np.float32)
            z_coor = np.zeros((config['max_objects'], 1), dtype=np.float32)
            dimension = np.zeros((config['max_objects'], 3), dtype=np.float32)
            vel = np.zeros((config['max_objects'], 2), dtype=np.float32)
            indices_center = np.zeros((config['max_objects']), dtype=np.int64)
            obj_mask = np.zeros((config['max_objects']), dtype=np.uint8)
            indices_track = np.zeros(
                (config['max_objects']), dtype=np.int64) - 1

            for k in range(num_objects):
                x, y, z, l, w, h, yaw, track_id, car_vx, car_vy, cls_id = gt_boxes[k]
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                cls_id = int(cls_id)

                car_vx = np.clip(car_vx, -40, 40) / 40  # 双向裁剪和归一化到[-1,1]
                car_vy = np.clip(car_vy, -40, 40) / 40

                bbox_l = l / bound_size_x * hm_x  # x axis
                bbox_w = w / bound_size_y * hm_y  # y axis

                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = max(0, int(radius))

                center_x = (x - minX) / bound_size_x * hm_x
                center_y = (y - minY) / bound_size_y * hm_y
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)

                # 函数模型x-w,但是
                gen_hm_radius(hm_main_center[cls_id - 1], center_int, radius)
                # 本应: xy: 240, 96; 但是里面的数值height为x,所以需要对掉

                # import matplotlib.pyplot as plt
                # plt.imshow(hm_main_center[0])
                # plt.savefig('hm_main_center.png')
                # plt.show()

                indices_center[k] = center_int[1] * \
                    hm_x + center_int[0]  # x*hm_y + y 按照行展开
                cen_offset[k] = center - center_int

                dimension[k, 0] = l
                dimension[k, 1] = w
                dimension[k, 2] = h
                direction[k, 0] = math.cos(float(yaw))
                direction[k, 1] = math.sin(float(yaw))
                z_coor[k, 0] = z
                vel[k, 0] = car_vx
                vel[k, 1] = car_vy
                obj_mask[k] = 1
                indices_track[k] = track_id

            data_dict['gt_curr_hm_cen'] = hm_main_center
            data_dict['gt_curr_cen_offset'] = cen_offset
            data_dict['gt_curr_direction'] = direction
            data_dict['gt_curr_z_coor'] = z_coor
            data_dict['gt_curr_dim'] = dimension
            data_dict['gt_curr_vel'] = vel
            data_dict['gt_curr_indices_center'] = indices_center
            data_dict['gt_curr_obj_mask'] = obj_mask
            # data_dict['gt_curr_indice_track']   = indices_track

        return data_dict

    def build_track(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_track, config=config)
        if config.TRACK_ENABLED[self.mode]:
            point_cloud_range = self.point_cloud_range
            minX, maxX, minY, maxY, minZ, maxZ = point_cloud_range[0], point_cloud_range[
                3], point_cloud_range[1], point_cloud_range[4], point_cloud_range[2], point_cloud_range[5]
            gt_boxes = data_dict['gt_boxes']
            num_objects = min(gt_boxes.shape[0], config['max_objects'])
            hm_l, hm_w = config.hm_size
            hm_w, hm_l = int(hm_l), int(hm_w)
            bound_size_x, bound_size_y = maxX-minX, maxY-minY
            track_hm_center = np.zeros((1, hm_w, hm_l), dtype=np.float32)
            track_xy = np.zeros((config['max_objects'], 2), dtype=np.float32)
            former_xy = np.zeros((config['max_objects'], 2), dtype=np.float32)
            track_ind = np.zeros((config['max_objects']), dtype=np.int64)
            track_mask = np.zeros((config['max_objects']), dtype=np.uint8)
            track_i = np.zeros((1))
            for k in range(num_objects):
                x, y, z, l, w, h, yaw, track_id, v_car, v_target, cls_id = gt_boxes[k]
                if track_id not in data_dict['gt_boxes_former'][:, -4]:
                    continue
                mask = track_id == data_dict['gt_boxes_former'][:, -4]
                current_gtboxes = data_dict['gt_boxes_former'][mask][0]
                cls_id = 0
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                bbox_l = l / bound_size_x * hm_l
                bbox_w = w / bound_size_y * hm_w
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                center_x = (x - minX) / bound_size_x * hm_l
                center_y = (y - minY) / bound_size_y * hm_w
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)
                gen_hm_radius(track_hm_center[0], center_int, radius)
                track_ind[k] = center_int[1] * hm_l + center_int[0]
                track_i[0] += 1
                track_mask[k] = 1
                x1, y1, z1, l1, w1, h1, yaw1, track_id, v_car, v_target, cls_id = current_gtboxes
                track_xy[k, 0] = x1
                track_xy[k, 1] = y1
                former_xy[k, 0] = x
                former_xy[k, 1] = y

            data_dict['track_hm_center'] = track_hm_center  # cla的使用
            data_dict['track_xy'] = track_xy
            data_dict['former_xy'] = former_xy
            data_dict['track_ind'] = track_ind
            data_dict['track_mask'] = track_mask
            data_dict['track_i'] = track_i

        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(
                    near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(
                    choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (
                self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    # def downsample_depth_map(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
    #         return partial(self.downsample_depth_map, config=config)

    #     data_dict['depth_maps'] = transform.downscale_local_mean(
    #         image=data_dict['depth_maps'],
    #         factors=(self.depth_downsample_factor, self.depth_downsample_factor)
    #     )
    #     return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
