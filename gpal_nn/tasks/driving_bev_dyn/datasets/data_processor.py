from functools import partial

import numpy as np
# from skimage import transform
from gpal_nn.tasks.driving_bev_dyn.utils.fusion_utils import compute_radius, gen_hm_radius  # !!!!
from gpal_nn.tasks.driving_bev_dyn.utils import box_utils, common_utils
import math

# from ...ops.roiaware_pool3d.roiaware_pool3d_utils import (points_in_boxes_cpu,
#                                                           points_in_boxes_gpu)

def encode_multibin(alpha):
    v = np.zeros(6)
    alpha = np.remainder(alpha, np.pi * 2)
    if alpha < np.pi / 2 + np.pi / 6.0 or alpha > np.pi / 2 * 3 - np.pi / 6:
        # if alpha < np.pi / 2 + np.pi / 6:
        #     alpha = alpha
        # else:
            # pass
        v[0] = 1
        v[2] = np.sin(alpha)
        v[3] = np.cos(alpha)
    if np.pi / 2 - np.pi / 6.0 < alpha < np.pi / 2 * 3 + np.pi / 6:
        # if alpha > np.pi:
        #     alpha = alpha - 2 * np.pi
        v[1] = 1
        v[4] = np.sin(alpha)
        v[5] = np.cos(alpha)
    return v


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

        if data_dict.get('gt_boxes', None) is not None and config['REMOVE_OUTSIDE_BOXES']:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 5))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        return data_dict

    def build_targets_track(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.build_targets_track, config=config)
        
        n_objs = config['max_objects']
        max_v  = 1
        if True:
            minX, minY, minZ, maxX, maxY, maxZ = self.point_cloud_range
            bound_size_x, bound_size_y = maxX - minX, maxY - minY

            gt_boxes       = data_dict['gt_boxes']
            num_objects    = min(gt_boxes.shape[0], n_objs)
            hm_l, hm_w     = config['hm_size']
            hm_w, hm_l     = int(hm_l), int(hm_w)
            hm_main_center = np.zeros((config['num_classes'], hm_w, hm_l), dtype=np.float32)
            cen_offset     = np.zeros((n_objs, 2), dtype=np.float32)
            direction      = np.zeros((n_objs, 2), dtype=np.float32)
            z_coor         = np.zeros((n_objs, 1), dtype=np.float32)
            dimension      = np.zeros((n_objs, 3), dtype=np.float32)
            vel            = np.zeros((n_objs, 2), dtype=np.float32)
            indices_center = np.zeros((n_objs), dtype=np.int64)
            obj_mask       = np.zeros((n_objs), dtype=np.uint8)
            # indices_track  = np.zeros((n_objs), dtype=np.int64) - 1
            multibin_direction = np.zeros((n_objs, 6), dtype=np.float32)

            for k in range(num_objects):
                x, y, z, l, w, h, yaw, track_id, vx, vy, cls_id = gt_boxes[k]  # len = 11
                # vx = np.min([vx, max_v])/max_v
                # vy = np.min([vy, max_v])/max_v
                cls_id = int(cls_id)
                if not ((minX < x < maxX) and (minY < y < maxY)):
                    continue
                if (w < 0) or (l < 0):
                    continue
                bbox_l = l / bound_size_x * hm_l
                bbox_w = w / bound_size_y * hm_w
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                radius = 0
                radius = max(0, int(radius))
                # x --> y (invert to 2D image space)
                center_x = (x - minX) / bound_size_x * hm_l
                center_y = (y - minY) / bound_size_y * hm_w  # y --> x
                center = np.array([center_x, center_y], dtype=np.float32)
                center_int = center.astype(np.int32)
                gen_hm_radius(hm_main_center[cls_id-1], center_int, radius)
                indices_center[k] = center_int[1] * hm_l + center_int[0]
                cen_offset[k] = np.array([x, y])
                dimension[k, 0] = l
                dimension[k, 1] = w
                dimension[k, 2] = h
                direction[k, 0] = math.sin(float(yaw))  # rad -> -1~1
                direction[k, 1] = math.cos(float(yaw))  # rad -> -1~1
                multibin_direction[k] = encode_multibin(float(yaw))
                z_coor[k, 0] = z
                vel[k, 0] = vx
                vel[k, 1] = vy
                obj_mask[k] = 1
                # indices_track[k] = track_id

            data_dict['gt_curr_vel'] = vel
            data_dict['gt_curr_hm_cen'] = hm_main_center
            data_dict['gt_curr_cen_offset'] = cen_offset
            data_dict['gt_curr_direction'] = direction
            data_dict['gt_curr_multibin_direction'] = multibin_direction
            data_dict['gt_curr_z_coor'] = z_coor
            data_dict['gt_curr_dim'] = dimension
            data_dict['gt_curr_indices_center'] = indices_center
            data_dict['gt_curr_obj_mask'] = obj_mask
            # data_dict['gt_curr_indice_track']   = indices_track

        return data_dict

    def forward(self, data_dict):
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict