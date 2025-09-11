import torch
import numpy as np


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def cal_reference_point_from_gt_to_pred(box_dt, box_gt):
    def find_closest_corner_to_ego(box_r):
        # 找到离原点最近的角点
        distances = box_r ** 2
        distances_x2y2 = distances.sum(axis=1)
        min_index = np.argmin(distances_x2y2)
        min_distance_corner = box_r[min_index]

        return min_distance_corner, min_index
    
    corners3d_gt = boxes_to_corners_3d(box_gt).squeeze(0)
    bev_corner_gt = corners3d_gt[:4, :2]
    min_point_gt, min_index_gt = find_closest_corner_to_ego(bev_corner_gt)
    
    corners3d_dt = boxes_to_corners_3d(box_dt).squeeze(0)
    bev_corner_dt = corners3d_dt[:4, :2]
    min_point_dt, min_index_dt = find_closest_corner_to_ego(bev_corner_dt)

    if min_index_dt != min_index_gt:
        min_index_dt = min_index_gt
        min_point_dt = bev_corner_gt[min_index_gt]
    
    ref_pt_error = min_point_gt - min_point_dt
    
    return ref_pt_error

def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1, use_center_to_filter=True):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    if use_center_to_filter:
        box_centers = boxes[:, 0:3]
        mask = ((box_centers >= limit_range[0:3]) & (box_centers <= limit_range[3:6])).all(axis=-1)
    else:
        corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
        corners = corners[:, :, 0:2]
        mask = ((corners >= limit_range[0:2]) & (corners <= limit_range[3:5])).all(axis=2)
        mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask