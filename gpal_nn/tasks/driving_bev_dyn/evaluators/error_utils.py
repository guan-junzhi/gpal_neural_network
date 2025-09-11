import numpy as np


# 辅助函数（如果原代码中没有定义的话）
def center_distance(gt_center, pred_center):
    """计算中心点距离"""
    return np.linalg.norm(np.array(gt_center) - np.array(pred_center))

def scale_iou(gt_size, pred_size):
    """计算尺寸IoU"""
    gt_size = np.array(gt_size)
    pred_size = np.array(pred_size)
    
    # 计算交集和并集
    intersection = np.minimum(gt_size, pred_size)
    union = np.maximum(gt_size, pred_size)
    
    # 计算体积
    intersection_volume = np.prod(intersection)
    union_volume = np.prod(union)
    
    return intersection_volume / (union_volume + 1e-16)

def angle_diff(angle1, angle2):
    """计算角度差"""
    diff = angle1 - angle2
    # 将角度差归一化到 [-pi, pi]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def velocity_l2(pred_vel, gt_vel):
    """计算速度L2距离"""
    if len(pred_vel) == 0 or len(gt_vel) == 0:
        return 0.0
    return np.linalg.norm(np.array(pred_vel) - np.array(gt_vel))