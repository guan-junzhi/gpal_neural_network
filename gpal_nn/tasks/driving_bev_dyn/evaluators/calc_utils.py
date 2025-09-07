import numpy as np


def calculate_projection_distances_vectorized(gt_points, dt_points, ego_pos=None):
    """
    矢量化计算投影分解距离
    
    Args:
        gt_points: GT点坐标 (N, 2) - [x_gt, y_gt]
        dt_points: DT点坐标 (M, 2) - [x_dt, y_dt]  
        ego_pos: 自车位置 (2,) - [x_ego, y_ego], 默认为[0, 0]
    
    Returns:
        d1_matrix: 纵向投影距离矩阵 (N, M)
        d2_matrix: 横向投影距离矩阵 (N, M)
        total_distance_matrix: 总投影距离矩阵 (N, M)
    """
    if ego_pos is None:
        ego_pos = np.array([0.0, 0.0])
    
    N = gt_points.shape[0]  # GT点数量
    M = dt_points.shape[0]  # DT点数量
    
    # 计算GT到自车的向量 (N, 2)
    gt_to_ego = gt_points - ego_pos[np.newaxis, :]
    
    # 计算连线长度 (N,)
    line_lengths = np.linalg.norm(gt_to_ego, axis=1)
    
    # 处理长度为0的情况(GT点与自车重合)
    valid_mask = line_lengths > 1e-8
    
    # 初始化结果矩阵
    d1_matrix = np.zeros((N, M))
    d2_matrix = np.zeros((N, M))
    
    if not np.any(valid_mask):
        total_distance_matrix = np.sqrt(d1_matrix**2 + d2_matrix**2)
        return d1_matrix, d2_matrix, total_distance_matrix
    
    # 计算单位向量(N, 2)
    # 纵向单位向量(沿连线方向)
    u_longitudinal = np.zeros_like(gt_to_ego)
    u_longitudinal[valid_mask] = gt_to_ego[valid_mask] / line_lengths[valid_mask, np.newaxis]
    
    # 横向单位向量（垂直连线方向，逆时针90度旋转）
    u_lateral = np.zeros_like(gt_to_ego)
    u_lateral[valid_mask, 0] = -u_longitudinal[valid_mask, 1]  # -sin
    u_lateral[valid_mask, 1] = u_longitudinal[valid_mask, 0]   # cos
    
    # 对于每个GT点，计算所有DT点的投影
    for i, gt_point in enumerate(gt_points):
        if not valid_mask[i]:
            continue
            
        # DT到GT的向量 (M, 2)
        dt_to_gt = dt_points - gt_point[np.newaxis, :]
        
        # 计算投影距离
        # d1: 纵向投影(沿连线方向，正值表示远离自车)
        d1_matrix[i, :] = np.dot(dt_to_gt, u_longitudinal[i])
        
        # d2: 横向投影(垂直连线方向)
        d2_matrix[i, :] = np.dot(dt_to_gt, u_lateral[i])
    
    # 计算总投影距离
    total_distance_matrix = np.sqrt(d1_matrix**2 + d2_matrix**2)
    
    return d1_matrix, d2_matrix, total_distance_matrix
        
def select_best_match(pred_i, 
                      candidate_gt_indices, 
                      d1_matrix, 
                      d2_matrix, 
                      distance_matrix, 
                      strategy='min_total_distance'):
    """
    从多个候选GT框中选择最佳匹配
    
    Args:
        pred_i: 预测框索引
        candidate_gt_indices: 候选GT框索引列表
        d1_matrix: 纵向距离矩阵 (pred_num, gt_num)
        d2_matrix: 横向距离矩阵 (pred_num, gt_num)
        distance_matrix: 总距离矩阵 (pred_num, gt_num)
        strategy: 匹配策略
    
    Returns:
        best_gt_idx: 最佳匹配的GT框索引
    """
    
    if len(candidate_gt_indices) == 1:
        return candidate_gt_indices[0]
    
    # 提取当前预测框与所有候选GT框的距离
    d1_errors = np.abs(d1_matrix[pred_i, candidate_gt_indices])
    d2_errors = np.abs(d2_matrix[pred_i, candidate_gt_indices])
    total_distances = distance_matrix[pred_i, candidate_gt_indices]
    
    if strategy == 'min_total_distance':
        # 选择总投影距离最小的GT框
        best_idx = np.argmin(total_distances)
        
    elif strategy == 'min_longitudinal':
        # 选择纵向距离最小的GT框
        best_idx = np.argmin(d1_errors)
        
    elif strategy == 'min_lateral':
        # 选择横向距离最小的GT框
        best_idx = np.argmin(d2_errors)
        
    elif strategy == 'weighted_distance':
        # 使用加权距离：纵向权重更高
        longitudinal_weight = 0.7
        lateral_weight = 0.3
        weighted_distances = longitudinal_weight * d1_errors + lateral_weight * d2_errors
        best_idx = np.argmin(weighted_distances)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return candidate_gt_indices[best_idx]