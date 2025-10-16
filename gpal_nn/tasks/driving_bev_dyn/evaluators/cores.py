import os
import json
    
import numpy as np


from gpal_nn.tasks.driving_bev_dyn.evaluators.box_utils import (cal_reference_point_from_gt_to_pred,
                       mask_boxes_outside_range_numpy,
                       )
from gpal_nn.tasks.driving_bev_dyn.evaluators.calc_utils import (calculate_projection_distances_vectorized,
                        select_best_match,
                        )
from gpal_nn.tasks.driving_bev_dyn.evaluators.error_utils import (center_distance,
                         scale_iou,
                         angle_diff,
                         velocity_l2,
                         )
from tqdm import tqdm

def get_one_sample_statistics_rotated_3d_boxes_distance(
    # 输入
    pred_boxes:np.ndarray, 
    pred_scores:np.ndarray, 
    pred_labels:np.ndarray, 
    gt_boxes:np.ndarray,
    
    # 循环输出
    distance_errors_list:list, 
    
    # 控制参数
    use_theory_to_mask_range:str = 'cuboid',
    det_range_list:list = None,
    
    use_projection:bool = True,
    ego_pos:np.ndarray = None,
    
    distance_threshold_list:list = None,
    restricted_ratio:list = [0.05, 0.005],
    best_match_strategy: str = 'min_total_distance',  # 'min_total_distance', 'min_longitudinal', 'min_lateral', 'weighted_distance'
    frame_idx=None,
    
    is_record_bad_cases: bool = True,  # 新增参数
    distance_threshold_ratio: float = 0.8,  # 用于定义hard case的阈值比例
    
    loggerinfo=None,
    is_print_during_info: bool = False,
    
    class_names:list = None,
    
    *args,
    **kwargs,
    ):

    assert len(restricted_ratio) == 2, (
        f"restricted_ratio should be a list of length 2, got {len(restricted_ratio)}")
    assert len(det_range_list) >= 1 and all([len(i) == 6 for i in det_range_list]), (
        f"det_range_list should be a list of length >= 1, and each element should be a list of length 6, got {len(det_range_list)}")
    
    assert len(distance_threshold_list) >= 1 and all([isinstance(i, (int, float)) for i in distance_threshold_list]), (
        f"distance_threshold_list should be a list of length >= 1, and each element should be a number, got {len(distance_threshold_list)}")
    
    
    longitudinal_ratio = restricted_ratio[0] # 径向
    lateral_ratio = restricted_ratio[1]      # 横向
    fixed_threshold = distance_threshold_list[0] if distance_threshold_list else None
    
    batch_metrics = []
    true_positives = np.zeros((pred_boxes.shape[0])).astype(np.float32)
    
    # 在匹配循环开始前记录所有GT框信息
    gt_matched_status = np.zeros(gt_boxes.shape[0], dtype=bool)  # 追踪GT匹配状态
    
    # ================ 距离页面标记 ================
    # 分距离页面是因为人为的会写入有区域重叠的情况
    range_mask_dt = np.ones(shape=((pred_boxes.shape[0]), len(det_range_list))).astype(np.float32) * -1
    range_mask_gt = np.ones(shape=((gt_boxes.shape[0]), len(det_range_list))).astype(np.float32) * -1

    # === 新增：初始化帧级别的bad case记录 ===
    frame_bad_cases = {
        'frame_idx': frame_idx,
        'pred_empty': pred_boxes.shape[0] == 0,
        'gt_empty': gt_boxes.shape[0] == 0,
        'total_pred_count': pred_boxes.shape[0],
        'total_gt_count': gt_boxes.shape[0],
        'fn_cases': [],
        'fp_cases': [],
        'hard_cases': []
    }

    # === 边界检查 ===
    pred_empty = pred_boxes.shape[0] == 0
    gt_empty = gt_boxes.shape[0] == 0
    # breakpoint()
    if pred_empty and gt_empty:
        # 情况1: 预测和GT都为空
        loggerinfo(f'{frame_idx} Both pred_boxes and gt_boxes are empty')
        pred_scores = np.zeros((0,)).astype(np.float32)
        pred_labels = np.zeros((0,)).astype(np.float32)
        true_positives = np.zeros((0,)).astype(np.bool_)
        range_mask_dt = np.full((0, len(det_range_list)), -1, dtype=np.int32)
        range_mask_gt = np.full((0, len(det_range_list)), -1, dtype=np.int32)
        
        # === 记录帧级别的empty case ===
        if is_record_bad_cases:
            frame_bad_cases['case_type'] = 'both_empty'

            record_frame_bad_case(frame_bad_cases, distance_errors_list)

        if is_print_during_info:
            print_frame_statistics(frame_idx, gt_boxes, pred_boxes, pred_labels, np.array([]), 
                                   class_names, {}, loggerinfo)
        
        batch_metrics.append([true_positives, pred_scores, pred_labels, range_mask_dt, range_mask_gt])
        return batch_metrics, distance_errors_list

    elif pred_empty and not gt_empty:
        # 情况2: 预测为空，GT不为空
        loggerinfo(f'{frame_idx} pred_boxes is empty, gt_boxes has {gt_boxes.shape[0]} boxes')
        pred_scores = np.zeros((0,)).astype(np.float32)
        pred_labels = np.zeros((0,)).astype(np.float32)
        true_positives = np.zeros((0,)).astype(np.bool_)
        range_mask_dt = np.full((0, len(det_range_list)), -1, dtype=np.int32)
        
        # 处理GT的range mask
        range_mask_gt = np.full((gt_boxes.shape[0], len(det_range_list)), -1, dtype=np.int32)
        for range_i, curr_range in enumerate(det_range_list):
            if use_theory_to_mask_range == 'cuboid':
                curr_range_mask_gt = mask_boxes_outside_range_numpy(gt_boxes, curr_range, min_num_corners=1, use_center_to_filter=True)
                range_mask_gt[curr_range_mask_gt, range_i] = range_i
            else:
                raise NotImplementedError(f'Not implemented for {use_theory_to_mask_range}')
        
        # === 记录所有GT框为FN cases ===
        if is_record_bad_cases:
            frame_bad_cases['case_type'] = 'pred_empty'
            gt_label_ids = gt_boxes[:, -1].astype(np.int32)
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                fn_case = {
                    'gt_idx': gt_idx,
                    'gt_box': gt_box.copy(),
                    'gt_label': gt_label_ids[gt_idx],
                    'reason': 'pred_empty',
                    'ego_distance': np.linalg.norm(gt_box[:2] - ego_pos) if ego_pos is not None else None,
                }
                
                cls_seq = np.int_(gt_label_ids[gt_idx]) - 1
                if 0 <= cls_seq < len(distance_errors_list):
                    record_bad_case('fn', fn_case, distance_errors_list, cls_seq)
                    
                frame_bad_cases['fn_cases'].append(fn_case)
            
            record_frame_bad_case(frame_bad_cases, distance_errors_list)

        if is_print_during_info:
            print_frame_statistics(frame_idx, gt_boxes, pred_boxes, pred_labels, gt_label_ids, 
                                   class_names, {}, loggerinfo)
                    
        batch_metrics.append([true_positives, pred_scores, pred_labels, range_mask_dt, range_mask_gt])
        return batch_metrics, distance_errors_list

    elif not pred_empty and gt_empty:
        # 情况3: 预测不为空，GT为空
        loggerinfo(f'{frame_idx} gt_boxes is empty, pred_boxes has {pred_boxes.shape[0]} boxes')
        true_positives = np.zeros((pred_boxes.shape[0],)).astype(np.bool_)  # 全部为FP
        range_mask_gt = np.full((0, len(det_range_list)), -1, dtype=np.int32)
        
        # 处理预测框的range mask
        range_mask_dt = np.full((pred_boxes.shape[0], len(det_range_list)), -1, dtype=np.int32)
        for range_i, curr_range in enumerate(det_range_list):
            if use_theory_to_mask_range == 'cuboid':
                curr_range_mask_dt = mask_boxes_outside_range_numpy(pred_boxes, curr_range, min_num_corners=1, use_center_to_filter=True)
                range_mask_dt[curr_range_mask_dt, range_i] = range_i
            else:
                raise NotImplementedError(f'Not implemented for {use_theory_to_mask_range}')
        
        # === 记录所有预测框为FP cases ===
        if is_record_bad_cases:
            frame_bad_cases['case_type'] = 'gt_empty'
            
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            pred_label_ids = pred_labels[sort_idx].astype(np.int32)
            
            for pred_i, pred_box in enumerate(pred_boxes):
                fp_case = {
                    'pred_idx': pred_i,
                    'pred_box': pred_box.copy(),
                    'pred_label': pred_label_ids[pred_i],
                    'pred_score': pred_scores[pred_i],
                    'reason': 'gt_empty',
                    'ego_distance': np.linalg.norm(pred_box[:2] - ego_pos) if ego_pos is not None else None,
                }
                
                cls_seq = np.int_(pred_label_ids[pred_i]) - 1
                if 0 <= cls_seq < len(distance_errors_list):
                    record_bad_case('fp', fp_case, distance_errors_list, cls_seq)
                
                frame_bad_cases['fp_cases'].append(fp_case)
            
            record_frame_bad_case(frame_bad_cases, distance_errors_list)
        
        if is_print_during_info:
            print_frame_statistics(frame_idx, gt_boxes, pred_boxes, pred_labels, np.array([]), 
                                   class_names, {}, loggerinfo)
            
        batch_metrics.append([true_positives, pred_scores, pred_labels, range_mask_dt, range_mask_gt])
        return batch_metrics, distance_errors_list
    # === 边界检查结束 ===
    else:
        # 情况4: 预测和GT都不为空，继续正常匹配逻辑
        frame_bad_cases['case_type'] = 'normal_matching'
        
        for range_i, curr_range in enumerate(det_range_list):
            if use_theory_to_mask_range == 'cuboid':
                curr_range_mask_dt = mask_boxes_outside_range_numpy(pred_boxes, curr_range, min_num_corners=1, use_center_to_filter=True)
                range_mask_dt[curr_range_mask_dt, range_i] = range_i
                
                curr_range_mask_gt = mask_boxes_outside_range_numpy(gt_boxes, curr_range, min_num_corners=1, use_center_to_filter=True)
                range_mask_gt[curr_range_mask_gt, range_i] = range_i
            else:
                raise NotImplementedError(f'Not implemented for {use_theory_to_mask_range}')
    # ================ 距离页面标记 ================
    
    
    gt_label_ids = gt_boxes[:, -1].astype(np.int32)
    
    # === 降序排序
    sort_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sort_idx]
    pred_scores = pred_scores[sort_idx]
    pred_label_ids = pred_labels[sort_idx].astype(np.int32)
    range_mask_dt = range_mask_dt[sort_idx]
    
    range_mask_dt_ori = range_mask_dt.copy()
    
    # 初始化以gt类别的matched_gt_boxes集合
    matched_gt_boxes_per_class = {}
    unique_label_ids = np.unique(gt_label_ids)
    for label_id in unique_label_ids:
        matched_gt_boxes_per_class[label_id] = set()
    
    # 计算距离矩阵
    if use_projection and ego_pos is not None:
        # 使用投影分解距离
        gt_centers_2d = gt_boxes[:, :2]
        pred_centers_2d = pred_boxes[:, :2]
        d1_matrix, d2_matrix, total_distance_matrix = calculate_projection_distances_vectorized(
            gt_centers_2d, pred_centers_2d, ego_pos
        )
        # 注意：投影距离矩阵的维度是 (gt_num, pred_num)，需转置 -> 转置为 (pred_num, gt_num)
        d1_matrix = d1_matrix.T
        d2_matrix = d2_matrix.T
        distance_matrix = total_distance_matrix.T
        
        gt_to_ego_distances = np.linalg.norm(gt_centers_2d - ego_pos[np.newaxis, :], axis=1)  # 用于动态阈值gt (gt_num,)
        
        # d1_error_threshold_gt = np.ones_like(abs(gt_to_ego_distances)) * fixed_threshold
        # d2_error_threshold_gt = np.ones_like(abs(gt_to_ego_distances))  * fixed_threshold
        d1_error_threshold_gt = np.maximum(abs(gt_to_ego_distances) * longitudinal_ratio, fixed_threshold)
        d2_error_threshold_gt = np.maximum(abs(gt_to_ego_distances) * lateral_ratio, fixed_threshold)     
        
    else:
        raise NotImplementedError(f'Not other distance metrics implemented for use_projection: {use_projection}')
    
    
    # 临时 | (配合单帧统计信息)|统计每个类别的预测框数量
    # pred_counts_per_class = {}
    # for gt_label_id in unique_label_ids:
    #     pred_counts_per_class[gt_label_id] = np.sum(pred_label_ids == gt_label_id)
    best_gt_idx_list = []
    # ====== 目的: 十分精准的匹配 TP ======
    for pred_i, (pred_box, pred_label_id) in enumerate(zip(pred_boxes, pred_label_ids)):
        matched = False 
        
        class_mask = gt_label_ids == pred_label_id  # 只观察在gt中是否出现
        if (not np.any(class_mask)):
            
            # 记录FP case - 类别不匹配
            if is_record_bad_cases:
                fp_case = {
                    'frame_idx': frame_idx,
                    'pred_idx': pred_i,
                    # 'all_gt_box': gt_boxes.copy(),
                    'pred_box': pred_box.copy(),
                    'pred_label': pred_label_id,
                    'pred_score': pred_scores[pred_i],
                    'reason': 'no_gt_class_match',
                    'available_gt_classes': np.unique(gt_label_ids).tolist(),
                    'ego_distance': np.linalg.norm(pred_box[:2] - ego_pos),
                }
                cls_seq = np.int_(pred_label_id) - 1
                if 0 <= cls_seq < len(distance_errors_list):
                    record_bad_case('fp', fp_case, distance_errors_list, cls_seq)
                
                frame_bad_cases['fp_cases'].append(fp_case)

            continue

        # ==== 核心匹配逻辑 ====
        # 1. 使用动态距离阈值进行匹配
        # 2. 需要控制类别正确
        # 3. 是否已经被匹配过了
        
        # 1. 距离判断 - 找到所有可能的候选GT框
        if use_projection and ego_pos is not None:
            d1_error_one_pred_and_mul_gt = abs(d1_matrix[pred_i, :])  # 径向误差
            d2_error_one_pred_and_mul_gt = abs(d2_matrix[pred_i, :])  # 横向误差
            
            longitudinal_flag = d1_error_one_pred_and_mul_gt <= d1_error_threshold_gt
            lateral_flag = d2_error_one_pred_and_mul_gt <= d2_error_threshold_gt
            distance_gt_mask_flag = longitudinal_flag & lateral_flag
        else:
            raise NotImplementedError(f'Not other distance metrics implemented for use_projection: {use_projection}')
        
        if not np.any(distance_gt_mask_flag):
            
            # 记录FP case - 距离阈值不满足
            if is_record_bad_cases:
                # 找到最近的同类GT框
                same_class_gt_indices = np.where(class_mask)[0]
                if len(same_class_gt_indices) > 0:
                    distances_to_same_class = distance_matrix[pred_i, same_class_gt_indices]
                    closest_gt_idx = same_class_gt_indices[np.argmin(distances_to_same_class)]
                    
                    fp_case = {
                        'frame_idx': frame_idx,
                        'pred_idx': pred_i,
                        'pred_box': pred_box.copy(),
                        # 'all_gt_box': gt_boxes.copy(),
                        'pred_label': pred_label_id,
                        'pred_score': pred_scores[pred_i],
                        'reason': 'distance_threshold_exceeded',
                        'closest_gt_idx': closest_gt_idx,
                        'closest_gt_box': gt_boxes[closest_gt_idx].copy(),
                        'closest_distance': distances_to_same_class[np.argmin(distances_to_same_class)],
                        'longitudinal_error': d1_error_one_pred_and_mul_gt[closest_gt_idx],
                        'lateral_error': d2_error_one_pred_and_mul_gt[closest_gt_idx],
                        'longitudinal_threshold': d1_error_threshold_gt[closest_gt_idx],
                        'lateral_threshold': d2_error_threshold_gt[closest_gt_idx],
                        'ego_distance': np.linalg.norm(pred_box[:2] - ego_pos),
                    }
                    cls_seq = np.int_(pred_label_id) - 1
                    if 0 <= cls_seq < len(distance_errors_list):
                        record_bad_case('fp', fp_case, distance_errors_list, cls_seq)
                    
                    frame_bad_cases['fp_cases'].append(fp_case)
                    
            continue
        
        # 2. 找到类别正确且距离满足条件的候选GT框
        candidate_gt_indices = np.where(distance_gt_mask_flag & class_mask)[0] # 可能多个? 如何才能最佳？
        if len(candidate_gt_indices) == 0:
            
            if is_record_bad_cases:
                distances_to_gt_class = distance_matrix[pred_i, :]
                closest_gt_idx = np.argmin(distances_to_gt_class)
                
                fp_case = {
                    'frame_idx': frame_idx,
                    'pred_idx': pred_i,
                    'pred_box': pred_box.copy(),
                    # 'all_gt_box': gt_boxes.copy(),
                    'pred_label': pred_label_id,
                    'pred_score': pred_scores[pred_i],
                    'reason': 'pred_no_matched_class',
                    'closest_gt_idx': closest_gt_idx,
                    'closest_gt_box': gt_boxes[closest_gt_idx].copy(),
                    'closest_distance': distance_matrix[pred_i, closest_gt_idx],
                    'longitudinal_error': d1_error_one_pred_and_mul_gt[closest_gt_idx],
                    'lateral_error': d2_error_one_pred_and_mul_gt[closest_gt_idx],
                    'longitudinal_threshold': d1_error_threshold_gt[closest_gt_idx],
                    'lateral_threshold': d2_error_threshold_gt[closest_gt_idx],
                    'ego_distance': np.linalg.norm(pred_box[:2] - ego_pos),
                }
                cls_seq = np.int_(pred_label_id) - 1
                if 0 <= cls_seq < len(distance_errors_list):
                    record_bad_case('fp', fp_case, distance_errors_list, cls_seq)
                
                frame_bad_cases['fp_cases'].append(fp_case)
            
            continue
        
        # 3. 过滤掉已经被匹配的GT框
        unmatched_candidates = []
        for gt_idx in candidate_gt_indices:
            if gt_idx not in matched_gt_boxes_per_class[pred_label_id]:
                unmatched_candidates.append(gt_idx)

        if len(unmatched_candidates) == 0:
            # 记录FP case - 所有候选GT都已被匹配
            if is_record_bad_cases:
                fp_case = {
                    'frame_idx': frame_idx,
                    'pred_idx': pred_i,
                    'pred_box': pred_box.copy(),
                    # 'all_gt_box': gt_boxes.copy(),
                    'pred_label': pred_label_id,
                    'pred_score': pred_scores[pred_i],
                    'reason': 'all_candidates_already_matched',
                    'candidate_gt_indices': candidate_gt_indices.tolist(),
                    'ego_distance': np.linalg.norm(pred_box[:2] - ego_pos),
                }
                cls_seq = np.int_(pred_label_id) - 1
                if 0 <= cls_seq < len(distance_errors_list):
                    record_bad_case('fp', fp_case, distance_errors_list, cls_seq)
            
                frame_bad_cases['fp_cases'].append(fp_case)
            
            continue
        
        # 4. 从多个候选GT框中选择最佳匹配
        best_gt_idx = select_best_match(
            pred_i, 
            unmatched_candidates, 
            d1_matrix, 
            d2_matrix, 
            distance_matrix,
            strategy=best_match_strategy
        )
        
        if best_gt_idx is None:
            continue
        
        if best_gt_idx is not None:
            # 记录匹配状态
            # 5. 执行匹配
            matched = True
            gt_matched_status[best_gt_idx] = True
            matched_gt_boxes_per_class[pred_label_id].add(best_gt_idx)
            true_positives[pred_i] = 1
            
            # 检查是否为hard case（接近阈值边界）
            if is_record_bad_cases:
                d1_error = d1_error_one_pred_and_mul_gt[best_gt_idx]
                d2_error = d2_error_one_pred_and_mul_gt[best_gt_idx]
                d1_threshold = d1_error_threshold_gt[best_gt_idx]
                d2_threshold = d2_error_threshold_gt[best_gt_idx]
                
                # 定义hard case: 距离误差超过阈值的一定比例
                is_hard_longitudinal = d1_error > d1_threshold * distance_threshold_ratio
                is_hard_lateral = d2_error > d2_threshold * distance_threshold_ratio
                
                if is_hard_longitudinal or is_hard_lateral:
                    hard_case = {
                        'frame_idx': frame_idx,
                        'pred_idx': pred_i,
                        'gt_idx': best_gt_idx,
                        'pred_box': pred_box.copy(),
                        # 'all_gt_box': gt_boxes.copy(),
                        'gt_box': gt_boxes[best_gt_idx].copy(),
                        'pred_label': pred_label_id,
                        'pred_score': pred_scores[pred_i],
                        'longitudinal_error': d1_error,
                        'lateral_error': d2_error,
                        'longitudinal_threshold': d1_threshold,
                        'lateral_threshold': d2_threshold,
                        'longitudinal_ratio': d1_error / d1_threshold,
                        'lateral_ratio': d2_error / d2_threshold,
                        'total_distance': distance_matrix[pred_i, best_gt_idx],
                        'ego_distance': np.linalg.norm(pred_box[:2] - ego_pos),
                        'match_strategy': best_match_strategy,
                    }
                    cls_seq = np.int_(pred_label_id) - 1
                    if 0 <= cls_seq < len(distance_errors_list):
                        record_bad_case('hard', hard_case, distance_errors_list, cls_seq)
        
                    frame_bad_cases['hard_cases'].append(hard_case)
        
        # 记录匹配信息
        gt_box = gt_boxes[best_gt_idx]
        
        # loggerinfo(f"Pred[{pred_i}] -> GT[{best_gt_idx}]: ego_dist={gt_to_ego_distances[best_gt_idx]:.2f}m, "
        #       f"d1={d1_error_one_pred_and_mul_gt[best_gt_idx]:.3f}m(thr={d1_error_threshold_gt[best_gt_idx]:.3f}m), "
        #       f"d2={d2_error_one_pred_and_mul_gt[best_gt_idx]:.3f}m(thr={d2_error_threshold_gt[best_gt_idx]:.3f}m), "
        #       f"strategy={best_match_strategy} "
        #       f"pred_box={pred_box}, gt_box={gt_box}"
        #       )
        best_gt_idx_list.append(best_gt_idx)
        # [计算]各维度误差并[记录]
        class_flag = True
        unmatched_flag = True
        if class_flag and unmatched_flag:
            # 匹配上了距离页面保持一致性, 防止边界的错乱, 保证TP的一致性
            range_mask_dt[pred_i, :] = range_mask_gt[best_gt_idx, ]
            
            # 计算各维度误差
            dx = pred_box[0] - gt_box[0]
            dy = pred_box[1] - gt_box[1]
            dz = pred_box[2] - gt_box[2]
            dl = pred_box[3] - gt_box[3]
            dw = pred_box[4] - gt_box[4]
            dh = pred_box[5] - gt_box[5]
            dr = pred_box[6] - gt_box[6]
            
            # 计算特定误差指标（需要确保这些函数已定义）
            t_e = center_distance(gt_box[:2], pred_box[:2])  # ATE  xy
            s_e = 1 - scale_iou(gt_box[3:6], pred_box[3:6])  # ASE  lwh 0~1
            o_e = angle_diff(gt_box[6], pred_box[6])         # AOE  rot abs -> 0~pi  
            o_e = abs(o_e)
            v_e = velocity_l2(pred_box[7:9], gt_box[8:10]) if pred_box.shape[0] > 7 and gt_box.shape[0] > 7 else 0.0  # AVE  v

            dv = v_e
            
            ref_pt_error = cal_reference_point_from_gt_to_pred(box_gt=gt_box.reshape(1, -1), 
                                                               box_dt=pred_box.reshape(1, -1))
            
            # 记录信息
            cls_seq = np.int_(pred_label_id) - 1
            distance_errors_list[cls_seq]['ref_x_err'].append(ref_pt_error[0])
            distance_errors_list[cls_seq]['ref_y_err'].append(ref_pt_error[1])
            
            distance_errors_list[cls_seq]['x'].append(dx)
            distance_errors_list[cls_seq]['y'].append(dy)
            distance_errors_list[cls_seq]['z'].append(dz)
            distance_errors_list[cls_seq]['l'].append(dl)
            distance_errors_list[cls_seq]['w'].append(dw)
            distance_errors_list[cls_seq]['h'].append(dh)
            distance_errors_list[cls_seq]['r'].append(dr)
            distance_errors_list[cls_seq]['v'].append(dv)
            distance_errors_list[cls_seq]['trans_err'].append(t_e)
            distance_errors_list[cls_seq]['vel_err'].append(v_e)
            distance_errors_list[cls_seq]['scale_err'].append(s_e)
            distance_errors_list[cls_seq]['orient_err'].append(o_e)
            distance_errors_list[cls_seq]['range'].append(range_mask_gt[[best_gt_idx], :])  # 从于gt,保存距离页面的信息

            
            # 添加动态阈值信息记录
            # if 'longitudinal_threshold' not in distance_errors_list[cls_seq]:
            #     distance_errors_list[cls_seq]['longitudinal_threshold'] = []
            # distance_errors_list[cls_seq]['longitudinal_threshold'].append(longitudinal_threshold)
    
    # loggerinfo(range_mask_dt_ori==range_mask_dt)
    # loggerinfo(range_mask_dt)
    # 临时|打印单帧详细的统计信息
        
    if is_record_bad_cases:
        for gt_idx, gt_box in enumerate(gt_boxes):
            if not gt_matched_status[gt_idx]:
                # 未匹配的GT框 - FN case
                gt_label = gt_label_ids[gt_idx]
                
                # 找到最近的同类预测框
                same_class_pred_mask = pred_label_ids == gt_label
                if np.any(same_class_pred_mask):
                    same_class_pred_indices = np.where(same_class_pred_mask)[0]
                    distances_to_same_class = distance_matrix[same_class_pred_indices, gt_idx]
                    closest_pred_idx = same_class_pred_indices[np.argmin(distances_to_same_class)]
                    
                    fn_case = {
                        'frame_idx': frame_idx,
                        'gt_idx': gt_idx,
                        # 'all_gt_box': gt_boxes.copy(),
                        'gt_box': gt_box.copy(),
                        'gt_label': gt_label,
                        'reason': 'no_pred_match',
                        'closest_pred_idx': closest_pred_idx,
                        'closest_pred_box': pred_boxes[closest_pred_idx].copy(),
                        'closest_pred_score': pred_scores[closest_pred_idx],
                        'closest_distance': distances_to_same_class[np.argmin(distances_to_same_class)],
                        'ego_distance': np.linalg.norm(gt_box[:2] - ego_pos),
                    }
                else:
                    fn_case = {
                        'frame_idx': frame_idx,
                        'gt_idx': gt_idx,
                        # 'all_gt_box': gt_boxes.copy(),
                        'gt_box': gt_box.copy(),
                        'gt_label': gt_label,
                        'reason': 'no_pred_same_class',
                        'ego_distance': np.linalg.norm(gt_box[:2] - ego_pos),
                    }
                
                cls_seq = np.int_(gt_label) - 1
                if 0 <= cls_seq < len(distance_errors_list):
                    record_bad_case('fn', fn_case, distance_errors_list, cls_seq)
            
                frame_bad_cases['fn_cases'].append(fn_case)
        
        # === 记录整帧的bad case信息 ===
        record_frame_bad_case(frame_bad_cases, distance_errors_list)    
    
    if is_print_during_info:
        print_frame_statistics(frame_idx, 
                                gt_boxes, 
                                pred_boxes, 
                                pred_label_ids, 
                                gt_label_ids, 
                                class_names, 
                                matched_gt_boxes_per_class, 
                                loggerinfo)
        
        # print_frame_statistics_from_bad_cases(frame_idx, gt_boxes, pred_boxes, pred_label_ids, gt_label_ids, 
        #                                       class_names, matched_gt_boxes_per_class=matched_gt_boxes_per_class, 
        #                                       frame_bad_cases=frame_bad_cases, loggerinfo=loggerinfo)
        
        # debug_result = debug_fp_mismatch(
        #     frame_idx=frame_idx,
        #     pred_labels=pred_labels,
        #     matched_gt_boxes_per_class=matched_gt_boxes_per_class,
        #     frame_bad_cases=frame_bad_cases,  # 如果有的话
        #     class_id=2  # 
        # )
        
        # print_frame_statistics_debug(frame_idx, gt_boxes, pred_boxes, pred_label_ids, gt_label_ids, 
        #                         class_names, matched_gt_boxes_per_class=matched_gt_boxes_per_class, 
        #                         frame_bad_cases=frame_bad_cases, loggerinfo=loggerinfo)
        
        
    batch_metrics.append([true_positives, pred_scores, pred_label_ids, range_mask_dt, range_mask_gt,])
    return batch_metrics, distance_errors_list

def record_bad_case(case_type, case_info, distance_errors_list, cls_idx):
    """
    记录bad case信息
    
    Args:
        case_type: 'fn', 'fp', 'hard'
        case_info: dict containing case details
        distance_errors_list: 距离误差列表
        cls_idx: 类别索引
    """
    if case_type == 'fn':
        distance_errors_list[cls_idx]['bad_cases']['fn_cases'].append(case_info)
    elif case_type == 'fp':
        distance_errors_list[cls_idx]['bad_cases']['fp_cases'].append(case_info)
    elif case_type == 'hard':
        distance_errors_list[cls_idx]['bad_cases']['hard_cases'].append(case_info)
    else:
        raise ValueError(f"Unknown case type: {case_type}")
    
def print_bad_cases_analysis(class_names, distance_errors_list, loggerinfo=print):
    """
    打印bad case分析结果
    """
    loggerinfo(f'============= 全区域 Bad Cases分析 =============')
    
    for cls_idx, cls_name in enumerate(class_names):
        bad_cases = distance_errors_list[cls_idx]['bad_cases']
        
        fn_cases = bad_cases['fn_cases']
        fp_cases = bad_cases['fp_cases'] 
        hard_cases = bad_cases['hard_cases']
        
        loggerinfo(f"=== {cls_name} 类别 Bad Cases ===")
        loggerinfo(f" FN Cases: {len(fn_cases):^8} | FP Cases: {len(fp_cases):^8} | Hard Cases: {len(hard_cases)}")
        
        # FN Cases 分析s
        if fn_cases:
            fn_reasons = {}
            fn_distances = []
            for case in fn_cases:
                reason = case['reason']
                fn_reasons[reason] = fn_reasons.get(reason, 0) + 1
                fn_distances.append(case['ego_distance'])
            
            loggerinfo(" --- FN Cases 分析:")
            for reason, count in fn_reasons.items():
                loggerinfo(f"  {reason}: {count}")
            
            if fn_distances:
                loggerinfo(f"  FN距离分布: mean={np.mean(fn_distances):.2f}m, "
                      f"std={np.std(fn_distances):.2f}m, "
                      f"min={np.min(fn_distances):.2f}m, "
                      f"max={np.max(fn_distances):.2f}m")
        
        # FP Cases 分析
        if fp_cases:
            fp_reasons = {}
            fp_distances = []
            for case in fp_cases:
                reason = case['reason']
                fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
                fp_distances.append(case['ego_distance'])
            
            loggerinfo(" --- FP Cases 分析:")
            for reason, count in fp_reasons.items():
                loggerinfo(f"  {reason}: {count}")
            
            if fp_distances:
                loggerinfo(f"  FP距离分布: mean={np.mean(fp_distances):.2f}m, "
                      f"std={np.std(fp_distances):.2f}m, "
                      f"min={np.min(fp_distances):.2f}m, "
                      f"max={np.max(fp_distances):.2f}m")
        
        # Hard Cases 分析
        if hard_cases:
            loggerinfo(f" --- Hard Cases 分析:")
            hard_long_ratios = [case['longitudinal_ratio'] for case in hard_cases]
            hard_lat_ratios = [case['lateral_ratio'] for case in hard_cases]
            hard_distances = [case['ego_distance'] for case in hard_cases]
            
            loggerinfo(f"  Hard Cases距离分布: mean={np.mean(hard_distances):.2f}m")
            loggerinfo(f"  纵向阈值使用率: mean={np.mean(hard_long_ratios):.3f}, max={np.max(hard_long_ratios):.3f}")
            loggerinfo(f"  横向阈值使用率: mean={np.mean(hard_lat_ratios):.3f}, max={np.max(hard_lat_ratios):.3f}")
            
        loggerinfo(f"="*40)
        

def export_bad_cases_to_file(class_names, distance_errors_list, output_dir="./bad_cases", loggerinfo=print):
    """
    将bad cases导出到文件
    """
    for cls_idx, cls_name in enumerate(class_names):
        bad_cases = distance_errors_list[cls_idx]['bad_cases']
        
        # 转换numpy数组和numpy标量为可序列化类型
        def convert_numpy_to_list(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        bad_cases_serializable = convert_numpy_to_list(bad_cases)
        
        output_file = os.path.join(output_dir, f"{cls_name}_bad_cases.json")
        with open(output_file, 'w') as f:
            json.dump(bad_cases_serializable, f, indent=2)
        
        loggerinfo(f"Bad cases for {cls_name} saved to => {output_file}")

def export_frames_bad_cases_to_file(class_names, distance_errors_list, output_dir="./bad_cases", loggerinfo=print):
    """
    将整帧的bad cases导出到文件
    """
    def convert_numpy_to_list(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    bad_cases_frame = distance_errors_list[-1]['frame_statistics']
    bad_cases_frame_serializable = convert_numpy_to_list(bad_cases_frame)
    
    output_file = os.path.join(output_dir, f"frame_statistics_bad_cases.json")
    with open(output_file, 'w') as f:
        json.dump(bad_cases_frame_serializable, f, indent=2)
    
    loggerinfo(f"Bad cases for frame_statistics saved to => {output_file}")


def record_frame_bad_case(frame_bad_cases, distance_errors_list):
    """
    记录整帧的bad case信息到frame_statistics中，以帧为单位统一管理
    
    Args:
        frame_bad_cases: 帧级别的bad case字典
        distance_errors_list: 距离误差列表，最后一个元素为frame_statistics
    """
    # 获取frame_statistics存储位置 (最后一个元素)
    frame_stats_container = distance_errors_list[-1]['frame_statistics']
    
    frame_data = {
        # 在上层函数中一开始就传入了具体的数量
        'frame_idx': frame_bad_cases['frame_idx'],
        'frame_case_type': frame_bad_cases['case_type'],
        'pred_empty': frame_bad_cases['pred_empty'],
        'gt_empty': frame_bad_cases['gt_empty'],
        'total_pred_count': frame_bad_cases['total_pred_count'],
        'total_gt_count': frame_bad_cases['total_gt_count'],
        'summary': {
            'total_fn_count': len(frame_bad_cases['fn_cases']),
            'total_fp_count': len(frame_bad_cases['fp_cases']),
            'total_hard_count': len(frame_bad_cases['hard_cases']),
            'classes_count': 0,
            'involved_classes': [],
        },
        'classes_data': {},  # 按类别组织的数据
    }
    
    # 收集所有涉及的类别
    all_classes = set()
    
    # 从FN cases中收集类别
    for fn_case in frame_bad_cases['fn_cases']:
        gt_label = int(fn_case['gt_label'])
        all_classes.add(gt_label)
    
    # 从FP cases中收集类别
    for fp_case in frame_bad_cases['fp_cases']:
        pred_label = int(fp_case['pred_label'])
        all_classes.add(pred_label)
    
    # 从Hard cases中收集类别
    for hard_case in frame_bad_cases['hard_cases']:
        pred_label = int(hard_case['pred_label'])
        all_classes.add(pred_label)
    
    # 为每个类别初始化数据结构
    for class_label in all_classes:
        frame_data['classes_data'][str(class_label)] = {
            'class_label': class_label,
            'fn_cases': [],
            'fp_cases': [], 
            'hard_cases': [],
            'fn_count': 0,
            'fp_count': 0,
            'hard_count': 0,
        }
    
    # 整理FN cases
    for fn_case in frame_bad_cases['fn_cases']:
        gt_label = fn_case['gt_label']
        str_label = str(gt_label)
        frame_data['classes_data'][str_label]['fn_cases'].append(fn_case)
        frame_data['classes_data'][str_label]['fn_count'] += 1
    
    # 整理FP cases
    for fp_case in frame_bad_cases['fp_cases']:
        pred_label = int(fp_case['pred_label'])
        str_label = str(pred_label)
        frame_data['classes_data'][str_label]['fp_cases'].append(fp_case)
        frame_data['classes_data'][str_label]['fp_count'] += 1

    # 整理Hard cases
    for hard_case in frame_bad_cases['hard_cases']:
        pred_label = int(hard_case['pred_label'])
        str_label = str(pred_label)

        frame_data['classes_data'][str_label]['hard_cases'].append(hard_case)
        frame_data['classes_data'][str_label]['hard_count'] += 1

    
    # 计算帧级别汇总统计
    # 更新summary信息
    
    for class_label in all_classes:
        frame_data['summary'][str(class_label)] = {
            # "fn_cases": frame_data['classes_data'][str(class_label)]['fn_cases'],
            # "fp_cases": frame_data['classes_data'][str(class_label)]['fp_cases'],
            # "hard_cases": frame_data['classes_data'][str(class_label)]['hard_cases'],
            "fn_count": frame_data['classes_data'][str(class_label)]['fn_count'],
            "fp_count": frame_data['classes_data'][str(class_label)]['fp_count'],
            "hard_count": frame_data['classes_data'][str(class_label)]['hard_count'],
        }
    
    frame_data['summary']['involved_classes'] = list(all_classes)
    frame_data['summary']['classes_count'] = len(list(all_classes))
    
    # 添加到frame_statistics
    frame_stats_container.append(frame_data)


def print_frame_statistics(frame_idx, gt_boxes, pred_boxes, pred_labels, gt_label_ids, 
                           class_names, matched_gt_boxes_per_class=None, loggerinfo=None):
    """
    统一打印帧统计信息的函数
    Args:
        frame_idx: 帧索引
        gt_boxes: GT框数组
        pred_boxes: 预测框数组  
        pred_labels: 预测标签数组
        gt_label_ids: GT标签ID数组
        class_names: 类别名称列表
        matched_gt_boxes_per_class: 每个类别的匹配GT框集合字典
        loggerinfo: 日志函数
    """
    if loggerinfo is None:
        loggerinfo = print
        
    num_len = 8
    header_list = ['Class', 'GT', 'DT', 'MT', 'R', 'P', 'FN', 'FP']
    
    loggerinfo(f"===>>> 第 [{frame_idx:^6}] 样本检测结果统计 {','.join(class_names)}")
    loggerinfo(f"{'|'.join([f'{i:^{num_len}}' for i in header_list])}|")
    
    class_names_ids = [int(i)+1 for i in range(len(class_names))]
    
    # 统计每个类别的预测框数量
    pred_counts_per_class = {}
    if len(pred_boxes) > 0 and pred_labels is not None:
        for class_id in class_names_ids:
            pred_counts_per_class[class_id] = np.sum(pred_labels == class_id)
    else:
        for class_id in class_names_ids:
            pred_counts_per_class[class_id] = 0
    
    total_list = []
    for label in class_names_ids:
        # GT数量统计
        gt_count = int(np.sum(gt_label_ids == label)) if len(gt_boxes) > 0 else 0
        pred_count = int(pred_counts_per_class[label])
        
        # 匹配数量统计
        if matched_gt_boxes_per_class is not None and label in matched_gt_boxes_per_class:
            mt_count = int(len(matched_gt_boxes_per_class[label]))
        else:
            mt_count = 0
        
        # 计算指标
        recall = mt_count / gt_count if gt_count > 0 else 0.0
        precision = mt_count / pred_count if pred_count > 0 else 0.0
        fn = gt_count - mt_count
        fp = pred_count - mt_count
        
        print_info = (f"{label:^{num_len}}|"
                    f"{gt_count:^{num_len}}|"
                    f"{pred_count:^{num_len}}|"
                    f"{mt_count:^{num_len}}|"
                    f"{recall:^{num_len}.3f}|"
                    f"{precision:^{num_len}.3f}|"
                    f"{fn:^{num_len}}|"
                    f"{fp:^{num_len}}|")
        loggerinfo(print_info)
        total_list.append([gt_count, pred_count, mt_count, recall, precision, fn, fp])
    
    total_data = np.array(total_list).sum(0)
    # total_data[0] = total_data[0]
    # total_data[1] = total_data[1]
    # total_data[2] = total_data[2]
    # total_data[5] = total_data[5]
    # total_data[6] = total_data[6]
    
    total_data[3] = total_data[2] / total_data[0] if total_data[0] > 0 else 0.0
    total_data[4] = total_data[4] / total_data[1] if total_data[1] > 0 else 0.0
    total_str = 'Total'
    print_info = (f"{total_str:^{num_len}}|"
                  f"{int(total_data[0]):^{num_len}}|"
                  f"{int(total_data[1]):^{num_len}}|"
                  f"{int(total_data[2]):^{num_len}}|"
                  f"{total_data[3]:^{num_len}.3f}|"
                  f"{total_data[4]:^{num_len}.3f}|"
                  f"{int(total_data[5]):^{num_len}}|"
                  f"{int(total_data[6]):^{num_len}}|"
                  )
    loggerinfo(print_info)
    
    loggerinfo(f"="*((num_len+1)*len(header_list)))


def get_distance_errors(cls_nums=4):
    distance = [
        {
            'x':[], 'y':[], 'z':[],
            'l':[], 'w':[], 'h':[],
            'r':[], 'v':[],
            'trans_err':[],  # ATE
            'vel_err':[],    # AVE
            'scale_err':[],  # ASE
            'orient_err':[], # AOE
            'attr_err':[],   # NOne
            'range':[],      # -1 | 0,1,2,3,... | 根据detrange个数
            'ref_x_err':[],    # 参考点x误差
            'ref_y_err':[],    # 参考点y误差
            
            # === 新增bad case记录 ===
            'bad_cases': {
                'fn_cases': [],  # False Negative cases
                'fp_cases': [],  # False Positive cases
                'hard_cases': [], # 难匹配cases (距离阈值边界)
            }
            
        } for _ in range(cls_nums)
        ]
    # 帧级别的统计信息
    distance.append({'frame_statistics': []})
    return distance