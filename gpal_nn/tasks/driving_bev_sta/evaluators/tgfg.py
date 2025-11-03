import numpy as np
import similaritymeasures
from .utils import bbox_overlaps
from scipy.spatial import distance
from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString, Polygon
from shapely.strtree import STRtree
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

def find_nearest_points(point_cloud_a, point_cloud_b, k=1):
    pc_a = np.zeros((point_cloud_a.shape[0], 3))
    pc_b = np.zeros((point_cloud_b.shape[0], 3))
    pc_a[:, :2] = point_cloud_a[:, :2]
    pc_b[:, :2] = point_cloud_b[:, :2]
    
    # 构建点云B的KDTree索引
    kdtree = KDTree(pc_b)
    
    # 搜索每个点在点云A中的最近邻（k=1表示只找最近的一个）
    distances, indices = kdtree.query(pc_a, k=k)
    
    return indices

def get_dense_polyline(curve, n_points):
    if not isinstance(curve, np.ndarray):
        curve = np.asarray(curve)
    # 计算累积弦长
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    
    # 生成插值点
    total_length = cum_length[-1]
    # n = max(2, int(total_length / 0.1))
    distances = np.linspace(0, total_length, n_points)
    
    # 线性插值
    interp_x = np.interp(distances, cum_length, curve[:,0])
    interp_y = np.interp(distances, cum_length, curve[:,1])
    sampled_points = np.column_stack((interp_x, interp_y)).astype(np.float32)
    return sampled_points

def get_dis_error(pred_points, gt_points):

    pred_len = np.sum(np.sqrt(np.sum(np.diff(pred_points, axis=0)**2, axis=1)))
    gt_len = np.sum(np.sqrt(np.sum(np.diff(gt_points, axis=0)**2, axis=1)))

    pred_points = get_dense_polyline(pred_points, int(pred_len / 0.01) + 2)
    gt_points = get_dense_polyline(gt_points, int(gt_len / 0.01) + 2)

    indice_gt_for_pred = find_nearest_points(pred_points, gt_points)
    indice_pred_for_gt = find_nearest_points(gt_points, pred_points)

    
    # 构建有效匹配掩码
    pre_indices = np.arange(len(pred_points))
    mask = indice_pred_for_gt[indice_gt_for_pred] == pre_indices
    
    if not np.any(mask):
        return np.nan
    # 计算距离
    dists = np.sqrt(((pred_points[mask] - gt_points[indice_gt_for_pred][mask]) ** 2).sum(axis=1))
    dist_error = np.mean(dists)
    return dist_error
def tpfp_gen(gen_lines,
             gt_lines,
             threshold=0.5,
             coord_dim=2,
             metric='POR'):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.ones((num_gens), dtype=np.float32)
    centerline_type_acc = []
    lane_marking_type_acc = []
    lane_marking_color_acc = []
    shape_type_acc = []
    dist_error_list = []
    pred_shapes = []  # 新增：存储预测的形状类型
    true_shapes = []  # 新增：存储真实的形状类型

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp, dist_error_list, lane_marking_type_acc, lane_marking_color_acc, shape_type_acc, pred_shapes, true_shapes, centerline_type_acc
    
    if num_gens == 0:
        return tp, fp, dist_error_list, lane_marking_type_acc, lane_marking_color_acc, shape_type_acc, pred_shapes, true_shapes, centerline_type_acc
    
    gen_scores = gen_lines[:,-1] # n
    # distance matrix: n x m
    matrix = polyline_score(
            gen_lines[:,:-5].reshape(num_gens,-1,coord_dim), 
            gt_lines[:,:-4].reshape(num_gts,-1,coord_dim),linewidth=2.,metric=metric)

    row_ind, col_ind = linear_sum_assignment(-matrix)
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        if matrix[pred_idx, gt_idx] >= threshold:
            tp[pred_idx] = 1
            fp[pred_idx] = 0
            dist_error = get_dis_error(gen_lines[pred_idx,:-5].reshape(-1, 2), gt_lines[gt_idx,:-4].reshape(-1, 2))
            dist_error_list.append(dist_error)
            # shape_type_acc.append(gen_lines[pred_idx,-1] == gt_lines[gt_idx,-1])
            pred_lane_marking_type = gen_lines[pred_idx, -4]  # 预测的车道线类别
            true_lane_marking_type = gt_lines[gt_idx, -4]     # 真实的车道线类别
            lane_marking_type_acc.append(pred_lane_marking_type == true_lane_marking_type)
            pred_lane_marking_color = gen_lines[pred_idx, -3]  # 预测的车道线颜色
            true_lane_marking_color = gt_lines[gt_idx, -3]     # 真实的车道线颜色
            lane_marking_color_acc.append(pred_lane_marking_color == true_lane_marking_color)

            # 获取形状类型并判断是否正确
            pred_shape = gen_lines[pred_idx, -2]  # 预测的形状类型
            true_shape = gt_lines[gt_idx, -2]     # 真实的形状类型
            shape_type_acc.append(pred_shape == true_shape)
            
            # 新增：记录形状类型用于混淆矩阵
            pred_shapes.append(pred_shape)
            true_shapes.append(true_shape)

            # 获取中心线类型并判断是否正确
            pred_centerline_type = gen_lines[pred_idx, -1]  # 预测的形状类型
            true_centerline_type = gt_lines[gt_idx, -1]     # 真实的形状类型
            centerline_type_acc.append(pred_centerline_type == true_centerline_type)
    # # for each det, the max iou with all gts
    # matrix_max = matrix.max(axis=1)
    # # for each det, which gt overlaps most with it
    # matrix_argmax = matrix.argmax(axis=1)
    # # sort all dets in descending order by scores
    # sort_inds = np.argsort(-gen_scores)

    # gt_covered = np.zeros(num_gts, dtype=bool)

    # # tp = 0 and fp = 0 means ignore this detected bbox,
    # for i in sort_inds:
    #     if matrix_max[i] >= threshold:
    #         matched_gt = matrix_argmax[i]
    #         if not gt_covered[matched_gt]:
    #             gt_covered[matched_gt] = True
    #             tp[i] = 1
    #         else:
    #             fp[i] = 1
    #     else:
    #         fp[i] = 1
    return tp, fp, dist_error_list, lane_marking_type_acc, lane_marking_color_acc, shape_type_acc, pred_shapes, true_shapes, centerline_type_acc


def polyline_score(pred_lines, gt_lines, linewidth=1., metric='POR'):
    '''
        each line with 1 meter width
        pred_lines: num_preds, List [npts, 2]
        gt_lines: num_gts, npts, 2
        gt_mask: num_gts, npts, 2
    '''
    positive_threshold = 1.
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = \
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                          for i in pred_lines]
    gt_lines_shapely =\
        [LineString(i).buffer(linewidth,
            cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                        for i in gt_lines]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    if metric=='POR':
        iou_matrix = np.zeros((num_preds, num_gts),dtype=np.float64)
    elif metric=='frechet':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric=='chamfer':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric=='chamfer_v2':
        iou_matrix = np.full((num_preds, num_gts), -100.)

    for i, pline in enumerate(gt_lines_shapely):

        for query_idx in tree.query(pline):
            # if o.intersects(pline):
            if pred_lines_shapely[query_idx].intersects(pline):
                pred_id = query_idx #index_by_id[id(o)]

                if metric=='POR':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    
                    valid_ab = (dist_mat.min(-1) < positive_threshold).sum()
                    valid_ba = (dist_mat.min(-2) < positive_threshold).sum()

                    iou_matrix[pred_id, i] = min(valid_ba,valid_ab) / line_length
                    # iou_matrix[pred_id, i] = ((valid_ba+valid_ab)/2) / line_length
                    # assert iou_matrix[pred_id, i] <= 1. and iou_matrix[pred_id, i] >= 0.
                elif metric=='frechet':
                    fdistance_1 = \
                        -similaritymeasures.frechet_dist(pred_lines[pred_id], gt_lines[i])
                    fdistance_2 = \
                        -similaritymeasures.frechet_dist(pred_lines[pred_id][::-1], gt_lines[i])
                    fdistance = max(fdistance_1,fdistance_2)
                    iou_matrix[pred_id, i] = fdistance

                elif metric=='chamfer':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id][:, :2], gt_lines[i][:, :2], 'euclidean')
                    
                    valid_ab = dist_mat.min(-1).sum()
                    valid_ba = dist_mat.min(-2).sum()

                    iou_matrix[pred_id, i] = -(valid_ba+valid_ab)/(2*line_length)
                    # if iou_matrix[pred_id, i] == 0:
                    #     import ipdb; ipdb.set_trace()
                elif metric=='chamfer_v2':
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], 'euclidean')
                    
                    valid_ab = dist_mat.min(-1).sum()
                    valid_ba = dist_mat.min(-2).sum()

                    iou_matrix[pred_id, i] = -(valid_ba/pred_lines[pred_id].shape[0]
                                                +valid_ab/gt_lines[i].shape[0])/2

    
    return iou_matrix
