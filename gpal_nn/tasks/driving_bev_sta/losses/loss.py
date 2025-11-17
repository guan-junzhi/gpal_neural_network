import torch
import numpy as np
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks.base.losses.loss import BaseLoss
from gpal_lightning.neural_network.tasks.builder import LOSSES
from gpal_nn.tasks.driving_bev_sta.losses.transform_gt import transform_gt_box, permute_line
from gpal_nn.tasks.driving_bev_sta.losses.map_loss import BaseMapLossCost
from gpal_lightning.utils.profiling import GetMemInfo, TrainSpeedRec, PrintTopProcesses, DetailProf
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *
import time
from tools_scripts.data_format_cvt import ShowDataStruct
import pickle as pkl
from shapely.geometry import LineString, Point

def project_point_to_polyline(polyline_coords, point_coords):
    """
    计算点到polyline的投影点，并将投影位置归一化到0-1范围
    :param polyline_coords: 折线坐标列表，格式为[(x1,y1), (x2,y2), ...]
    :param point_coords: 点坐标，格式为(x,y)
    :return: 投影点坐标(tuple)和归一化位置(float)
    """
    # 创建线对象和点对象
    line = LineString(polyline_coords)
    point = Point(point_coords)
    
    # 计算线的总长度
    line_length = line.length
    
    # 处理线长度为0的特殊情况
    if line_length == 0:
        return (polyline_coords[0], 0.0) if polyline_coords else ((0,0), 0.0)
    
    # 计算点到线的投影距离
    project_distance = line.project(point)
    
    # 计算归一化位置（限制在0-1范围内）
    normalized_position = project_distance / line_length
    normalized_position = max(0.0, min(1.0, normalized_position))
    
    # 根据归一化位置计算投影点
    projected_point = line.interpolate(project_distance)
    
    return (tuple(projected_point.coords[0]), normalized_position)

def pack_polyline_gt_points(data):
    annos = []
    classes = []
    lane_marking_types = []
    lane_marking_colors = []
    shape_types = []
    centerline_types = []
    centerline_directions = []
    is_split_merges = []
    keypoint_norms = []
    polygon_classes = []
    arrow_classes = []

    if 'points' in data['polylines']:
        annos.append(data['polylines']['points'])
        polyline_num = len(data['polylines']['points'])
        lane_marking_types.append(data['polylines']['classes'])
        lane_marking_colors.append(data['polylines']['color_type'])
        shape_types.append(data['polylines']['shape_type'])
        classes.append(np.ones(len(data['polylines']['points'])) * main_class_type_map['lane_marking'])
        centerline_types.append(np.ones(polyline_num) * (-1))
        centerline_directions.append(np.ones(polyline_num) * (-1))
        is_split_merges.append(np.zeros(polyline_num, dtype=np.int32))
        keypoint_norms.append(np.array([0] * polyline_num, dtype=np.float32))
        polygon_classes.append(np.ones(polyline_num) * (-1))
        arrow_classes.append(np.ones(polyline_num) * (-1))

    if 'points' in data['edges']:
        annos.append(data['edges']['points'])
        edge_num = len(data['edges']['points'])
        lane_marking_types.append(np.ones(edge_num) * (-1))
        lane_marking_colors.append(np.ones(edge_num) * (-1))
        shape_types.append(np.ones(edge_num) * (-1))
        centerline_types.append(np.ones(edge_num) * (-1))
        centerline_directions.append(np.ones(edge_num) * (-1))
        classes.append(np.ones(edge_num) * main_class_type_map['edge'])
        is_split_merges.append(np.zeros(edge_num, dtype=np.int32))
        keypoint_norms.append(np.array([0] * edge_num, dtype=np.float32))
        polygon_classes.append(np.ones(edge_num) * (-1))
        arrow_classes.append(np.ones(edge_num) * (-1))
    
    if 'centerlines' in data and 'points' in data['centerlines']:
        annos.append(data['centerlines']['points'])
        centerline_num = len(data['centerlines']['points'])
        lane_marking_types.append(np.ones(centerline_num) * (-1))
        lane_marking_colors.append(np.ones(centerline_num) * (-1))
        shape_types.append(np.ones(centerline_num) * (-1))
        centerline_types.append(data['centerlines']['classes'])
        centerline_directions.append(np.zeros(centerline_num))
        classes.append(np.ones(centerline_num) * main_class_type_map['centerline'])
        is_split_merges.append(np.array(data['centerlines']['is_split_merge'], dtype=np.int32))
        keypoint_norm = []
        for keypoint_valid, points, keypoint in zip(data['centerlines']['is_split_merge'], \
                                                    data['centerlines']['points'], \
                                                    data['centerlines']['keypoint']):
            if not keypoint_valid:
                keypoint_norm.append(0)
                continue
            projected_point, projected_point_norm = project_point_to_polyline(points, keypoint)
            keypoint_norm.append(projected_point_norm)
        keypoint_norms.append(np.array(keypoint_norm, dtype=np.float32))
        polygon_classes.append(np.ones(centerline_num) * (-1))
        arrow_classes.append(np.ones(centerline_num) * (-1))

    assert 'guideline' in data, "guideline must be in data"
    annos.append(data['guideline']['ego_path'])
    lane_marking_types.append(np.ones(1) * (-1))
    lane_marking_colors.append(np.ones(1) * (-1))
    shape_types.append(np.ones(1) * (-1))
    centerline_types.append(np.ones(1) * (-1))
    centerline_directions.append(np.ones(1) * (-1))
    classes.append(np.ones(1) * main_class_type_map['guideline_ego_path'])
    is_split_merges.append(np.zeros(1, dtype=np.int32))
    keypoint_norms.append(np.array([0] * 1, dtype=np.float32))
    polygon_classes.append(np.ones(1) * (-1))
    arrow_classes.append(np.ones(1) * (-1))

    if 'points' in data['polygons']:
        polygon_num = len(data['polygons']['points'])
        annos.append(data['polygons']['points'])
        classes.append(np.ones(polygon_num) * main_class_type_map['polygon'])
        lane_marking_types.append(np.ones(polygon_num) * (-1))
        lane_marking_colors.append(np.ones(polygon_num) * (-1))
        shape_types.append(np.ones(polygon_num) * (-1))
        centerline_types.append(np.ones(polygon_num) * (-1))
        centerline_directions.append(np.ones(polygon_num) * (-1))
        is_split_merges.append(np.zeros(polygon_num, dtype=np.int32))
        keypoint_norms.append(np.array([0] * polygon_num, dtype=np.float32))
        polygon_classes.append(data['polygons']['classes'])
        arrow_classes.append(np.ones(polygon_num) * (-1))

    if 'points' in data['arrows']:
        arrow_num = len(data['arrows']['points'])
        annos.append(data['arrows']['points'])
        classes.append(np.ones(arrow_num) * main_class_type_map['arrow'])
        lane_marking_types.append(np.ones(arrow_num) * (-1))
        lane_marking_colors.append(np.ones(arrow_num) * (-1))
        shape_types.append(np.ones(arrow_num) * (-1))
        centerline_types.append(np.ones(arrow_num) * (-1))
        centerline_directions.append(np.ones(arrow_num) * (-1))
        is_split_merges.append(np.zeros(arrow_num, dtype=np.int32))
        keypoint_norms.append(np.array([0] * arrow_num, dtype=np.float32))
        polygon_classes.append(np.ones(arrow_num) * (-1))
        arrow_classes.append(data['arrows']['classes'])


    if len(annos) > 0:
        annos = np.concatenate(annos, axis=0)
        classes = np.concatenate(classes, axis=0)
        lane_marking_types = np.concatenate(lane_marking_types, axis=0)
        lane_marking_colors = np.concatenate(lane_marking_colors, axis=0)
        shape_types = np.concatenate(shape_types, axis=0)
        centerline_types = np.concatenate(centerline_types, axis=0)
        centerline_directions = np.concatenate(centerline_directions, axis=0)
        is_split_merges = np.concatenate(is_split_merges, axis=0)
        keypoint_norms = np.concatenate(keypoint_norms, axis=0)
        polygon_classes = np.concatenate(polygon_classes, axis=0)
        arrow_classes = np.concatenate(arrow_classes, axis=0)

    return annos, classes, lane_marking_types, lane_marking_colors, shape_types, centerline_types, \
        centerline_directions, is_split_merges, keypoint_norms, polygon_classes, arrow_classes

def lane_loss_computation(preds, trues, loss_func, centerline_dataset):
    if "all_bbox_preds" not in preds:
        return {}
    all_cls_scores, all_bbox_pred, all_pts_pred, \
        all_lane_marking_types_pred, all_lane_marking_colors_pred, \
        all_shape_types_pred, all_centerline_types_preds, all_centerline_directions_preds, all_keypoint_classes_preds, all_keypoint_regs_pred, \
        all_polygon_classes_preds, all_arrow_classes_preds = \
        preds['all_cls_scores'], preds['all_bbox_preds'], preds['all_pts_preds'], \
        preds['all_lane_marking_types_preds'], preds['all_lane_marking_colors_preds'], \
        preds['all_shape_types_preds'], preds['all_centerline_types_preds'], \
        preds['all_centerline_directions_preds'], preds['all_keypoint_classes_preds'], \
        preds['all_keypoint_regs_preds'], preds['all_polygon_classes_preds'], preds['all_arrow_classes_preds']
    num_iter, bs, _, pts_per_vector, _ = all_pts_pred.shape
    loss_list = list()
    for k in range(num_iter):
        loss_dict = dict()
        time_dp = DetailProf()
        time_dp.Tic("begin")

        score_pred, bbox_pred, pts_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, centerline_direction_pred, \
            keypoint_cls_pred, keypoint_reg_pred, polygon_cls_pred, arrow_cls_pred = all_cls_scores[k], all_bbox_pred[k], all_pts_pred[k], \
            all_lane_marking_types_pred[k], all_lane_marking_colors_pred[k],all_shape_types_pred[k], all_centerline_types_preds[k], all_centerline_directions_preds[k], \
            all_keypoint_classes_preds[k], all_keypoint_regs_pred[k], all_polygon_classes_preds[k], all_arrow_classes_preds[k]
        time_dp.Duration("lane_loss_computation_all_1", "begin")
        
        stage_pred = [score_pred, bbox_pred, pts_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, 
                      centerline_type_pred, centerline_direction_pred, keypoint_cls_pred, keypoint_reg_pred, polygon_cls_pred, arrow_cls_pred]
        single_loss_dict = loss_func(stage_pred, trues)

        time_dp.Duration("lane_loss_computation_all_4",
                            "lane_loss_computation_all_1")
        
        single_loss_dict = {k: single_loss_dict[k] / bs for k in single_loss_dict}
        time_dp.Duration("lane_loss_computation_all_5",
                            "lane_loss_computation_all_4")

        time_dp.Duration("lane_loss_computation_all", "begin")
        # time_dp.Print()

        loss_list.append(single_loss_dict)
    total_dict = {}
    final_total_loss = 0.0
    for k in range(num_iter):
        d_loss_dict = loss_list[k]
        for key in d_loss_dict.keys():
            total_dict[f"lane_d{k}.{key}"] = d_loss_dict[key]

    for key in total_dict.keys():
        if "loss" in key:
            final_total_loss += total_dict[key]
    total_dict['total_loss'] = final_total_loss
    return total_dict


def ExpandTensor(data_tensor, default_len = 256):
    if data_tensor.shape[0] >= default_len:
        print(f"warning default_len = {default_len} data_tensor.shape[0] = {data_tensor.shape[0]}")
        return data_tensor[:default_len]
    tensor_size = list(data_tensor.shape)
    tensor_size[0] = default_len - tensor_size[0]
    return torch.cat([data_tensor, torch.zeros(tensor_size, device = data_tensor.device, dtype = data_tensor.dtype)], dim = 0)

def ProcessGt(trues, preds, centerline_dataset, output_group, max_ele_num=256):
    processed_gt = {"batchsize": len(trues)}
    start_x = 120
    start_y = 16
    all_pts_pred = preds['all_pts_preds']
    num_iter, bs, _, pts_per_vector, _ = all_pts_pred.shape
    gt_batched = {f"group_{idx}": {"valid_mask":[], "valid_len": [], "classes": [], "bboxes": [], "points": [], 
                                   "lane_marking_types": [], "lane_marking_colors": [], "types": [], "centerline_types": [], 
                                   "centerline_directions": [], "keyp_cls": [], "keyp_reg": [], "center_line_flag": [],
                                   "polygon_classes": [], "arrow_classes": []} for idx in range(len(output_group))}
    #TODO
    for gt in trues:
        annos, classes, lane_marking_types, lane_marking_colors, shape_types, centerline_types, centerline_directions, \
            is_split_merges, keypoint_norms, polygon_classes, arrow_classes = pack_polyline_gt_points(gt)
        # gt ploylines to gt bboxes  [n, 4], [n, 20, 2]
        bboxes_gt, points_gt = transform_gt_box(annos, start_x, start_y,
                                                num_pts_per_vec=pts_per_vector, y_first=False, device=all_pts_pred.device)

        # [n,20, 2]->[n, 2, 20, 2]  矢量线翻转建模
        points_gt = permute_line(points_gt)

        keypoint_cls_gt = is_split_merges
        if len(keypoint_norms) > 0:
            keypoint_norms_filp = 1 - keypoint_norms
            keypoint_norms_filp *= is_split_merges
            keypoint_reg_gt = np.stack(
                [keypoint_norms, keypoint_norms_filp], axis=0).transpose(1, 0)
        else:
            keypoint_reg_gt = np.zeros((0, 2), dtype=np.float32)

        if len(shape_types) == 0:
            lane_marking_types = torch.zeros(0).to(all_pts_pred.device).long()
            lane_marking_colors = torch.zeros(0).to(all_pts_pred.device).long()
            shape_types = torch.zeros(0).to(all_pts_pred.device).long()
            centerline_types = torch.zeros(0).to(all_pts_pred.device).long()
            centerline_directions = torch.zeros(0).to(all_pts_pred.device).long()
            keypoint_cls_gt = torch.zeros(0).to(all_pts_pred.device).long()
            keypoint_reg_gt = torch.zeros([0, 2]).to(
                all_pts_pred.device).float()
            classes = torch.zeros(0).to(all_pts_pred.device).long()
            polygon_classes = torch.zeros(0).to(all_pts_pred.device).long()
            arrow_classes = torch.zeros(0).to(all_pts_pred.device).long()
        else:
            lane_marking_types = torch.from_numpy(lane_marking_types).to(all_pts_pred.device).long()
            lane_marking_colors = torch.from_numpy(lane_marking_colors).to(all_pts_pred.device).long()
            shape_types = torch.from_numpy(shape_types).to(all_pts_pred.device).long()
            centerline_types = torch.from_numpy(centerline_types).to(all_pts_pred.device).long()
            centerline_directions = torch.from_numpy(centerline_directions).to(all_pts_pred.device).long()
            keypoint_cls_gt = torch.from_numpy(keypoint_cls_gt).to(all_pts_pred.device).long()
            keypoint_reg_gt = torch.from_numpy(keypoint_reg_gt).to(all_pts_pred.device).float()
            classes = torch.from_numpy(classes).to(all_pts_pred.device).long()
            polygon_classes = torch.from_numpy(polygon_classes).to(all_pts_pred.device).long()
            arrow_classes = torch.from_numpy(arrow_classes).to(all_pts_pred.device).long()
        for group_idx, group in enumerate(output_group):
            type_mask = torch.zeros_like(classes)
            for target_cls in group[0]:
                type_mask += (target_cls == classes)
            type_mask = type_mask > 0
            
            group_flag = f"group_{group_idx}"
            valid_mask = torch.zeros(max_ele_num).to(all_pts_pred.device).long()
            valid_mask[:len(classes[type_mask])] = 1
            gt_batched[group_flag]["valid_mask"].append(valid_mask)
            gt_batched[group_flag]["valid_len"].append(torch.tensor(len(classes[type_mask])).to(all_pts_pred.device).long())
            gt_batched[group_flag]["classes"].append(ExpandTensor(classes[type_mask], max_ele_num))
            gt_batched[group_flag]["bboxes"].append(ExpandTensor(bboxes_gt[type_mask], max_ele_num))
            gt_batched[group_flag]["points"].append(ExpandTensor(points_gt[type_mask], max_ele_num))
            gt_batched[group_flag]["lane_marking_types"].append(ExpandTensor(lane_marking_types[type_mask], max_ele_num))
            gt_batched[group_flag]["lane_marking_colors"].append(ExpandTensor(lane_marking_colors[type_mask], max_ele_num))
            gt_batched[group_flag]["types"].append(ExpandTensor(shape_types[type_mask], max_ele_num))
            gt_batched[group_flag]["centerline_types"].append(ExpandTensor(centerline_types[type_mask], max_ele_num))
            gt_batched[group_flag]["centerline_directions"].append(ExpandTensor(centerline_directions[type_mask], max_ele_num))
            gt_batched[group_flag]["keyp_cls"].append(ExpandTensor(keypoint_cls_gt[type_mask], max_ele_num))
            gt_batched[group_flag]["keyp_reg"].append(ExpandTensor(keypoint_reg_gt[type_mask], max_ele_num))
            gt_batched[group_flag]["center_line_flag"].append(torch.tensor(gt['calib_type'] in centerline_dataset).to(all_pts_pred.device).long())
            gt_batched[group_flag]["polygon_classes"].append(ExpandTensor(polygon_classes[type_mask], max_ele_num))
            gt_batched[group_flag]["arrow_classes"].append(ExpandTensor(arrow_classes[type_mask], max_ele_num))

    for group_flag in gt_batched:
        gt_batched[group_flag] = {k: torch.stack(gt_batched[group_flag][k], dim = 0) for k in gt_batched[group_flag]}

    return gt_batched


def loss_computation(preds, data, loss_func, output_group, centerline_dataset=None):
    total_dict = {}
    time_dp = DetailProf()
    time_dp.Tic("begin")
    processed_gt = ProcessGt(
        data, preds, centerline_dataset, output_group, max_ele_num=128)
    time_dp.Duration("ProcessGt", "begin")
    lane_total_dict = lane_loss_computation(
        preds, processed_gt, loss_func, centerline_dataset)
    time_dp.Duration("lane_loss_computation", "ProcessGt")
    # time_dp.Print()

    total_dict.update(lane_total_dict)
    return total_dict


@LOSSES.register_module()
class DRIVING_BEV_STALoss(BaseLoss):
    def __init__(self, global_config: GlobalConfig, task_config):

        pc_range = task_config.pc_range
        pc_range = [0, 0, 0, 32.0, 120.0, 0]
        self.centerline_dataset = task_config.centerline_dataset
        super(DRIVING_BEV_STALoss, self).__init__(pc_range, task_config)
        self.output_group = [([main_class_type_map[name] for name in group[0]], group[1]) for group in task_config.output_name_group.values()]
        self.polyline_loss = BaseMapLossCost(pc_range, cls_loss_weight=1.0, l1_loss_weight=4.0,
                                             giou_loss_weight=0.01, pts_l1_loss_weight=5.0, pts_dir_loss_weight=0.005, output_group=self.output_group)

    def forward(self, preds: torch.Tensor, trues: torch.Tensor, masks: torch.Tensor) -> dict:
        """

        Args:
            preds: preds tensor from model
            trues: trues tensor from dataloader
            masks: masks tensor from preprocess modules, used for mask invalid labels

        Returns: dictionary contains the loss of current batch.

        """

        loss = loss_computation(
            preds[0], trues, self.polyline_loss, self.output_group, self.centerline_dataset)
        return loss

if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("loss_computation.pkl", 'rb'))

    loss_computation(*inputs)
