import os
import pickle
import datetime

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from gpal_nn.tasks.driving_bev_dyn.evaluators.ap_utils import (
    ap_per_class_with_curves,)

from gpal_nn.tasks.driving_bev_dyn.evaluators.vis_utils import (
    plot_pr_curves,
    visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box,
    )

from gpal_nn.tasks.driving_bev_dyn.evaluators.cvt_utils import (
    convert_dict_to_numpy,
    )

from gpal_nn.tasks.driving_bev_dyn.evaluators.cores import (get_one_sample_statistics_rotated_3d_boxes_distance,
                   print_bad_cases_analysis,
                   export_bad_cases_to_file,
                   export_frames_bad_cases_to_file,
                   get_distance_errors,
                   )


class ObjectDetectionEvaluator:
    def __init__(self, 
                    class_names, 
                    det_range_list, 
                    r_at_p,
                    precision_points,
                    
                    distance_threshold_list, 
                    restricted_ratio,

                    ):
        
        self.class_names = class_names
        self.det_range_list = det_range_list

        self.slice_line_len = 295
        self.data_print_len = 12
        self.class_name_print_len = 20
        
        self.r_at_p = r_at_p
        self.precision_points = precision_points
        
        self.distance_threshold_list = distance_threshold_list
        self.restricted_ratio = restricted_ratio

        self.stat_name_width = 15    # 统计项名称列宽度
        self.range_col_width = 13
        
        self.metric_col_width = 15  # 指标名称列宽
        self.class_col_width = 13   # 每个类别子列宽度（含内部竖线分隔）

        self.print_title_name_list = [
            'Precision',
            'Recall',
            f'R@P{self.r_at_p}',
            'AP',
            'F1',
            
            'Ref_x_mean', 
            'Ref_y_mean', 
            
            'E_x_max@0.9', 
            'E_y_max@0.9', 
            
            'E_x_mean', 
            'E_y_mean', 
            'E_z_mean', 
            
            'E_l_mean', 
            'E_w_mean', 
            'E_h_mean', 
            'E_r_mean', 
            'E_v_mean',

            'ATE',
            'ASE',
            'AOE',
            'AVE',
        ]
        
        self.visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box = visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box
        # self.__setattr__(visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box.__name__, visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box)

    @staticmethod
    def load_det_annos(format_agnostic):
        
        def load_custom_data(filename_pkl):
            with open(filename_pkl, 'rb') as f:
                det_annos = det_annos_with_gt = pickle.load(f)
            return det_annos
        
        if isinstance(format_agnostic, str):
            if not os.path.exists(format_agnostic):
                raise FileNotFoundError(f"文件不存在: {format_agnostic}")
            det_annos = load_custom_data(format_agnostic)
            isinstance(det_annos, list)
        elif isinstance(format_agnostic, list):
            det_annos = format_agnostic
        else:
            raise NotImplementedError
        
        
        det_annos_numpy = []
        
        for det_anno in det_annos:
            det_anno_numpy = convert_dict_to_numpy(det_anno)
            
            curr_gt_boxes_3d_ = det_anno_numpy['gt_boxes']
            cur_gt = curr_gt_boxes_3d_.copy()
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            curr_gt_boxes_3d = cur_gt[:k + 1]
            det_anno_numpy['gt_boxes'] = curr_gt_boxes_3d
            if curr_gt_boxes_3d.shape[0] == 0:
                det_anno_numpy['gt_boxes'] = np.empty(shape=(0, 9), dtype=np.float32)
            
            
            det_annos_numpy.append(det_anno_numpy)
        
        return det_annos_numpy
    
    
    def evaluate(self, det_annos, loggerinfo=None, save_dir='./results', is_print_during_info=False, use_print_format='0'):
        os.makedirs(save_dir, exist_ok=True)
        if loggerinfo is None:
            loggerinfo = print
            loggerinfo("开始评估 ===>>> ====> ...")
        use_print_format = str(use_print_format)
        if use_print_format == '0':
            self.evaluate_print_format_0(det_annos, loggerinfo=loggerinfo, save_dir=save_dir, is_print_during_info=is_print_during_info)
            return 

        
        if use_print_format in['1', '2']:
            distance_errors_list = get_distance_errors(cls_nums=len(self.class_names))
            ego_pos = np.array([0.0, 0.0])
            
            dt_sample_metrics_tot = []
            gt_labels_tot = []
            
            for frame_i, curr_dt_gt_dict in enumerate(tqdm(det_annos, desc='Evaluating', leave=True, ncols=100)):
                
                # gt
                curr_gt_boxes_3d = curr_dt_gt_dict['gt_boxes'].copy()  # xyz lwh r vx vy cls_id = 10 | cls_id + 1

                # dt
                curr_dt_boxes = curr_dt_gt_dict['boxes_lidar'].copy()  # xyz lwh r vx vy = 9 | cls_id + 1
                curr_dt_label = curr_dt_gt_dict['pred_labels'].copy()
                # curr_dt_names = curr_dt_gt_dict['name'].copy()  # 已经按顺序取到
                curr_dt_score = curr_dt_gt_dict['score'].copy()
                
                # curr_gt_boxes_3d = np.empty((0, 10)).astype(np.float32)
                
                # curr_dt_boxes = np.empty((0, 9)).astype(np.float32)
                # curr_dt_label = np.empty((0)).astype(np.int32)
                # curr_dt_names = np.empty((0)).astype(np.str_)
                # curr_dt_score = np.empty((0)).astype(np.float32)
                
                # breakpoint()
                
                sample_metrics, distance_errors_list = get_one_sample_statistics_rotated_3d_boxes_distance(
                    curr_dt_boxes, 
                    curr_dt_score, 
                    curr_dt_label, 
                    curr_gt_boxes_3d, 
                    distance_errors_list, 
                    use_theory_to_mask_range='cuboid',  # 或者 'l2distance'
                    det_range_list=self.det_range_list,
                    use_projection=True, 
                    ego_pos=ego_pos,
                    distance_threshold_list=self.distance_threshold_list,
                    restricted_ratio=self.restricted_ratio,  # 径向 横向
                    best_match_strategy='min_total_distance',  # 'min_total_distance', 'min_longitudinal', 'min_lateral', 'weighted_distance'
                    frame_idx=frame_i,
                    
                    is_record_bad_cases=True,
                    distance_threshold_ratio=0.8,  # 用于定义hard case的阈值比例
                    
                    loggerinfo=loggerinfo,
                    is_print_during_info=is_print_during_info,
                    class_names=self.class_names,
                )

                dt_sample_metrics_tot = dt_sample_metrics_tot + sample_metrics
                gt_labels_tot = gt_labels_tot + curr_gt_boxes_3d[:, -1].astype(np.int32).tolist()
            
            gt_labels_tot_np = np.array(gt_labels_tot).astype(np.int32)

            # loggerinfo = print  # 只打印结果信息
            
            loggerinfo("\t>>>\t Start to compute metrics ...")
            dt_sample_metrics_tot = list(zip(*dt_sample_metrics_tot))
            
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in dt_sample_metrics_tot[:3]]
            pred_range_pages = np.concatenate(dt_sample_metrics_tot[3], 0).astype(np.int32)
            gt_range_pages = np.concatenate(dt_sample_metrics_tot[4], 0).astype(np.int32)
            
            # 存储每个range的数据用于最终打印
            range_data = {}
            range_summary = {}

            # 先计算每个range的数据
            for range_i, curr_range in enumerate(self.det_range_list):
                
                # pred 的区域限制
                curr_mask = pred_range_pages[:, range_i] == range_i
                curr_true_positives = true_positives[curr_mask]
                curr_pred_scores = pred_scores[curr_mask]
                curr_pred_labels = pred_labels[curr_mask].astype(np.int32)
                
                # gt的区域限制 !!!
                curr_mask_gt = gt_range_pages[:, range_i] == range_i
                curr_gt_labels = gt_labels_tot_np[curr_mask_gt]
                
                # === 区域头信息 
                total_tp = int(np.sum(curr_true_positives))
                total_gt = len(curr_gt_labels)
                total_pred = len(curr_pred_labels)
                
                range_summary[range_i] = {
                    'range': curr_range,
                    'total_pred': total_pred,
                    'total_gt': total_gt,
                    'total_tp': total_tp,
                    'precision': total_tp/max(total_pred, 1),
                    'recall': total_tp/max(total_gt, 1)
                }
                
                ret = ap_per_class_with_curves(curr_true_positives, 
                                            curr_pred_scores, 
                                            curr_pred_labels,
                                            curr_gt_labels,
                                            class_names=self.class_names,
                                            precision_points=self.precision_points,
                                            )
                curr_range_precision, curr_range_recall, curr_range_ap, curr_range_f1, pr_curves, recall_at_precision = ret
                
                # 绘制所有类别的PR曲线
                plot_pr_curves(pr_curves=pr_curves, 
                            extra_info=curr_range,
                            class_names=self.class_names, 
                            unique_classes=range(1, len(self.class_names)+1), 
                            recall_at_p50=None, 
                            recall_at_precision=recall_at_precision, 
                            save_path=f'{save_dir}/pr_curves_with_R@P_{range_i}.png')
                
                # 存储每个类别在当前range的数据
                range_data[range_i] = {}
                
                for curr_j, curr_name in enumerate(self.class_names):
                    
                    # 性能数据
                    p = curr_range_precision[curr_j]
                    r = curr_range_recall[curr_j]
                    ap = curr_range_ap[curr_j]
                    f1 = curr_range_f1[curr_j]
                    
                    if p < 0 and r < 0 and ap < 0 and f1 < 0:
                        p = r = ap = f1 = '-'
                    
                    # 误差数据
                    curr_range_mask_tp = np.array(distance_errors_list[curr_j]['range']).reshape(-1, len(self.det_range_list)).astype(np.int32)
                    curr_range_mask_check = curr_range_mask_tp[:, range_i] == range_i
                    
                    if not np.any(curr_range_mask_check):
                        range_data[range_i][curr_name] = ['-' for _ in range(24)]  # 打印字段数
                        continue
                    
                    # 原始的含有正负信息的误差
                    dis_x = np.array(distance_errors_list[curr_j]['x'])[curr_range_mask_check]
                    dis_y = np.array(distance_errors_list[curr_j]['y'])[curr_range_mask_check]
                    dis_z = np.array(distance_errors_list[curr_j]['z'])[curr_range_mask_check]
                    dis_l = np.array(distance_errors_list[curr_j]['l'])[curr_range_mask_check]
                    dis_w = np.array(distance_errors_list[curr_j]['w'])[curr_range_mask_check]
                    dis_h = np.array(distance_errors_list[curr_j]['h'])[curr_range_mask_check]
                    dis_r = np.array(distance_errors_list[curr_j]['r'])[curr_range_mask_check]
                    dis_v = np.array(distance_errors_list[curr_j]['v'])[curr_range_mask_check]
                    ref_x = np.array(distance_errors_list[curr_j]['ref_x_err'])[curr_range_mask_check]
                    ref_y = np.array(distance_errors_list[curr_j]['ref_y_err'])[curr_range_mask_check]
                    
                    # 计算各项指标
                    dis_mean_x = np.mean(abs(dis_x))
                    dis_mean_y = np.mean(abs(dis_y))
                    dis_max_x = np.percentile(abs(dis_x), 90)
                    dis_max_y = np.percentile(abs(dis_y), 90)
                    ref_mean_x = np.mean(abs(ref_x))
                    ref_mean_y = np.mean(abs(ref_y))
                    dis_mean_z = np.mean(abs(dis_z))
                    dis_mean_l = np.mean(abs(dis_l))
                    dis_mean_w = np.mean(abs(dis_w))
                    dis_mean_h = np.mean(abs(dis_h))
                    dis_mean_r = np.mean(abs(dis_r))
                    dis_mean_v = np.mean(abs(dis_v))
                    
                    ATE = np.mean(np.array(distance_errors_list[curr_j]['trans_err'])[curr_range_mask_check])
                    ASE = np.mean(np.array(distance_errors_list[curr_j]['scale_err'])[curr_range_mask_check])
                    AOE = np.mean(np.array(distance_errors_list[curr_j]['orient_err'])[curr_range_mask_check])
                    AVE = np.mean(np.array(distance_errors_list[curr_j]['vel_err'])[curr_range_mask_check])
                    
                    try:
                        R_P07 = recall_at_precision[curr_j+1][f'R@P{self.r_at_p}']
                    except:
                        R_P07 = recall_at_precision[curr_j+1][f'R@P0.7']
                    
                    # 存储数据
                    range_data[range_i][curr_name] = [
                        pr_curves[curr_j+1]['num_Dt'],
                        pr_curves[curr_j+1]['num_Gt'],
                        int(pr_curves[curr_j+1]['max_tp']),
                        p, r, R_P07, ap, f1,
                        ref_mean_x, ref_mean_y,
                        dis_max_x, dis_max_y,
                        dis_mean_x, dis_mean_y, dis_mean_z,
                        dis_mean_l, dis_mean_w, dis_mean_h,
                        dis_mean_r, dis_mean_v,
                        ATE, ASE, AOE, AVE
                    ]
            # === 新的打印格式 ===
            # 打印range总体信息表头
            loggerinfo("各区域统计信息汇总: ")

            # 打印range统计信息表头
            stat_name_width = self.stat_name_width    # 统计项名称列宽度
            range_col_width = self.range_col_width
            metric_col_width = self.metric_col_width  # 指标名称列宽
            class_col_width = self.class_col_width   # 每个类别子列宽度（含内部竖线分隔）
            summary_col_width = len(self.det_range_list) * range_col_width + 1  # 计算总宽度
            
            for range_i in range(len(self.det_range_list)):
                range_info = f"Area, Range {range_i}: {str(range_summary[range_i]['range'])}"
                loggerinfo(range_info)
            
            separator = f"{'-'*(stat_name_width + len(self.det_range_list*(range_col_width+1)))}"
            loggerinfo(separator)
            
            header = f"{'Metrics, Range':<{stat_name_width}}"
            for range_i in range(len(self.det_range_list)):
                range_info = f"Range {range_i}"  # 多个空格
                header += f"|{range_info:^{range_col_width}}"
            loggerinfo(header)

            separator = f"{'-'*(stat_name_width + len(self.det_range_list*(range_col_width+1)))}"
            loggerinfo(separator)

            # 各项统计信息
            stats_names = ['Total Pred', 'Total Gt', 'Match Nums', 'P(precision)', 'R(recall)']
            stats_keys = ['total_pred', 'total_gt', 'total_tp', 'precision', 'recall']

            for stat_name, stat_key in zip(stats_names, stats_keys):
                line = f"{stat_name:<{stat_name_width}}"
                for range_i in range(len(self.det_range_list)):
                    if stat_key in ['precision', 'recall']:
                        value = f"{range_summary[range_i][stat_key]:.3f}"
                    else:
                        value = f"{range_summary[range_i][stat_key]}"
                    line += f"|{value:^{range_col_width}}"
                loggerinfo(line)

            separator = f"{'-'*(stat_name_width + len(self.det_range_list*(range_col_width+1)))}"
            loggerinfo(separator)
            
            if use_print_format == '1':
                self.format_1(
                    gt_labels_tot_np=gt_labels_tot_np,
                    true_positives=true_positives, 
                    pred_scores=pred_scores, 
                    pred_labels=pred_labels, 
                    pred_range_pages=pred_range_pages,
                    gt_range_pages=gt_range_pages,
                    range_data=range_data,
                    range_summary=range_summary,
                    loggerinfo=loggerinfo,
                    save_dir=save_dir,
                    distance_errors_list=distance_errors_list,
                )
            elif use_print_format == '2':
                self.format_2(
                    gt_labels_tot_np=gt_labels_tot_np,
                    true_positives=true_positives, 
                    pred_scores=pred_scores, 
                    pred_labels=pred_labels, 
                    pred_range_pages=pred_range_pages,
                    gt_range_pages=gt_range_pages,
                    range_data=range_data,
                    range_summary=range_summary,
                    loggerinfo=loggerinfo,
                    save_dir=save_dir,
                    distance_errors_list=distance_errors_list,
                )
        else:
            raise NotImplementedError
        
        # 全区域汇总信息
        loggerinfo(f"全区域: P 匹配的预测框数量: {int(np.sum(true_positives))}/{len(true_positives)} = {int(np.sum(true_positives))/max(len(true_positives), 1):.3f}")
        loggerinfo(f"全区域: R 匹配的预测框数量: {int(np.sum(true_positives))}/{len(gt_labels_tot_np)} = {int(np.sum(true_positives))/max(len(gt_labels_tot_np), 1):.3f}")

        loggerinfo("")
        loggerinfo(f"结果保存路径: {save_dir}")
        loggerinfo(f"Done! Date: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # 处理图像合并
        img_data_list = []
        for curr_i in range(len(self.det_range_list)):
            curr_img_file = f'{save_dir}/pr_curves_with_R@P_{curr_i}.png'
            data = plt.imread(curr_img_file)
            os.remove(curr_img_file)
            img_data_list.append(data)
        # 合并图像
        merged_image = np.concatenate(img_data_list, axis=1)
        plt.imsave(f'{save_dir}/pr_curves_with_R@P_all.png', merged_image)
        
        print_bad_cases_analysis(self.class_names, distance_errors_list, loggerinfo=loggerinfo)
        export_bad_cases_to_file(self.class_names, distance_errors_list, output_dir=save_dir, loggerinfo=loggerinfo)
        export_frames_bad_cases_to_file(self.class_names, distance_errors_list, output_dir=save_dir, loggerinfo=loggerinfo)
        
        return distance_errors_list

    def evaluate_print_format_0(self, det_annos, loggerinfo=None, save_dir='./results', is_print_during_info=False):
        distance_errors_list = get_distance_errors(cls_nums=len(self.class_names))
        ego_pos = np.array([0.0, 0.0])
        
        dt_sample_metrics_tot = []
        gt_labels_tot = []
        
        for frame_i, curr_dt_gt_dict in enumerate(tqdm(det_annos, desc='Evaluating', leave=True, ncols=100)):
            
            # gt
            curr_gt_boxes_3d = curr_dt_gt_dict['gt_boxes']  # xyz lwh r vx vy cls_id = 10 | cls_id + 1
            
            # dt
            curr_dt_boxes = curr_dt_gt_dict['boxes_lidar']  # xyz lwh r vx vy = 9 | cls_id + 1
            curr_dt_label = curr_dt_gt_dict['pred_labels']
            curr_dt_names = curr_dt_gt_dict['name']  # 已经按顺序取到
            curr_dt_score = curr_dt_gt_dict['score']
            
            sample_metrics, distance_errors_list = get_one_sample_statistics_rotated_3d_boxes_distance(
                curr_dt_boxes, 
                curr_dt_score, 
                curr_dt_label, 
                curr_gt_boxes_3d, 
                distance_errors_list, 
                use_theory_to_mask_range='cuboid',  # 或者 'l2distance'
                det_range_list=self.det_range_list,
                use_projection=True, 
                ego_pos=ego_pos,
                distance_threshold_list=self.distance_threshold_list,
                restricted_ratio=self.restricted_ratio,  # 径向 横向
                frame_idx=frame_i,
                loggerinfo=loggerinfo,
                is_print_during_info=is_print_during_info,
            )

            dt_sample_metrics_tot = dt_sample_metrics_tot + sample_metrics
            gt_labels_tot = gt_labels_tot + curr_gt_boxes_3d[:, -1].astype(np.int32).tolist()
        
        gt_labels_tot_np = np.array(gt_labels_tot).astype(np.int32)
            
        loggerinfo("\t>>>\t Start to compute metrics ...")
        dt_sample_metrics_tot = list(zip(*dt_sample_metrics_tot))
        
        # true_positives, pred_scores, pred_labels, pred_range_pages = [np.concatenate(x, 0) for x in list(zip(*dt_sample_metrics_tot))]
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in dt_sample_metrics_tot[:3]]
        pred_range_pages = np.concatenate(dt_sample_metrics_tot[3], 0).astype(np.int32)
        gt_range_pages = np.concatenate(dt_sample_metrics_tot[4], 0).astype(np.int32)
        
        
        for range_i, curr_range in enumerate(self.det_range_list):
            
            # pred 的区域限制
            curr_mask = pred_range_pages[:, range_i] == range_i
            curr_true_positives = true_positives[curr_mask]
            curr_pred_scores = pred_scores[curr_mask]
            curr_pred_labels = pred_labels[curr_mask].astype(np.int32)
            
            # gt的区域限制 !!!
            curr_mask_gt = gt_range_pages[:, range_i] == range_i
            curr_gt_labels = gt_labels_tot_np[curr_mask_gt]
            
            
            # === 区域头信息 
            total_tp = int(np.sum(curr_true_positives))
            total_gt = len(curr_gt_labels)
            total_pred = len(curr_pred_labels)
            
            loggerinfo(f"  区域 [{range_i}]: {curr_range} 统计信息:")
            loggerinfo(f"  预测(DT)框总数: {total_pred}")
            loggerinfo(f"  真值(GT)框总数: {total_gt}")
            loggerinfo(f"  匹配(TP)成功数: {total_tp}")
            loggerinfo(f"  精准率(P)检查: {total_tp}/{total_pred} = {total_tp/max(total_pred, 1):.3f}")
            loggerinfo(f"  召回率(R)检查: {total_tp}/{total_gt} = {total_tp/max(total_gt, 1):.3f}")
            
            ret = ap_per_class_with_curves(curr_true_positives, 
                                        curr_pred_scores, 
                                        curr_pred_labels,
                                        curr_gt_labels,
                                        class_names=self.class_names,
                                        precision_points=self.precision_points,
                                        )
            curr_range_precision, curr_range_recall, curr_range_ap, curr_range_f1, pr_curves, recall_at_precision = ret
            
            # 绘制所有类别的PR曲线
            # plot_pr_curves(pr_curves, class_names, name=curr_range, save_path=f'./results/pr_curves_{range_i}.png')
            plot_pr_curves(pr_curves=pr_curves, 
                        extra_info=curr_range,
                        class_names=self.class_names, 
                        unique_classes=range(1, len(self.class_names)+1), 
                        recall_at_p50=None, 
                        recall_at_precision=recall_at_precision, 
                        save_path=f'{save_dir}/pr_curves_with_R@P_{range_i}.png')
            
            
            title = (
                f"\t>>> [Range {range_i}] >> {''.join([f'-' for i in range(self.slice_line_len)])}"
                )
            loggerinfo(title)

            # loggerinfo(f'\n\n============= 误差指标-Errors =============')

            title = (
                    f"\t>>> [Range {range_i}] >> {'class_name':^{self.class_name_print_len}} {'|'.join([f'{i:^{self.data_print_len}}' for i in self.print_title_name_list])}"
                )
            loggerinfo(title)
            
            title = (
                f"\t>>> [Range {range_i}] >> {''.join([f'-' for i in range(self.slice_line_len)])}"
                )
            loggerinfo(title)
            
            for curr_j, curr_name in enumerate(self.class_names):
                
                # 性能打印
                p = curr_range_precision[curr_j]
                r = curr_range_recall[curr_j]
                ap = curr_range_ap[curr_j]
                f1 = curr_range_f1[curr_j]
                
                if p < 0 and r < 0 and ap < 0 and f1 < 0:
                    p = r = ap = f1 = np.array('-')
                
                # 误差打印
                curr_range_mask_tp = np.array(distance_errors_list[curr_j]['range']).reshape(-1, len(self.det_range_list)).astype(np.int32)
                curr_range_mask_check = curr_range_mask_tp[:, range_i] == range_i  # 当前类别中属于当前range的
                
                if not np.any(curr_range_mask_check):
                    data_list = ['-' for i in range(20)]
                    data_info = (
                    f"\t>>> [Range {range_i}] >> {curr_name:^{self.class_name_print_len}} {'|'.join([f'{i:^{self.data_print_len}}' for i in data_list])}"
                    )
                    loggerinfo(data_info)
                    continue
                
                # 原始的含有正负信息的误差
                dis_x = np.array(distance_errors_list[curr_j]['x'])[curr_range_mask_check]
                dis_y = np.array(distance_errors_list[curr_j]['y'])[curr_range_mask_check]
                dis_z = np.array(distance_errors_list[curr_j]['z'])[curr_range_mask_check]
                dis_l = np.array(distance_errors_list[curr_j]['l'])[curr_range_mask_check]
                dis_w = np.array(distance_errors_list[curr_j]['w'])[curr_range_mask_check]
                dis_h = np.array(distance_errors_list[curr_j]['h'])[curr_range_mask_check]
                dis_r = np.array(distance_errors_list[curr_j]['r'])[curr_range_mask_check]
                dis_v = np.array(distance_errors_list[curr_j]['v'])[curr_range_mask_check]
                ref_x = np.array(distance_errors_list[curr_j]['ref_x_err'])[curr_range_mask_check]
                ref_y = np.array(distance_errors_list[curr_j]['ref_y_err'])[curr_range_mask_check]
                
                # 将被打印的信息
                dis_mean_x  = np.mean(abs(dis_x))
                dis_mean_y  = np.mean(abs(dis_y))
                
                dis_max_x = np.percentile(abs(dis_x), 90)
                dis_max_y = np.percentile(abs(dis_y), 90)
                
                ref_mean_x  = np.mean(abs(ref_x))
                ref_mean_y  = np.mean(abs(ref_y))
                
                dis_mean_z  = np.mean(abs(dis_z))
                dis_mean_l  = np.mean(abs(dis_l))
                dis_mean_w  = np.mean(abs(dis_w))
                dis_mean_h  = np.mean(abs(dis_h))
                dis_mean_r  = np.mean(abs(dis_r))
                dis_mean_v  = np.mean(abs(dis_v))
                
                ATE = np.mean(np.array(distance_errors_list[curr_j]['trans_err'])[curr_range_mask_check])
                ASE = np.mean(np.array(distance_errors_list[curr_j]['scale_err'])[curr_range_mask_check])
                AOE = np.mean(np.array(distance_errors_list[curr_j]['orient_err'])[curr_range_mask_check])
                AVE = np.mean(np.array(distance_errors_list[curr_j]['vel_err'])[curr_range_mask_check])
                try:
                    R_P07 = recall_at_precision[curr_j+1][f'R@P{self.r_at_p}']
                except:
                    raise NotImplementedError
                    R_P07 = recall_at_precision[curr_j+1][f'R@P0.7']
                
                # 合并
                data_list = [
                    p,
                    r,
                    R_P07,
                    ap,
                    f1,
                    
                    ref_mean_x,
                    ref_mean_y,
                    
                    dis_max_x,
                    dis_max_y,
                    
                    dis_mean_x,
                    dis_mean_y,
                    dis_mean_z,
                    
                    dis_mean_l,
                    dis_mean_w,
                    dis_mean_h,
                    dis_mean_r,
                    dis_mean_v,

                    ATE,
                    ASE,
                    AOE,
                    AVE,
                    ]
                assert len(data_list) == len(self.print_title_name_list), f'列表维度需一一对应 {len(data_list)} != {len(self.print_title_name_list)}'
                
                data_info = (
                    f"\t>>> [Range {range_i}] >> {curr_name:^{self.class_name_print_len}} {'|'.join([f'{i:^{self.data_print_len}.4f}' for i in data_list])}"
                )
                loggerinfo(data_info)

            title = (
                f"\t>>> [Range {range_i} Done] >> {''.join([f'-' for i in range(self.slice_line_len - 5)])}"
                )
            loggerinfo(title)
        # 全区域汇总信息
        loggerinfo(f"全区域: P 匹配的预测框数量: {int(np.sum(true_positives))}/{len(true_positives)} = {int(np.sum(true_positives))/max(len(true_positives), 1):.3f}")
        loggerinfo(f"全区域: R 匹配的预测框数量: {int(np.sum(true_positives))}/{len(gt_labels_tot_np)} = {int(np.sum(true_positives))/max(len(gt_labels_tot_np), 1):.3f}")

        loggerinfo("")
        loggerinfo(f"结果保存路径: {save_dir}")
        loggerinfo(f"Done! Date: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # 处理图像合并
        img_data_list = []
        for curr_i in range(len(self.det_range_list)):
            curr_img_file = f'{save_dir}/pr_curves_with_R@P_{curr_i}.png'
            data = plt.imread(curr_img_file)
            os.remove(curr_img_file)
            img_data_list.append(data)
        # 合并图像
        merged_image = np.concatenate(img_data_list, axis=1)
        plt.imsave(f'{save_dir}/pr_curves_with_R@P_all.png', merged_image)
        
        print_bad_cases_analysis(self.class_names, distance_errors_list, loggerinfo=loggerinfo)
        export_bad_cases_to_file(self.class_names, distance_errors_list, output_dir=save_dir, loggerinfo=loggerinfo)
        export_frames_bad_cases_to_file(self.class_names, distance_errors_list, output_dir=save_dir, loggerinfo=loggerinfo)
        
    def format_1(self, 
                 gt_labels_tot_np,
                 true_positives, 
                 pred_scores, 
                 pred_labels, 
                 pred_range_pages,
                 gt_range_pages,
                 range_data,
                 range_summary,
                 loggerinfo,
                 save_dir,
                 distance_errors_list,
                 ):

        loggerinfo("")
        loggerinfo("各类别详细指标 (以Range为列, 指标为行)")

        stat_name_width = self.stat_name_width    # 统计项名称列宽度
        range_col_width = self.range_col_width
        metric_col_width = self.metric_col_width  # 指标名称列宽
        class_col_width = self.class_col_width   # 每个类别子列宽度（含内部竖线分隔）

        summary_col_width = len(self.det_range_list) * (class_col_width*len(self.class_names)+1+len(self.class_names))  # 计算总宽度
        separator = f"{'-'*(metric_col_width + summary_col_width)}"
        loggerinfo(separator.replace('-', '='))

        header_range = f"{'Metrics, Range':<{metric_col_width}}"
        header = f"{'Metrics':<{metric_col_width}}"
        for range_i in range(len(self.det_range_list)):
            range_title_parts = []
            for class_idx, class_name in enumerate(self.class_names):
                part = f"{class_name:^{class_col_width}}"
                range_title_parts.append(part)

            range_title = "|" + "|".join(range_title_parts).strip("|") + "|"
            header += range_title
            
            txt = f'Range {range_i}: {self.det_range_list[range_i]}'
            header_range += f'|{txt:^{(len(self.class_names)*class_col_width+len(self.class_names)-1)}}|'
            
        loggerinfo(header_range)
        loggerinfo(separator)
        loggerinfo(header)
        loggerinfo(separator)
        
        metric_names = [
            'num_Dt', 'num_Gt', 'max_tp',
            'Precision', 'Recall', f'R@P{self.r_at_p}', 'AP', 'F1',
                        'Ref_x_mean', 'Ref_y_mean', 'E_x_max@0.9', 'E_y_max@0.9',
                        'E_x_mean', 'E_y_mean', 'E_z_mean',
                        'E_l_mean', 'E_w_mean', 'E_h_mean',
                        'E_r_mean', 'E_v_mean',
                        'ATE', 'ASE', 'AOE', 'AVE']

        for metric_idx, metric_name in enumerate(metric_names):
            line = f"{metric_name:<{metric_col_width}}"  # 指标列左对齐
            for range_i in range(len(self.det_range_list)):
                data_parts = []
                for class_idx, class_name in enumerate(self.class_names):
                    value = range_data[range_i][class_name][metric_idx]
                    # 数值格式化：统一4位小数（整数补.0000）
                    if isinstance(value, (int, np.int32, np.int64)) and value!= '-':
                        formatted_value = f"{value}"
                    elif isinstance(value, (float, np.float32)) and value!= '-':
                        formatted_value = f"{value:^.4f}"
                    else:
                        formatted_value = str(value)
                    part = f"{formatted_value:^{class_col_width}}"
                    data_parts.append(part)
                # 合并为大列数据（去掉首尾空格）
                data_range = "|" + "|".join(data_parts).strip("|") + "|"
                line += data_range
            loggerinfo(line)
            if metric_idx == 2 or metric_idx == 7 or metric_idx == len(metric_names) - 5:
                loggerinfo(separator)
        
        # 打印结束分隔线（与表头对齐）
        loggerinfo(separator.replace('-', '='))
        loggerinfo("")
            
    def format_2(self, 
                 gt_labels_tot_np,
                 true_positives, 
                 pred_scores, 
                 pred_labels, 
                 pred_range_pages,
                 gt_range_pages,
                 range_data,
                 range_summary,
                 loggerinfo,
                 save_dir,
                 distance_errors_list,
                 ):
        
        stat_name_width = self.stat_name_width    # 统计项名称列宽度
        range_col_width = self.range_col_width
        metric_col_width = self.metric_col_width  # 指标名称列宽
        class_col_width = self.class_col_width   # 每个类别子列宽度（含内部竖线分隔）

        summary_col_width = len(self.det_range_list) * range_col_width + 1  # 计算总宽度
        
        loggerinfo("")
        loggerinfo("各类别详细指标 (以Range为大列, 类别为子列, 指标为行)")
        
        metric_names = [
            'num_Dt', 'num_Gt', 'max_tp',
            'Precision', 'Recall', f'R@P{self.r_at_p}', 'AP', 'F1',
                        'Ref_x_mean', 'Ref_y_mean', 'E_x_max@0.9', 'E_y_max@0.9',
                        'E_x_mean', 'E_y_mean', 'E_z_mean',
                        'E_l_mean', 'E_w_mean', 'E_h_mean',
                        'E_r_mean', 'E_v_mean',
                        'ATE', 'ASE', 'AOE', 'AVE']

        summary_col_width = class_col_width*len(self.class_names)+1+len(self.class_names)  # 计算总宽度
        separator = f"{'-'*(metric_col_width + summary_col_width)}"

        for range_i in range(len(self.det_range_list)):
            
            # 标题
            loggerinfo(separator.replace('-', '='))
            header_range = f"{'Metrics, Range':<{metric_col_width}}"
            header = f"{'Metrics':<{metric_col_width}}"
            range_title_parts = []
            for class_idx, class_name in enumerate(self.class_names):
                part = f"{class_name:^{class_col_width}}"
                range_title_parts.append(part)
            
            range_title = "|" + "|".join(range_title_parts).strip("|") + "|"
            header += range_title

            txt = f'Range {range_i}: {self.det_range_list[range_i]}'
            header_range += f'|{txt:^{(len(self.class_names)*class_col_width+len(self.class_names)-1)}}|'

            loggerinfo(header_range)
            loggerinfo(separator)
            loggerinfo(header)
            loggerinfo(separator)

            # 数据
            for metric_idx, metric_name in enumerate(metric_names):
                data_parts = []
                line = f"{metric_name:<{metric_col_width}}"  # 指标列左对齐
                for class_idx, class_name in enumerate(self.class_names):
                    value = range_data[range_i][class_name][metric_idx]
                    # 数值格式化：统一4位小数（整数补.0000）
                    if isinstance(value, (int, np.int32, np.int64)) and value!= '-':
                        formatted_value = f"{value}"
                    elif isinstance(value, (float, np.float32)) and value!= '-':
                        formatted_value = f"{value:^.4f}"
                    else:
                        formatted_value = str(value)
                    part = f"{formatted_value:^{class_col_width}}"
                    data_parts.append(part) 
                # 合并为大列数据（去掉首尾空格）
                data_range = "|" + "|".join(data_parts).strip("|") + "|"
                line += data_range
                loggerinfo(line)
                
                if metric_idx == 2 or metric_idx == 7 or metric_idx == len(metric_names) - 5:
                    loggerinfo(separator)

            # 打印结束分隔线（与表头对齐）
            loggerinfo(separator.replace('-', '='))
            loggerinfo("")
    
    def get_worst_or_best_frames_analysis_by_recall(self, distance_errors_list, top_n=100, recall_thr=0.5, find_worst=True):
        """
        分析bad case最多的前N帧
        
        Args:
            distance_errors_list: 距离误差列表
            top_n: 返回前N帧
            
        Returns:
            list: 按bad case数量排序的帧数据
        """
        frame_stats_container = distance_errors_list[-1]['frame_statistics']
        
        # 计算每帧的bad case总数
        frame_scores = []
        for frame_data in frame_stats_container:
            frame_data_new = frame_data.copy()
            
            total_gt = frame_data_new['total_gt_count']
            total_fn = frame_data_new['summary']['total_fn_count']

            total_recall = 1.0 - total_fn / max(total_gt, 1)
            
            frame_data_new.update({'total_recall': total_recall,})
            frame_scores.append(
                frame_data_new
            )
        
        # 按照挑选标准排序
        if find_worst:
            frame_scores.sort(key=lambda x: x['total_recall'], reverse=False)
        else:
            frame_scores.sort(key=lambda x: x['total_recall'], reverse=True)
        
        return frame_scores[:top_n]
    
    def get_frame_infos_from_distance_errors(self, distance_errors_list,):
        frame_stats_container = distance_errors_list[-1]['frame_statistics']
        
        # 计算每帧的bad case总数
        frame_scores = []
        for frame_data in frame_stats_container:
            frame_data_new = frame_data.copy()

            frame_scores.append(
                frame_data_new
            )
        
        return frame_scores
    
    @staticmethod
    def find_fn_fp_max_by_find_fn_fp_max_via_class_from_json(frame_statistics, top_n=100, recall_thr=0.5, find_worst=True, class_id='', find_fnfp=''):
        frame_stats_container = frame_statistics
        
        class_id_str = class_id
        
        frame_fn_fp_list = []
        
        
        for frame_data in frame_stats_container:
            frame_data_new = frame_data.copy()
            
            if class_id_str not in frame_data_new['classes_data']:
               continue 
            
            fn_nums = len(frame_data_new['classes_data'][class_id_str]['fn_cases'])
            fp_nums = len(frame_data_new['classes_data'][class_id_str]['fp_cases'])
            fnfp_nums = fn_nums + fp_nums
            
            if find_fnfp == 'fn':
                nums = fn_nums
            elif find_fnfp == 'fp':
                nums = fp_nums
            elif find_fnfp == 'fnfp':
                nums = fnfp_nums
            else:
                raise NotImplementedError
            
            frame_data_new.update({'fnfp_nums': nums})
            frame_fn_fp_list.append(frame_data_new)
            
        frame_fn_fp_list.sort(key=lambda x: x['fnfp_nums'], reverse=True)
        
        return frame_fn_fp_list[:top_n]
    
    def get_worst_class_frames_analysis(self, distance_errors_list, top_n=10):
        """
        分析每个类别bad case最多的前N帧
        
        Args:
            distance_errors_list: 距离误差列表
            top_n: 返回前N帧
            
        Returns:
            dict: 每个类别按bad case数量排序的帧数据
        """
    
    
    def query_raw_frame_info(self, query_frame_infos, key_frame_infos):
        """
        查询原始帧信息
        
        Args:
            frame_id: 帧ID
            
        Returns:
            dict: 帧信息
        """
        query_frame_infos_new = []
        
        for query_info in query_frame_infos:
            
            # 搜集信息
            curr_query_info = query_info.copy()
            frame_id = curr_query_info['frame_idx']
            
            # gt中信息
            curr_key_frame_info = key_frame_infos[frame_id].copy()
            curr_dt_boxes = curr_key_frame_info['boxes_lidar']
            curr_dt_label = curr_key_frame_info['pred_labels']
            curr_dt_score = curr_key_frame_info['score']
            curr_dt_names = curr_key_frame_info['name']
            curr_gt_boxes_3d = curr_key_frame_info['gt_boxes']
            
            # 排序
            sort_idx    = np.argsort(-curr_dt_score)
            curr_dt_boxes = curr_dt_boxes[sort_idx]
            curr_dt_score = curr_dt_score[sort_idx]
            curr_dt_label = curr_dt_label[sort_idx].astype(np.int32)
            curr_dt_names = curr_dt_names[sort_idx]
            
            curr_query_info.update({
                'tot_dt_boxes': curr_dt_boxes,
                'tot_dt_labels': curr_dt_label,
                'tot_dt_scores': curr_dt_score,
                'tot_dt_names': curr_dt_names,
                'tot_gt_boxes': curr_gt_boxes_3d,  # xyz lwh r vx vy cls_id = 10 | cls_id + 1
                
                # 'timestamp': curr_key_frame_info['frame_id'],
            })
            
            query_frame_infos_new.append(curr_query_info)
            
        return query_frame_infos_new
    
    def sparse_frame_infos_for_vis(self, frame_infos):
        """
        稀疏帧信息，只保留必要信息
        
        Args:
            frame_infos: 原始帧信息
            
        Returns:
            dict: 稀疏帧信息
        """
        
        frame_infos_new = []
        for frame_info in frame_infos:
            curr_frame_info = {
                # 帧ID
                # 'timestamp': frame_info['timestamp'],
                'frame_id': frame_info['frame_idx'],
                
                # 此帧全部预测和真值框
                'tot_dt_boxes': frame_info['tot_dt_boxes'],
                'tot_dt_labels': frame_info['tot_dt_labels'],
                'tot_dt_scores': frame_info['tot_dt_scores'],
                'tot_dt_names': frame_info['tot_dt_names'],
                'tot_gt_boxes': frame_info['tot_gt_boxes'],
                'tot_gt_boxes_label_ids': frame_info['tot_gt_boxes'][:, -1],
                
                
                # 分类别数据
                'classes_data': frame_info['classes_data'],
            }
            
            frame_infos_new.append(curr_frame_info)
        
        return frame_infos_new