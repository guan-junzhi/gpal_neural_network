import os
import cv2
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from glob import glob
import json
import pickle as pkl
import shutil
from tqdm import tqdm
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *
from tools_scripts.data_format_cvt import ShowDataStruct


def load_json(path, by_list=False):
    with open(path, 'r') as f:
        if by_list:
            data = []
            for line in f.readlines():
                data.append(json.loads(line.strip()))
        else:
            data = json.load(f)
    return data


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


class Bevlane_Evaluator(object):
    def __init__(self):
        self.code_size = 2
        self.num_pts_per_vec = 20
        self.pc_range = [0, -10.0, -2.0, 80.2, 10.2, 2.0]
        self.score_threshold = None
        self.main_classes = list(main_class_type_map.keys())
        self.eval_use_same_gt_sample_num_flag = True
        self.metric = 'chamfer'

    def format_eval_pred(self, preds, score_thr=0.35):
        pred_data = load_json(preds, by_list=True)
        pred_results = {
            'meta': {
                'use_lidar': False, 'use_camera': True,
                'use_radar': False, 'use_map': False, 'use_external': True
            },
            'results': [{'vectors': [f for f in pred['vectors'] if f["confidence_level"] >= score_thr]} for pred in
                        pred_data]
        }
        path = preds.replace('.json', '_hozon_map_results.json').replace(
            'input/', 'work_dirs/')
        self.write(pred_results, path)
        return path

    # def evaluate(self, pred_file, gt_file):
    #     # format_pred_file = self.format_eval_pred(pred_file, 0.35)
    #     format_pred_file = self.format_eval_pred(pred_file, 0.0)
    #     results_dict = self._evaluate_single(format_pred_file, gt_file, self.metric)
    #     return results_dict

    def evaluate(self, pred_file, gt_file):
        # format_pred_file = self.format_eval_pred(pred_file, 0.35)
        format_pred_file = self.format_eval_pred(pred_file, 0.0)
        results_dict = self._evaluate_single(
            format_pred_file, gt_file, self.metric)
        return results_dict

    def get_map_summary(self, mean_ap,
                        results_dict,
                        class_name=None,
                        thr=None):

        df_all = {}
        res_logs = {}
        for c_name, results in results_dict.items():
            # print("##"*20,c_name)

            if isinstance(results[0]['ap'], np.ndarray):
                num_scales = len(results[0]['ap'])
            else:
                num_scales = 1

            num_roi = len(results)

            recalls = np.zeros((num_scales, num_roi), dtype=np.float32)
            precisions = np.zeros((num_scales, num_roi), dtype=np.float32)
            aps = np.zeros((num_scales, num_roi), dtype=np.float32)
            num_gts = np.zeros((num_scales, num_roi), dtype=int)
            for i, roi_result in enumerate(results):
                if roi_result['recall'].size > 0:
                    recalls[:, i] = np.array(
                        roi_result['recall'], ndmin=2)[:, -1]
                    precisions[:, i] = np.array(
                        roi_result['precision'], ndmin=2)[:, -1]
                aps[:, i] = roi_result['ap']
                num_gts[:, i] = roi_result['num_gts']

            rois = [r['roi'] for r in results]

            if not isinstance(mean_ap, list):
                mean_ap = [mean_ap]

            header = ['gts', 'dets', 'Recall', 'Precision', 'AP', 'Dist', 'Dist@95', 'ShapeTypeAcc']
            for i in range(num_scales):
                table_data = []
                index = []
                res_log = {}
                for j in range(num_roi):
                    res_log[rois[j]] = {
                        "recall": recalls[i, j],
                        "precision": precisions[i, j],
                        "aps": aps[i, j],
                        "mean_dist_error": results[j]['mean_dist_error'],
                        "dist_error_95": results[j]['dist_error_95'],
                        "shape_type_acc": results[j]["shape_type_acc"],
                    }

                    row_data = [
                        num_gts[i, j], results[j]['num_dets'],
                        f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}', f'{aps[i, j]:.3f}',
                        f'{results[j]["mean_dist_error"]:.3f}', f'{results[j]["dist_error_95"]:.3f}',
                        f'{results[j]["shape_type_acc"]:.3f}',
                    ]
                    index.append(rois[j])
                    table_data.append(row_data)

                df_result = pd.DataFrame(
                    table_data, index=index, columns=header)
                # print(df_result)
                df_all[c_name] = df_result
                res_logs[c_name] = res_log
        return df_all, res_logs

    def evaluate_single(self, gen_results, annotations, metric='chamfer'):
        from .mean_ap import eval_map
        from .mean_ap import format_res_gt_by_classes
        cls_gens, cls_gts = format_res_gt_by_classes(gen_results,
                                                     annotations,
                                                     cls_names=self.main_classes,
                                                     num_pred_pts_per_instance=self.num_pts_per_vec,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range, code_size=self.code_size)
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            print('-*' * 10 + f'metric:{metric}' + '-*' * 10)

            if metric == 'chamfer':
                # thresholds = [-0.5,-1.0,-1.5]
                thresholds = [-0.7]
            elif metric == 'iou':
                thresholds = np.linspace(.5, 0.95, int(
                    np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), len(self.main_classes)))

            for i, thr in enumerate(thresholds):
                print('-*' * 10 + f'threshhold:{thr}' + '-*' * 10)
                mAP, cls_ap = eval_map(
                    gen_results,
                    annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.main_classes,
                    logger=None,
                    num_pred_pts_per_instance=self.num_pts_per_vec,
                    pc_range=self.pc_range,
                    metric=metric,
                    code_size=self.code_size)

                # 每个阈值下结果
                # cls_ap [{},{}]分别存了每个roi下的指标
                r_df, r_logs = self.get_map_summary(mAP, cls_ap, class_name=list(main_class_type_map.keys()), thr=thr)
                df_index = []
                for k_df, v_df in r_df.items():
                    df_index.append([k_df] + [''] * (len(v_df) - 1))

                df_index = [ii for i in df_index for ii in i]
                df_data = pd.concat(list(r_df.values()))

                m_index = pd.MultiIndex.from_arrays(
                    [df_index, df_data.index.tolist()])
                df_data.set_index(m_index, inplace=True)

                print(df_data)

        return r_logs

    def write(self, res, file):
        dirname = os.path.dirname(file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(file, 'w') as f:
            if isinstance(res, list):
                for line in res:
                    f.write(line + '\n')
            elif isinstance(res, dict):
                res = json.dumps(res)
                f.write(res)
            else:
                f.write(res)
        print(f'---- write file: {file}')
        return
