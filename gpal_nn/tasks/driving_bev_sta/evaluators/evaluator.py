import os
import bisect
import math
from collections import OrderedDict, defaultdict
import pickle

import numpy as np
import os.path as osp
import json
import logging
import shapely
import copy
from shapely.geometry import Polygon, Point
# import pandas as pd
from gpal_lightning.neural_network.tasks.base.evaluators.evaluator import \
    BaseEvaluator
from gpal_lightning.neural_network.tasks.builder import EVALUATORS
from gpal_nn.tasks.driving_bev_sta.evaluators.bevlane_evaluator import Bevlane_Evaluator


@EVALUATORS.register_module()
class DRIVING_BEV_STAEvaluator(BaseEvaluator):
    def __init__(self, global_config, task_config, print_to_terminal=False):
        super().__init__(global_config, task_config)
        self.pc_range = [0, -10.0, -2.0, 80.2, 10.2, 2.0]
        self.gt_range = [120, 16, 0, 0.0, -16.0, 0]

        self.pread_all = {}
        self.gt_all = []

    def generate_kpi(self) -> dict:
        evaloator = Bevlane_Evaluator()
        for thr in self.pread_all:
            print('-*' * 10 + f'score threshhold:{thr}' + '-*' * 10)
            metric_dict = evaloator.evaluate_single(
                copy.deepcopy(self.pread_all[thr]), copy.deepcopy(self.gt_all))
        return

    def format_gt_results(self, gt_list, gt_range=None):
        gt_all_data = []

        for gts in gt_list:
            gt_lanes = {
                'vectors': []
            }
            for gt in gts:
                gt_lane = {
                    'pts': [e[:2] for e in gt['pts']],
                    'type': 0,
                    'cls_name': 'normal'
                }
                gt_lanes['vectors'].append(gt_lane)
            gt_all_data.append(gt_lanes)

        return gt_all_data

    def get_annotations(self, results=None):
        anno_list = []
        for data in results:
            anno_list.append(data['gt_vectors'])
        gt = self.format_gt_results(anno_list, self.gt_range)

        return gt

    def get_dets(self, results, thr=0.0):
        dt_all_data = []

        for result in results:
            result['vectors'] = [e for e in result['vectors']]
            dt_all_data.append(result)

        return dt_all_data

    def compute_metrics(self, pred, true, epoch=0):
        """Compute the metrics from processed results.
        Args:
            results (List[dict]): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # 车道线

        gt = self.get_annotations(true)

        # for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        for thr in [0.35, ]:

            dt = self.get_dets(pred, thr=thr)
            if thr not in self.pread_all:
                self.pread_all[thr] = []
            self.pread_all[thr] += dt
        self.gt_all += gt

        return

    def process(self, pred: dict, true: dict, metadata: dict) -> None:
        self.compute_metrics(pred, true)
        pass
