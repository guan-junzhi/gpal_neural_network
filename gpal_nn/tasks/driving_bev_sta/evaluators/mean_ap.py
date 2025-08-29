# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool
from shapely.geometry import LineString, Polygon
import numpy as np
from .utils import print_log, Timer, dump
from terminaltables import AsciiTable
import json
from os import path as osp
import os
from functools import partial
from .tpfp import custom_tpfp_gen
from .tgfg import tpfp_gen
from gpal_nn.tasks.driving_bev_sta.datasets.collect import _fix_pts_interpolate


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def get_roi_points(p_list, rois=["0-10", "10-30", "30-50", "50-80", "80-100"]):
    np_p_list = np.array(p_list)
    p_list = _fix_pts_interpolate(np_p_list, int(LineString(np_p_list).length / 0.1)).tolist()  # 先稠密化再分区域
    p_list = sorted(p_list, key=lambda x: x[0])  # 升序

    # import pdb;pdb.set_trace()
    roi_points = {}
    for roi in rois:
        s_tmp = roi.split("-")
        s_tmp = [int(t) for t in s_tmp]
        if len(s_tmp) == 2:
            x_start, x_end = s_tmp

            roi_data = [p for p in p_list if x_start <= p[0] < x_end]
            roi_points[roi] = roi_data

    return roi_points


def get_cls_results_roi(gen_results,
                        annotations,
                        num_sample=100,
                        num_pred_pts_per_instance=30,
                        eval_use_same_gt_sample_num_flag=True,
                        class_id=0,
                        fix_interval=False,
                        code_size=2,
                        rois=["0-10", "10-30", "30-50", "50-80"]):
    """Get det results and gt information of a certain class.

    Args:
        gen_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    # if len(gen_results) == 0 or

    cls_gens, cls_scores, shape_types = [], [], []
    # import pdb;pdb.set_trace()
    # if len(rois) > 1:
    #     num_sample = num_sample * len(rois)
    for res in gen_results['vectors']:
        if res['type'] == class_id:
            if len(res['pts']) < 2:
                continue

            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res['pts'])
            else:
                line = res['pts']

                roi_lines = get_roi_points(line, rois)
                roi_lines_ = {}

                for roi, line in roi_lines.items():
                    if line and len(line) >= 2:
                        line = LineString(line)
                        if fix_interval:
                            distances = list(np.arange(1., line.length, 1.))
                            distances = [0, ] + distances + [line.length, ]
                            sampled_points = np.array([list(line.interpolate(distance).coords)
                                                       for distance in distances]).reshape(len(distances), -1)
                        else:
                            # there
                            distances = np.linspace(0, line.length, num_sample)
                            sampled_points = np.array([list(line.interpolate(distance).coords)
                                                       for distance in distances]).reshape(len(distances), -1)

                    else:
                        sampled_points = []
                    roi_lines_[roi] = sampled_points

            # 这一帧的某一条车道线
            cls_gens.append(roi_lines_)  # [{},{},{}] 每一条线按roi划分
            cls_scores.append(res['confidence_level'])
            shape_types.append(res['shape_type'])

    # 处理cls_gens 生成roi车道线 1->5

    # import pdb;pdb.set_trace()
    cls_gts = []
    shape_types_gt = []
    for ann in annotations['vectors']:
        if ann['type'] == class_id:
            line = ann['pts']
            roi_lines_gt = get_roi_points(line, rois)
            roi_lines_gt_ = {}
            for roi, line in roi_lines_gt.items():
                if line and len(line) >= 2:
                    line = LineString(line)
                    distances = np.linspace(0, line.length, num_sample)
                    sampled_points = np.array([list(line.interpolate(distance).coords)
                                               for distance in distances]).reshape(-1, code_size)
                    roi_lines_gt_[roi] = sampled_points
                # else:
                #     sampled_points = []

                # if isinstance(sampled_points,list):
                #    continue

                # roi_lines_gt[roi] = sampled_points

            cls_gts.append(roi_lines_gt_)
            shape_types_gt.append(ann['shape_type'])

    roi_gen_dict = {}
    roi_gt_dict = {}

    for roi in rois:

        cls_gens_ = [i[roi] for i in cls_gens]
        cls_scores_ = []
        shape_types_ = []

        for i, cg in enumerate(cls_gens_):
            if isinstance(cg, np.ndarray):
                cls_scores_.append(cls_scores[i])
                shape_types_.append(shape_types[i])
        cls_gens_ = [i for i in cls_gens_ if isinstance(i, np.ndarray)]
        num_res = len(cls_gens_)
        if num_res > 0:
            cls_gens_ = np.stack(cls_gens_).reshape(num_res, -1)
            cls_scores_ = np.array(cls_scores_)[:, np.newaxis]
            cls_shape_types_ = np.array(shape_types_)[:, np.newaxis]
            # print(roi,cls_gens_.shape,cls_scores_.shape)
            # import pdb;pdb.set_trace()
            cls_gens_ = np.concatenate([cls_gens_, cls_scores_, cls_shape_types_], axis=-1)

        else:
            if not eval_use_same_gt_sample_num_flag:
                cls_gens_ = np.zeros((0, num_pred_pts_per_instance * code_size + 2))
            else:
                cls_gens_ = np.zeros((0, num_sample * code_size + 2))
            # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')
        roi_gen_dict[roi] = cls_gens_

        # gt
        cls_gts_ = [i[roi] for i in cls_gts if roi in i]
        cls_gts_ = [i for i in cls_gts_ if isinstance(i, np.ndarray)]
        shape_types_gt_ = [shape_types_gt[i] for i in range(len(cls_gts)) if roi in cls_gts[i]]
        num_gts = len(cls_gts_)
        if num_gts > 0:
            # print([gg for gg in cls_gts_ if not isinstance(gg,np.ndarray)])

            cls_gts_ = np.stack(cls_gts_).reshape(num_gts, -1)
            shape_types_gt_ = np.array(shape_types_gt_)[:, np.newaxis]
            cls_gts_ = np.concatenate([cls_gts_, shape_types_gt_], axis=-1)

        else:
            cls_gts_ = np.zeros((0, num_sample * code_size + 1))
        roi_gt_dict[roi] = cls_gts_

    return roi_gen_dict, roi_gt_dict


def get_cls_results(gen_results,
                    annotations,
                    num_sample=100,
                    num_pred_pts_per_instance=30,
                    eval_use_same_gt_sample_num_flag=False,
                    class_id=0,
                    fix_interval=False,
                    code_size=2):
    """Get det results and gt information of a certain class.

    Args:
        gen_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    # if len(gen_results) == 0 or

    cls_gens, cls_scores = [], []
    for res in gen_results['vectors']:
        if res['type'] == class_id:
            if len(res['pts']) < 2:
                continue
            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res['pts'])
            else:
                line = res['pts']
                line = LineString(line)

                if fix_interval:
                    distances = list(np.arange(1., line.length, 1.))
                    distances = [0,] + distances + [line.length,]
                    sampled_points = np.array([list(line.interpolate(distance).coords)
                                               for distance in distances]).reshape(len(distances), -1)
                else:
                    distances = np.linspace(0, line.length, num_sample)
                    sampled_points = np.array([list(line.interpolate(distance).coords)
                                               for distance in distances]).reshape(len(distances), -1)
            cls_gens.append(sampled_points)
            cls_scores.append(res['confidence_level'])
    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res, -1)
        cls_scores = np.array(cls_scores)[:, np.newaxis]
        cls_gens = np.concatenate([cls_gens, cls_scores], axis=-1)
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')
    else:
        if not eval_use_same_gt_sample_num_flag:
            cls_gens = np.zeros((0, num_pred_pts_per_instance*code_size+1))
        else:
            cls_gens = np.zeros((0, num_sample*code_size+1))
        # print(f'for class {i}, cls_gens has shape {cls_gens.shape}')

    cls_gts = []
    for ann in annotations['vectors']:
        if ann['type'] == class_id:
            # line = ann['pts'] +  np.array((1,1)) # for hdmapnet
            line = ann['pts']
            # line = ann['pts'].cumsum(0)
            line = LineString(line)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, code_size)

            cls_gts.append(sampled_points)
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts, -1)
    else:
        cls_gts = np.zeros((0, num_sample*code_size))
    return cls_gens, cls_gts
    # ones = np.ones((num_gts,1))
    # tmp_cls_gens = np.concatenate([cls_gts,ones],axis=-1)
    # return tmp_cls_gens, cls_gts


def format_res_gt_by_classes(gen_results,
                             annotations,
                             cls_names=None,
                             num_pred_pts_per_instance=30,
                             eval_use_same_gt_sample_num_flag=False,
                             pc_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
                             nproc=24, code_size=2):
    assert cls_names is not None
    timer = Timer()
    num_fixed_sample_pts = 100
    fix_interval = False

    assert len(gen_results) == len(annotations)

    pool = Pool(nproc)
    cls_gens, cls_gts = {}, {}

    for i, clsname in enumerate(cls_names):
        gengts = pool.starmap(
            partial(get_cls_results_roi, num_sample=num_fixed_sample_pts,
                    num_pred_pts_per_instance=num_pred_pts_per_instance,
                    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag, class_id=i,
                    fix_interval=fix_interval, code_size=code_size),
            zip(gen_results, annotations))

        # import pdb;pdb.set_trace()
        gens, gts = tuple(zip(*gengts))
        cls_gens[clsname] = gens
        cls_gts[clsname] = gts
    pool.close()
    return cls_gens, cls_gts


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (np.array).
        precision: The precision curve (np.array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # 错位比较，前一个元素与其后一个元素比较,np.where()返回下标索引数组组成的元组
    i = np.where(mrec[:-1] != mrec[1:])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_map(gen_results,
             annotations,
             cls_gens,
             cls_gts,
             threshold=0.5,
             cls_names=None,
             logger=None,
             tpfp_fn=None,
             pc_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
             metric=None,
             num_pred_pts_per_instance=30,
             nproc=1,  # 24
             code_size=2):
    timer = Timer()
    pool = Pool(nproc)

    eval_results = {}

    for i, clsname in enumerate(cls_names):

        # get gt and det bboxes of this class
        cls_gen_roi = cls_gens[clsname]
        cls_gt_roi = cls_gts[clsname]

        roi_result = []
        for roi in ["0-10", "10-30", "30-50", "50-80"]:
            cls_gt = [i[roi] for i in cls_gt_roi]
            cls_gen = [i[roi] for i in cls_gen_roi]
            # choose proper function according to datasets to compute tp and fp
            # XXX
            # func_name = cls2func[clsname]
            # tpfp_fn = tpfp_fn_dict[tpfp_fn_name]
            # tpfp_fn = custom_tpfp_gen
            tpfp_fn = tpfp_gen
            # Trick for serialized
            # only top-level function can be serized
            # somehow use partitial the return function is defined
            # at the top level.

            # tpfp = tpfp_fn(cls_gen[i], cls_gt[i],threshold=threshold, metric=metric)
            # import pdb; pdb.set_trace()
            # TODO this is a hack
            tpfp_fn = partial(tpfp_fn, threshold=threshold,
                              metric=metric, coord_dim=code_size)
            args = []
            # compute tp and fp for each image with multiple processes
            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_gen, cls_gt, *args))

            # import pdb;pdb.set_trace()
            tp, fp, dist_error_tuple, shape_type_acc = tuple(zip(*tpfp))
            dist_error_list = []
            shape_type_acc_list = []
            for dist in dist_error_tuple:
                dist_error_list.extend(dist)
            for acc in shape_type_acc:
                shape_type_acc_list.extend(acc)

            mean_dist_error = np.nanmean(dist_error_list)
            dist_error_95 = np.nanpercentile(dist_error_list, 95)
            shape_type_acc = np.sum(shape_type_acc_list) / len(shape_type_acc_list)

            # map_results = map(
            #     tpfp_fn,
            #     cls_gen, cls_gt)
            # tp, fp = tuple(map(list, zip(*map_results)))

            # debug and testing
            # for i in range(len(cls_gen)):
            #     # print(i)
            #     tpfp = tpfp_fn(cls_gen[i], cls_gt[i],threshold=threshold)
            #     print(i)
            #     tpfp = (tpfp,)
            #     print(tpfp)
            # i = 0
            # tpfp = tpfp_fn(cls_gen[i], cls_gt[i],threshold=threshold)
            # import pdb; pdb.set_trace()

            # XXX

            num_gts = 0
            for j, bbox in enumerate(cls_gt):
                num_gts += bbox.shape[0]

            # sort all det bboxes by score, also sort tp and fp
            # import pdb;pdb.set_trace()
            cls_gen = np.vstack(cls_gen)
            num_dets = cls_gen.shape[0]
            # descending, high score front
            sort_inds = np.argsort(-cls_gen[:, -1])
            tp = np.hstack(tp)[sort_inds]
            fp = np.hstack(fp)[sort_inds]

            # calculate recall and precision with tp and fp
            # num_det*num_res
            tp = np.cumsum(tp, axis=0)
            fp = np.cumsum(fp, axis=0)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts, eps)
            precisions = tp / np.maximum((tp + fp), eps)

            # calculate AP
            # if dataset != 'voc07' else '11points'
            mode = 'area'
            ap = average_precision(recalls, precisions, mode)

            ap_ = compute_ap(recalls, precisions)

            # print("ap = {} ap_ = {}".format(ap,ap_))
            roi_result.append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap,
                'roi': roi,
                'mean_dist_error':mean_dist_error,
                'dist_error_95':dist_error_95,
                'shape_type_acc':shape_type_acc

            })
            # print('cls:{} done in {:2f}s!!'.format(clsname,float(timer.since_last_check())))

        # eval_results.append(roi_result)  ###每个class下各个roi的结果
        eval_results[clsname] = roi_result
    pool.close()

    # print(eval_results)
    mean_aps = []
    for cls_name, cls_result in eval_results.items():
        aps = [c["ap"] for c in cls_result if c["num_gts"] > 0]
        mean_ap = np.array(aps).mean().item() if len(aps) else 0.0
        mean_aps.append(mean_ap)

    # print_map_summary(
    #     mean_aps, eval_results, class_name=cls_names, logger=logger)

    # print(eval_results)
    return mean_aps, eval_results


def print_map_summary(mean_ap,
                      results,
                      class_name=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    for mean_ap, results in zip(mean_ap, results):
        if isinstance(results[0]['ap'], np.ndarray):
            num_scales = len(results[0]['ap'])
        else:
            num_scales = 1

        if scale_ranges is not None:
            assert len(scale_ranges) == num_scales

        num_classes = len(results)

        recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
        aps = np.zeros((num_scales, num_classes), dtype=np.float32)
        num_gts = np.zeros((num_scales, num_classes), dtype=int)
        for i, cls_result in enumerate(results):
            if cls_result['recall'].size > 0:
                recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            aps[:, i] = cls_result['ap']
            num_gts[:, i] = cls_result['num_gts']

        label_names = class_name

        if not isinstance(mean_ap, list):
            mean_ap = [mean_ap]

        header = ['class', 'gts', 'dets', 'recall', 'ap']

        print("#"*5, class_name, "#"*5, results["roi"])
        for i in range(num_scales):
            if scale_ranges is not None:
                print_log(f'Scale range {scale_ranges[i]}', logger=logger)
            table_data = [header]
            for j in range(num_classes):
                row_data = [
                    label_names[j], num_gts[i, j], results[j]['num_dets'],
                    f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
                ]
                table_data.append(row_data)
            table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
            table = AsciiTable(table_data)
            table.inner_footing_row_border = True
            print_log('\n' + table.table, logger=logger)


def get_map_summary(mean_ap,
                    results,
                    class_name=None,
                    thr=None):

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(
                cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = class_name

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['gts', 'dets', 'recall', 'precision', 'ap']
    for i in range(num_scales):
        table_data = []
        index = []
        for j in range(num_classes):
            row_data = [
                num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{precisions[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            index.append(label_names[j])
            table_data.append(row_data)
        table_data.append(['', '', '', '', f'{mean_ap[i]:.3f}'])
        index.append('mAP')
        print("table_data = {}".format(table_data))
        import pandas as pd

        df_result = pd.DataFrame(table_data, index=index, columns=header)
        df_result.columns.name = "threshold:{}".format(thr)
        print(df_result)
        return df_result
