import numpy as np


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    
    if len(recall) == 0 or len(precision) == 0:
        return 0.0
    
    epsilon = 1e-7  # 设置精度阈值
    
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # i = np.where(mrec[1:] != mrec[:-1])[0]
    seq = np.where(np.abs(mrec[1:] - mrec[:-1]) > epsilon)[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[seq + 1] - mrec[seq]) * mpre[seq + 1])
    return ap

def ap_per_class_with_curves(tp, conf, pred_cls, target_cls, class_names, precision_points=None):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
        class_names: List of class names.
        precision_points: List of recall values to compute precision at (e.g., [0.5, 0.6, 0.7, 0.8, 0.9])
    # Returns
        p: Final precision for each class
        r: Final recall for each class  
        ap: Average precision for each class
        f1: F1 score for each class
        pr_curves: Dictionary containing PR curves for each class
        precision_at_recall: Dictionary containing precision values at specified recall points
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(range(1, len(class_names)+1))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    pr_curves = {}  # 存储每个类别的PR曲线数据
    recall_at_precision = {}  # 存储每个类别的recall@precision
    if precision_points is None:
        precision_points = [0.5, 0.6, 0.7, 0.8, 0.9]  # 默认精度点
    
    # breakpoint()
    for c in unique_classes:
        curr_cls_mask = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = curr_cls_mask.sum()  # Number of predicted objects

        if n_p > 0:
            # 有预测时才计算累积TP和FP
            fpc = (1 - tp[curr_cls_mask]).cumsum()
            tpc = (tp[curr_cls_mask]).cumsum()
            conf_class = conf[curr_cls_mask]
        else:
            # 无预测时设置为空数组
            fpc = np.array([])
            tpc = np.array([])
            conf_class = np.array([])

        if n_p == 0 and n_gt == 0:
            # 情况1: 既没有预测也没有真实标签 - 该类别不存在
            ap.append(-1.0)  # 或者 np.nan，表示无法计算AP
            r.append(-1.0)
            p.append(-1.0)
            recall_curve = np.array([])
            precision_curve = np.array([])
            
        elif n_p == 0 and n_gt > 0:
            # 情况2: 有真实标签但没有预测 - 完全漏检
            ap.append(0.0)  # AP = 0，因为召回率为0
            r.append(0.0)   # 召回率 = 0
            p.append(-1.0)    # 精度无法定义（0/0）
            recall_curve = np.array([0.0])
            precision_curve = np.array([0.0])  # 可以设为 np.nan
            
        elif n_p > 0 and n_gt == 0:
            # 情况3: 有预测但没有真实标签 - 全部误检
            ap.append(0.0)  # AP = 0，因为精度为0
            r.append(-1.0)    # 召回率无法定义（0/0）
            p.append(0.0)   # 精度 = 0，所有预测都是FP
            recall_curve = np.array([0.0])     # 可以设为 np.nan
            precision_curve = tpc / (tpc + fpc)  # 实际计算，结果应该全为0
            
        else:
            # 情况4: 既有预测也有真实标签 - 正常计算
            # 使用之前计算的tpc和fpc
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

        # 统一存储PR曲线数据
        pr_curves[c] = {
            'recall': recall_curve,
            'precision': precision_curve,
            'confidence': conf_class,
            'num_Dt': n_p,
            'num_Gt': n_gt,
            'max_tp': tpc[-1] if len(tpc) > 0 else 0  # tpc的最大值，即该类别的TP总数
        }

        # 统一计算recall@precision
        recall_at_precision[c] = {}
        for p_val in precision_points:
            if len(precision_curve) == 0:
                # 没有预测数据
                recall_at_precision[c][f'R@P{p_val}'] = -1.0
            elif n_gt == 0:
                # 没有真实标签，recall无法定义
                recall_at_precision[c][f'R@P{p_val}'] = -1.0 if n_p == 0 else 0.0
            else:
                # 正常计算
                valid_indices = precision_curve >= p_val
                if np.any(valid_indices):
                    max_recall_at_precision = np.max(recall_curve[valid_indices])
                    recall_at_precision[c][f'R@P{p_val}'] = max_recall_at_precision
                else:
                    recall_at_precision[c][f'R@P{p_val}'] = 0.0

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, pr_curves, recall_at_precision