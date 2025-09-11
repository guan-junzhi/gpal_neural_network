import os
import cv2
import numpy as np
from gpal_lightning.neural_network.tasks.base.evaluators.evaluator import \
    BaseEvaluator
from gpal_lightning.neural_network.tasks.builder import EVALUATORS
from gpal_nn.tasks.radar4d_nn_sdk.datasets.txtlabel_instance_p3 import TXTLabelLoader


def initStatPack():
    StatPack = {}
    StatPack['point_pixel_error_sum'] = 0.0
    StatPack['angle_error'] = 0.0
    StatPack['point_total_num'] = 0
    StatPack['point_true_num'] = 0
    StatPack['point_miss_num'] = 0
    StatPack['point_false_num'] = 0
    StatPack['line_total_num'] = 0
    StatPack['line_true_num'] = 0
    StatPack['line_miss_num'] = 0
    StatPack['line_false_num'] = 0
    return StatPack


def updatePack(StatPack, resultPack):
    print("StatPack['point_pixel_error_sum'] is: {}; resultPack['point_error'] is: {}".format(
        StatPack['point_pixel_error_sum'], resultPack['point_error']))
    print("StatPack['angle_error'] is: {}; resultPack['angle_error'] is: {}".format(
        StatPack['angle_error'], resultPack['angle_error']))
    print("StatPack['point_total_num'] is: {}; resultPack['point_total_num'] is: {}".format(
        StatPack['point_total_num'], resultPack['point_total_num']))
    print("StatPack['point_true_num'] is: {}; resultPack['point_true_num'] is: {}".format(
        StatPack['point_true_num'], resultPack['point_true_num']))
    print("StatPack['point_miss_num'] is: {}; resultPack['point_miss_num'] is: {}".format(
        StatPack['point_miss_num'], resultPack['point_miss_num']))
    print("StatPack['point_false_num'] is: {}; resultPack['point_false_num'] is: {}".format(
        StatPack['point_false_num'], resultPack['point_false_num']))
    print("StatPack['line_total_num'] is: {}; resultPack['line_total_num'] is: {}".format(
        StatPack['line_total_num'], resultPack['line_total_num']))
    print("StatPack['line_true_num'] is: {}; resultPack['line_true_num'] is: {}".format(
        StatPack['line_true_num'], resultPack['line_true_num']))
    print("StatPack['line_miss_num'] is: {}; resultPack['line_miss_num'] is: {}".format(
        StatPack['line_miss_num'], resultPack['line_miss_num']))
    print("StatPack['line_false_num'] is: {}; resultPack['line_false_num'] is: {}".format(
        StatPack['line_false_num'], resultPack['line_false_num']))

    StatPack['point_pixel_error_sum'] = StatPack['point_pixel_error_sum'] + \
        resultPack['point_error']
    StatPack['angle_error'] = StatPack['angle_error'] + \
        resultPack['angle_error']
    StatPack['point_total_num'] = StatPack['point_total_num'] + \
        resultPack['point_total_num']
    StatPack['point_true_num'] = StatPack['point_true_num'] + \
        resultPack['point_true_num']
    StatPack['point_miss_num'] = StatPack['point_miss_num'] + \
        resultPack['point_miss_num']
    StatPack['point_false_num'] = StatPack['point_false_num'] + \
        resultPack['point_false_num']
    StatPack['line_total_num'] = StatPack['line_total_num'] + \
        resultPack['line_total_num']
    StatPack['line_true_num'] = StatPack['line_true_num'] + \
        resultPack['line_true_num']
    StatPack['line_miss_num'] = StatPack['line_miss_num'] + \
        resultPack['line_miss_num']
    StatPack['line_false_num'] = StatPack['line_false_num'] + \
        resultPack['line_false_num']


def printTitleInfo(weight_name):
    print("")
    print("--------------Slot Det Model Auto Test On Image DataSet---------------")
    print("Algorithom vrsion : ", 'torch v0')
    print("NetWork Description : ", " (ddrnet_23_slim)")
    print("trained model name : ", weight_name)
    # print ""


def outputStat(StatPack, weight_name):
    StatPack['point_pixel_error_sum'] = StatPack['point_pixel_error_sum'] / \
        StatPack['point_total_num']
    StatPack['angle_error'] = StatPack['angle_error'] / \
        StatPack['line_total_num']
    StatPack['point_recall'] = StatPack['point_true_num'] * \
        1.0 / StatPack['point_total_num']
    StatPack['point_false_rate'] = StatPack['point_false_num']*1.0 / \
        (StatPack['point_false_num'] + StatPack['point_true_num'])
    StatPack['line_recall'] = StatPack['line_true_num'] * \
        1.0 / StatPack['line_total_num']
    StatPack['line_false_rate'] = StatPack['line_false_num']*1.0 / \
        (StatPack['line_false_num'] + StatPack['line_true_num'])
    printTitleInfo(weight_name)
    print("--->point count result : ")
    print("total point numbers = ", StatPack['point_total_num'], " | point recall = ",
          StatPack['point_recall'], " | point false rate = ", StatPack['point_false_rate'])
    print("point average pixel error : ", StatPack['point_pixel_error_sum'])
    print("--->line count result : ")
    print("total line numbers = ", StatPack['line_total_num'], " | line recall = ",
          StatPack['line_recall'], " | line false rate = ", StatPack['line_false_rate'])
    print("point line angle error degree : ",
          np.degrees(StatPack['angle_error']))
    print("--------------------down------------------------")
    return


def DrawVe(gt, pred, meta, save_path):
    image_f = meta['last_img_path']
    w, h = meta['wh']

    img = cv2.imread(image_f, cv2.IMREAD_COLOR)
    mask = cv2.resize(img, (w, h))

    h, w, c = mask.shape
    for i in range(len(pred)):
        point = pred[i][0]
        orients = pred[i][1]
        for j in range(len(orients)):
            ori = orients[j]
            stp = (point[0], point[1])
            edp = (int(stp[0] + ori[0]*ori[3]),
                   int(stp[1] + ori[1]*ori[3]))
            cv2.line(mask, stp, edp, (0, 250, 0), 2)
            # mdp = ((edp[0]+stp[0])/2 , (edp[1]+stp[1])/2)
            # line_label = 'p' + str(i) + '_l' + str(j)
            # cv2.putText(mask, line_label, mdp, 1, 0.8, (0,0,0), 1)
    for i in range(len(pred)):
        point = pred[i][0]
        cv2.circle(mask, (point[0], point[1]), 2, (0, 0, 250), -1)
        dx = 5
        if point[0] > w/2:
            dx = -15
        mdp = (point[0] + dx, point[1] + 5)
        # point_label = 'p' + str(i)
        # cv2.putText(mask, point_label, mdp, 1, 0.8, (0,0,200), 1)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, image_f.split('/')[-1]), mask)


@EVALUATORS.register_module()
class RADAR4D_NN_SDKEvaluator(BaseEvaluator):
    def __init__(self, global_config, task_config, print_to_terminal=False):
        super().__init__(global_config, task_config)
        # self.pc_range = [0, -10.0, -2.0, 80.2, 10.2, 2.0]
        # self.gt_range = [120, 16, 0, 0.0, -16.0, 0]

        self.pread_all = []
        self.gt_all = []
        self.meta_all = []
        self.load_from = global_config.load_from
        self.save = os.path.join(global_config.save, "detect_res")

    def generate_kpi(self) -> dict:
        StatPackage = initStatPack()
        for pred, gt, meta in zip(self.pread_all, self.gt_all, self.meta_all):
            DrawVe(gt, pred, meta, self.save)
            evaloator = TXTLabelLoader(self.sw, self.sh)
            heatmapResultPack = evaloator.errorCaculate(pred, gt)
            updatePack(StatPackage, heatmapResultPack)
        outputStat(StatPackage, self.load_from)
        return

    def compute_metrics(self, pred, true, metadata):
        """Compute the metrics from processed results.
        Args:
            results (List[dict]): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        self.pread_all += pred
        self.gt_all += true
        self.meta_all += metadata

        return

    def process(self, pred: dict, true: dict, metadata: dict) -> None:
        self.sw, self.sh = metadata[0]['sw_sh']
        self.compute_metrics(pred, true, metadata)
        pass
