import numpy as np
from gpal_lightning.neural_network.tasks.base.evaluators.evaluator import \
    BaseEvaluator
from gpal_lightning.neural_network.tasks.builder import EVALUATORS
from gpal_nn.tasks.parking_ipm_sta.datasets.txtlabel_instance_p3 import TXTLabelLoader

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

@EVALUATORS.register_module()
class PARKING_IPM_STAEvaluator(BaseEvaluator):
    def __init__(self, global_config, task_config, print_to_terminal=False):
        super().__init__(global_config, task_config)
        # self.pc_range = [0, -10.0, -2.0, 80.2, 10.2, 2.0]
        # self.gt_range = [120, 16, 0, 0.0, -16.0, 0]

        self.pread_all = []
        self.gt_all = []
        self.load_from = global_config.load_from

    def generate_kpi(self) -> dict:
        StatPackage = initStatPack()
        for pred, gt in zip(self.pread_all, self.gt_all):
            evaloator = TXTLabelLoader(self.sw, self.sh)
            heatmapResultPack = evaloator.errorCaculate(pred, gt)
       
            updatePack(StatPackage, heatmapResultPack)
        outputStat(StatPackage, self.load_from)
        return

    def compute_metrics(self, pred, true, epoch=0):
        """Compute the metrics from processed results.
        Args:
            results (List[dict]): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        self.pread_all += pred
        self.gt_all += true

        return

    def process(self, pred: dict, true: dict, metadata: dict) -> None:
        
        self.sw, self.sh = metadata[0]['sw_sh']
        
        self.compute_metrics(pred, true)
        pass
