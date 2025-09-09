import os
import cv2
import copy
import numpy as np
from gpal_lightning.neural_network.tasks.base.evaluators.evaluator import \
    BaseEvaluator
from gpal_lightning.neural_network.tasks.builder import EVALUATORS
# from gpal_nn.tasks.driving_bev_dyn.datasets.txtlabel_instance_p3 import TXTLabelLoader
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.evaluators.evaluation_node import ObjectDetectionEvaluator
from tqdm import tqdm
import json

def evaluation(preds, gts, metas, class_names, result_dir="workspace/20250907_08_44_34/od_eval_result"):
    # if 'annos' not in self.infos[0].keys():
    #     return 'No ground-truth boxes for evaluation', {}
    # gt 需要去除补全框
    for det_anno in gts:
        curr_gt_boxes_3d_ = det_anno['gt_boxes']
        cur_gt = curr_gt_boxes_3d_.copy()
        cur_gt = np.array(cur_gt)
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        curr_gt_boxes_3d = cur_gt[:k + 1]
        det_anno['gt_boxes'] = curr_gt_boxes_3d
        if curr_gt_boxes_3d.shape[0] == 0:
            det_anno['gt_boxes'] = np.empty(shape=(0, 9), dtype=np.float32)
    
    for d in preds:
        for k in d:
            d[k] = np.array(d[k])

    class_names = [i[8:(8+13)] for i in class_names]
    evaluator = ObjectDetectionEvaluator(
        class_names=class_names,
            det_range_list=[
                [-30, -10, -2, 30, 10, 4],
                [-50, -10, -2, 50, 10, 4],

                [-30, -15, -2, 30, 15, 4],
                [-50, -15, -2, 50, 15, 4],

                [-30, -30, -2, 30, 30, 4],
                [-50, -30, -2, 50, 30, 4],
                [-70, -30, -2, 70, 30, 4],
                [-75.2/1, -75.2, -2, 75.2/1, 75.2, 4],

                [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0],
            ],

        # slice_line_len = 295,
        # data_print_len = 12,
        # class_name_print_len = 20,

            r_at_p=0.7,
            precision_points=[0.5, 0.6, 0.7, 0.8, 0.9],

            distance_threshold_list=[1.0],
            restricted_ratio=[0.05, 0.005],
    )
    # exit(1)
    save_dir = result_dir
    use_print_format = 2
    os.makedirs(save_dir, exist_ok=True)

    log_file = f'{save_dir}/record_logs_{use_print_format}.log'
    # logger = create_logger(log_file=log_file, rank=0, log_level=logging.INFO)

    det_annos = copy.deepcopy(preds)
    for i in range(len(gts)):
        det_annos[i].update(gts[i])
    # det_annox = evaluator.load_det_annos(
    #     det_annos)  # 文件路径 or list[dict, dict, ...]

    evaluator.evaluate(det_annos, loggerinfo=print, save_dir=save_dir,
                        is_print_during_info=False, use_print_format=use_print_format)
    # test_dynamic_thresholds(loggerinfo=logger.info)

    ap_result_str = ''
    ap_dict = {}

    return ap_result_str, ap_dict


@EVALUATORS.register_module()
class DRIVING_BEV_DYNEvaluator(BaseEvaluator):
    def __init__(self, global_config, task_config, print_to_terminal=False):
        super().__init__(global_config, task_config)

        self.pread_all = []
        self.gt_all = []
        self.meta_all = []
        # self.load_from = global_config.load_from
        # self.save = os.path.join(global_config.save, "detect_res")

        self.class_name = ["_".join(ele) for ele in list(task_config.class_dict.values())]
        self.output_dir = os.path.join(
            self.global_config.dump_path, "od_eval_result")
        os.makedirs(self.output_dir, exist_ok = True)

    def generate_kpi(self) -> dict:
        # print(len(self.pread_all), len(self.gt_all), len(self.meta_all))

        # import pickle as pkl

        # pkl.dump((self.pread_all, self.gt_all, self.meta_all, self.class_name), open(
        #     f"evaluator.pkl", 'wb'))
        # # exit(1)

        evaluation(self.pread_all, self.gt_all, self.meta_all,
                   self.class_name, self.output_dir)


        
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

    def process(self, pred: dict, true: dict, metadata: dict) -> None:
        # print(ShowDataStruct("pred", pred, 2, 2))
        # print(ShowDataStruct("true", true, 2, 2))
        self.compute_metrics(pred, true, metadata)
        

if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("evaluator.pkl", 'rb'))

    print(ShowDataStruct("inputs", inputs, 2, 4))

    root_dir = "workspace/20250908_11_58_09_eval_save/20250908115829/DRIVING_BEV_DYN/0"
    meta_file_list = [os.path.join(root_dir, "metadata", ele) for ele in os.listdir(os.path.join(root_dir, "metadata"))]
    gt_file_list = [os.path.join(root_dir, "trues", ele)
                    for ele in os.listdir(os.path.join(root_dir, "trues"))]
    pred_file_list = [os.path.join(root_dir, "preds", ele) for ele in os.listdir(os.path.join(root_dir, "preds"))]

    print(len(meta_file_list), len(gt_file_list), len(pred_file_list))
    # exit(1)

    
    metas = []
    gts = []
    preds = []

    for p, g, m in tqdm(zip(pred_file_list, gt_file_list, meta_file_list)):
        metas += json.load(open(m, 'r'))
        gts += json.load(open(g, 'r'))
        preds += json.load(open(p, 'r'))


    print(ShowDataStruct("gts", gts, 2, 4))

    # print(len(inputs))
    # print(inputs[-1])
    # evaluation(*inputs)
    evaluation(preds, gts, metas, inputs[3])

    
