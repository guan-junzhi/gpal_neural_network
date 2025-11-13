# import sys;sys.path.insert(0, "/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_neural_network")
# import sys;sys.path.insert(0, "/opt/GPAL_Repo_PYTHON/gpal_neural_network")
import sys;sys.path.insert(0, "/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_neural_network")

import os
import copy

import json
import logging
import datetime

import numpy as np
from tqdm import tqdm

import cv2
from gpal_lightning.neural_network.tasks.base.evaluators.evaluator import \
    BaseEvaluator
from gpal_lightning.neural_network.tasks.builder import EVALUATORS
# from gpal_nn.tasks.driving_bev_dyn.datasets.txtlabel_instance_p3 import TXTLabelLoader
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_dyn.evaluators.evaluation_node import ObjectDetectionEvaluator


def get_git_hash():
    git_hash = os.popen('git rev-parse HEAD').read().strip()
    return git_hash

def test_dynamic_thresholds(loggerinfo=print, restricted_ratio=[0.05, 0.005]):
    """
    测试不同距离下的动态阈值效果
    """
    loggerinfo("=== 动态阈值测试 ===")
    
    # ego_pos = np.array([0.0, 0.0])
    longitudinal_ratio = restricted_ratio[0]  # 5%
    lateral_ratio = restricted_ratio[1]      # 0.5%
    
    # 测试不同距离的GT
    test_distances = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 米
    
    loggerinfo("距离\t径向阈值\t横向阈值\t总阈值")
    loggerinfo("-" * 40)

    for dist in test_distances:
        long_thr = dist * longitudinal_ratio
        lat_thr = dist * lateral_ratio  
        total_thr = np.sqrt(long_thr**2 + lat_thr**2)
        
        loggerinfo(f"{dist:<3}m\t{long_thr:.3f}m\t\t{lat_thr:.3f}m\t\t{total_thr:.3f}m")
    
    loggerinfo("优势:")
    loggerinfo("- 近距离目标: 更严格的匹配要求")
    loggerinfo("- 远距离目标: 更宽松的匹配要求") 
    loggerinfo("- 自适应: 根据目标距离自动调整精度要求")


def create_logger(log_file=None, rank=0, log_level=logging.INFO, use_console=True):
    logger_name = __name__ if log_file is None else f"{__name__}.{log_file}"
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 根据 rank 设置日志级别（仅 rank=0 时记录指定级别，否则 ERROR）
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    
    # 仅在需要时添加控制台处理器
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(log_level if rank == 0 else logging.ERROR)
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    # 添加文件处理器（若指定了 log_file）
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False  # 防止传播到根 logger
    return logger


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
    
    cfgs = {
        # ============= 固定参数 ============= # 
        # default(无需额外传入参数进行控制)
        'class_names': class_names,
        'preds': preds,
        'color_list': ['k', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan'],
        'use_print_format': 2,
        'find_worst': True,
        'top_n': 100,
        
        # 切分评估范围
        'det_range_list': [
            [-30, -10, -2, 30, 10, 4],
            [-50, -10, -2, 50, 10, 4],
            [-30, -15, -2, 30, 15, 4],
            [-30, -30, -2, 30, 30, 4],
            [-50, -30, -2, 50, 30, 4],
            [-50, -30, -2, 70, 30, 4],
            [-50, -30, -2, 80, 30, 4],
            [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0],
        ],

        'distance_threshold_list': [0.5],
        'restricted_ratio': [0.1, 0.05],
    }
    
    if True:
        class_names             = cfgs['class_names']
        preds                   = cfgs['preds']
        color_list              = cfgs['color_list']
        use_print_format        = cfgs['use_print_format']
        find_worst              = cfgs['find_worst']
        top_n                   = cfgs['top_n']
        
        det_range_list          = cfgs['det_range_list']
        restricted_ratio        = cfgs['restricted_ratio']
        distance_threshold_list = cfgs['distance_threshold_list']
        # ============= 可变参数 ============= # 
    
    
    # 结果存储位置(评测结果和badcase图片)
    git_hash = get_git_hash()[:11]
    dirname = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{git_hash}_{use_print_format}_{distance_threshold_list}_{restricted_ratio}'
    save_dir = f'./{result_dir}/{dirname}'
    save_badcase_dir = f'{save_dir}/badcase_img' if find_worst else f'{save_dir}/goodcase_img'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_badcase_dir, exist_ok=True)
    
    log_file = f'{save_dir}/record_logs_format_{use_print_format}_{dirname}.log'
    logger = create_logger(log_file=log_file, rank=0, log_level=logging.INFO, use_console=False)
    
    logger.info(f'git_hash: {git_hash}')
    logger.info(f'save_dir: {save_dir}')
    logger.info(f'save_badcase_dir: {save_badcase_dir}')
    logger.info(f'color_list: {color_list}')
    logger.info(f'use_print_format: {use_print_format}')
    logger.info(f'restricted_ratio: {restricted_ratio}')
    logger.info(f'distance_threshold_list: {distance_threshold_list}')
    logger.info(f'find_worst: {find_worst}')
    logger.info(f'top_n: {top_n}')
    logger.info(f'preds: {len(preds)}')
    logger.info(f'class_names: {class_names}')
    logger.info(f'\n')
    
    evaluator = ObjectDetectionEvaluator(
        class_names = class_names,
        
        det_range_list = det_range_list,

        # slice_line_len = 295,
        # data_print_len = 12,
        # class_name_print_len = 20,

        r_at_p=0.7,
        precision_points=[0.5, 0.6, 0.7, 0.8, 0.9],

        distance_threshold_list = distance_threshold_list,
        restricted_ratio = restricted_ratio,
    )
    
    # exit(1)

    det_annos = copy.deepcopy(preds)
    for i in range(len(gts)):
        det_annos[i].update(gts[i])
    
    logger.info(f'det_annos: len: {len(det_annos)}')
    key_det_annos = [det_annos[i] for i in range(len(det_annos)) if metas[i]['is_key']]
    det_annos = key_det_annos
    logger.info(f'key_det_annos: len:{len(key_det_annos)}')
    # det_annox = evaluator.load_det_annos(
    #     det_annos)  # 文件路径 or list[dict, dict, ...]
    
    distance_errors_list = evaluator.evaluate(det_annos, # list[dict, dict, ...]
                                              loggerinfo=logger.info, 
                                              save_dir=save_dir,
                                              is_print_during_info=True, 
                                              use_print_format=use_print_format
                                              )
    logger.info(f'\n')
    test_dynamic_thresholds(loggerinfo=logger.info, restricted_ratio=restricted_ratio)
    logger.info(f'\n')
    
    DATA_COLLECT_ROOT = os.getenv("ENV_GPAL_NEURAL_NETWORK_DATA_COLLECT_ROOT")
    if DATA_COLLECT_ROOT is None:
        DATA_COLLECT_ROOT = "/data/dp_group/process-prod-bucket/data_collect/"
    image_dir = DATA_COLLECT_ROOT
    
    # badcase 展示不生成
    if True:
        frame_infos = evaluator.get_frame_infos_from_distance_errors(distance_errors_list=distance_errors_list,)
    
        # TODO 暂时调试使用，后续统一到函数接口内部
        value_frame_infos = evaluator.query_raw_frame_info(query_frame_infos=frame_infos, key_frame_infos=det_annos)
        parse_frame_infos = evaluator.sparse_frame_infos_for_vis(value_frame_infos)
        
        for worst_i, frame_info in enumerate(tqdm(parse_frame_infos, desc='vis badcase')):  # 有badcase的fp fn
            # timestamp = frame_info['timestamp']  # 是gt索引, 并不是timestamp
            frame_id = frame_info['frame_id']

            # if worst_i % 4 != 0:
            #     continue
            
            temp_image_file = f'{save_badcase_dir}/{worst_i:05d}_{frame_id}.png'
            evaluator.visualize_pred_scores_and_gt_3dbox_use_bev_and_side_box(
                pts_range = det_range_list[-1],
                class_colors = color_list,
                points = None,
                pred_boxes = frame_info['tot_dt_boxes'],
                pred_label_ids = frame_info['tot_dt_labels'],
                pred_scores = frame_info['tot_dt_scores'],
                gt_boxes = frame_info['tot_gt_boxes'],
                gt_boxes_label_ids = frame_info['tot_gt_boxes_label_ids'],
                save_imgfile = temp_image_file,
                frame_id = frame_id,
                save_dir = save_badcase_dir,
                fnfp_info = frame_info['classes_data'],
            )
            
            curr_meta_info = metas[worst_i]
            camera_name_list = curr_meta_info['camera_name']
            curr_clip_id = curr_meta_info['clip_id'].replace('^', '/')
            timestamp = curr_meta_info['timestamp']
            
            image_list = []
            for curr_cam_name in camera_name_list:
                curr_view_path = f'{image_dir}/{curr_clip_id}/{curr_cam_name}/{timestamp}.jpg'
                
                curr_view_data = cv2.imread(curr_view_path)
                cv2.putText(curr_view_data, curr_cam_name, (50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[254, 254, 254], thickness=2, fontScale=2)
                # cv2.imwrite(f'{save_badcase_dir}/{worst_i:05d}_{frame_id}_{curr_cam_name}.jpg', curr_view_data)
                image_list.append(curr_view_data)
            
            image_img = np.concatenate(image_list, axis=1)
            
            bev_img = cv2.imread(temp_image_file)
            if os.path.exists(temp_image_file):
                os.remove(temp_image_file)
                os.system(f'rm -f {temp_image_file}')
            
            HFm, WFm = image_img.shape[:2]
            HTo, WTo = bev_img.shape[:2]
            newH = int(HFm * WTo / WFm)
            image_img = cv2.resize(image_img, (WTo, newH))  # 宽对齐

            tot_img = np.concatenate([bev_img, image_img,], axis=0)
            
            cv2.putText(tot_img, f'{curr_clip_id}^{timestamp}', (20, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=[0, 0, 0], thickness=1, fontScale=0.5)
            cv2.imwrite(f'{save_badcase_dir}/{worst_i:05d}_{frame_id}_{timestamp}.jpg', tot_img)

        logger.info(f'\n')
        logger.info(f'restricted_ratio: {restricted_ratio}')
        logger.info(f'distance_threshold_list: {distance_threshold_list}')
        logger.info(f'use_print_format: {use_print_format}')
        logger.info(f'find_worst: {find_worst}')
        logger.info(f'top_n: {top_n}')
        logger.info(f'\n')
        test_dynamic_thresholds(loggerinfo=logger.info, restricted_ratio=restricted_ratio)
        logger.info(f'\n')
        

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

    root_dir = "20250913_06_38_02/20250913063814/DRIVING_BEV_DYN/0"
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

    # # === 自定义评估
    # relative_path = "gpal_neural_network_one_node_traning_job_on_airflow_for_sikong_20251112_13_08_23/20251113_06_37_10^epoch=643-step=42500_checkpoint/20251113063711/DRIVING_BEV_DYN/0"
    # relative_path = "gpal_neural_network_one_node_traning_job_on_airflow_for_sikong_20251112_13_08_23/20251113_07_32_14^epoch=643-step=42500_checkpoint/20251113073214/DRIVING_BEV_DYN/0"
    # root_dir = f"/data/ai_group/workdirs/od_occ_group/mendeswan/codes/gpal_neural_network/.vscode/workspace_ws_batch/{relative_path}"
    
    # # 对保存的结果再次单独评测使用
    # meta_file_list = sorted([os.path.join(root_dir, "metadata", ele) for ele in os.listdir(os.path.join(root_dir, "metadata"))])
    # gt_file_list = sorted([os.path.join(root_dir, "trues", ele)
    #                 for ele in os.listdir(os.path.join(root_dir, "trues"))])
    # pred_file_list = sorted([os.path.join(root_dir, "preds", ele) for ele in os.listdir(os.path.join(root_dir, "preds"))])
    
    # # 这种和模型评测的结果是一致的,但是和输出的json里对不上,应该是因为json里的顺序和这里的顺序不一致
    # # meta_file_list = [os.path.join(root_dir, "metadata", ele) for ele in os.listdir(os.path.join(root_dir, "metadata"))]
    # # gt_file_list = [os.path.join(root_dir, "trues", ele)
    # #                 for ele in os.listdir(os.path.join(root_dir, "trues"))]
    # # pred_file_list = [os.path.join(root_dir, "preds", ele) for ele in os.listdir(os.path.join(root_dir, "preds"))]
    # print(len(meta_file_list), len(gt_file_list), len(pred_file_list))
    
    # metas = []
    # gts = []
    # preds = []

    # for p, g, m in tqdm(zip(pred_file_list, gt_file_list, meta_file_list)):
    #     metas += json.load(open(m, 'r'))
    #     gts += json.load(open(g, 'r'))
    #     preds += json.load(open(p, 'r'))


    # # print(ShowDataStruct("gts", gts, 2, 4))

    # class_names = [
    #     "vehicle_car",
    #     "vehicle_truck",
    #     "vehicle_construction_vehicle",
    #     "vehicle_cyclist",
    #     "vehicle_tricycle",
    #     "human_pedestrian",
    # ]
    # evaluation(preds, gts, metas, class_names)