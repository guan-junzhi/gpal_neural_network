from gpal_lightning.neural_network.tasks.builder import POSTPROCESSES
from gpal_lightning.neural_network.tasks.base.postprocesses.postprocess import (
    BasePostProcess,
)

from tools_scripts.data_format_cvt import ShowDataStruct
import numpy as np
import torch
from gpal_nn.tasks.driving_bev_dyn.postprocess import model_nms_utils
from gpal_nn.tasks.driving_bev_dyn.postprocess.bev_points import Bev_To_Points

@POSTPROCESSES.register_module()
class DRIVING_BEV_DYNPostProcessing(BasePostProcess):
    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)
        OD_RANGE = [-51.2, -30.72, -1.0, 102.4, 30.72, 5.0]

        self.POST_PROCESSING = dict(
            RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
            SCORE_THRESH=0.28,
            OUTPUT_RAW_SCORE=False,
            EVAL_METRIC="kitti",
            NMS_CONFIG=dict(
                MULTI_CLASSES_NMS=False,
                NMS_TYPE="nms_gpu",
                NMS_THRESH=0.1,
                NMS_PRE_MAXSIZE=4096,
                NMS_POST_MAXSIZE=500
            ),
            DET_RANGE_LIST=[
                OD_RANGE,
            ],
            IOU_THRESH_LIST=[0.5, 0.5, 0.25, 0.25]
        )
        self.num_class = len(task_config.class_dict)
        self.class_name = list(task_config.class_dict.values())
        BEV_TO_POINTS = dict(
            NAME="Bev_To_Points",
            NUM_BEV_FEATURES=64,
            VOXEL_SIZE=[0.64, 0.64],
            SCORE_THRESH=0.28,
            DOWN_RATIO=2,
            NUM_KEYPOINTS=256,
            TRAIN=True,
            NUM_OUTPUT_FEATURES=64
        )
        self.bev_2_points = Bev_To_Points(model_cfg=BEV_TO_POINTS,
                                                  grid_size=[480, 192,  12],
                                                  voxel_size=[0.32, 0.32, 0.5],
                                                  point_cloud_range=[-51.2, -
                                                                     30.72, -1., 102.4, 30.72, 5.],
                                                  num_bev_features=[64, 64, 128, 64, 128, 128, 128])



    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict


    
    def generate_prediction_dicts(self, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros([num_samples, 1]) , 
                'score': np.zeros(num_samples), 
                'boxes_lidar': np.zeros([num_samples, 9]), 
                'pred_labels': np.zeros([num_samples]),
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].detach().cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].detach().cpu().numpy()
            pred_labels = box_dict['pred_labels'].detach().cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]  
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            # single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)


        return annos

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.POST_PROCESSING
        batch_size = batch_dict['Points_Loss']['estimation_cen'].shape[0]
        recall_dict = {}
        pred_dicts = []
        _, label = batch_dict['Points_Loss']['estimation_score_cls'].max(dim=1)  # 原始的score含通道
        batch_pred_labels = label.view(batch_size, -1) + 1

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg['NMS_CONFIG']['MULTI_CLASSES_NMS']:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg['NMS_CONFIG'],
                        score_thresh=post_process_cfg['SCORE_THRESH']
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    # label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    # label_preds = batch_dict[label_key][index]
                    label_preds = batch_pred_labels[index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg['NMS_CONFIG'],
                    score_thresh=post_process_cfg['SCORE_THRESH']
                )

                if post_process_cfg['OUTPUT_RAW_SCORE']:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg['RECALL_THRESH_LIST']
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict


    @torch.no_grad()
    def generate_predicted_boxes(self, outputs, batch_size, **kwargs):
        # template_xyz = outputs['template_xyz'][:, :, :2].view(-1, 2)
        pred_cen = outputs['estimation_cen'].permute(0, 2, 1)
        pred_z = outputs['estimation_z'].permute(0, 2, 1)
        pred_dim = outputs['estimation_dim'].permute(0, 2, 1)
        pred_dir = outputs['estimation_dir'].permute(0, 2, 1)
        
        bin_flag = (pred_dir[:, :, 0] < pred_dir[:, :, 1])
        pred_yaw =( torch.atan2(pred_dir[:, :, 2], pred_dir[:, :, 3]) * bin_flag.float() + \
            torch.atan2(pred_dir[:, :, 4], pred_dir[:, :, 5]) * (1.0-bin_flag.float())).unsqueeze(-1)

        pred_vel = outputs['estimation_vel'].permute(0, 2, 1)
        pred_score = outputs['estimation_score'].sigmoid().permute(
            0, 2, 1)  # 得分，时序特征融合后的得分

        indicee = torch.arange(0, 256).to(pred_vel).float().view(-1, 256, 1)

        track_boxes = torch.cat([pred_cen,
                                pred_z,
                                pred_dim,
                                pred_yaw,
                                pred_vel*80,
                                # template_xyz.view(-1, 256, 2),
                                #   indicee
                                ], dim=-1
                                )

        return pred_score, track_boxes

    def process(self, vectors, metadata: dict, is_gt: bool = False) -> dict:
        # print(ShowDataStruct(f"vectors gt = {is_gt}", vectors))
        if is_gt:
            gt_list = [{"gt_boxes": ele["gt_boxes"]} for ele in vectors]
            return gt_list
        else:
            #   if not self.training or self.predict_boxes_when_training:
            if "with_postprocess" not in vectors[0]:
                vectors[0] = self.bev_2_points(vectors[0])
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(vectors[0]['Points_Loss'],
                                                                             batch_size=vectors[0]['Points_Loss']['estimation_cen'].shape[0])
            vectors[0]['batch_cls_preds'] = batch_cls_preds
            vectors[0]['batch_box_preds'] = batch_box_preds
            vectors[0]['cls_preds_normalized'] = True
            vectors[0]['has_class_labels'] = True


            pred_dicts, recall_dict = self.post_processing(vectors[0])
            pred_dicts = self.generate_prediction_dicts(pred_dicts, self.class_name)
            return pred_dicts
    