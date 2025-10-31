from gpal_lightning.neural_network.tasks.builder import POSTPROCESSES
from gpal_lightning.neural_network.tasks.base.postprocesses.postprocess import (
    BasePostProcess,
)
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_sta.postprocess.decode_pred import decode_pred_with_score, coordinate_transport_local

# from testing.decode.decode_pred import decode_pred_with_score, coordinate_transport_local
import numpy as np
from gpal_nn.tasks.driving_bev_sta.datasets.LaneData_utils import *
from gpal_nn.tasks.driving_bev_sta.losses.loss import pack_polyline_gt_points


@POSTPROCESSES.register_module()
class DRIVING_BEV_STAPostProcessing(BasePostProcess):
    def __init__(self, global_config, task_config):
        super().__init__(global_config, task_config)
        # self.pc_range = [0, -16, -10.0, 120, 16.0, 10.0]
        self.pc_range = [0, 0, 0, 32, 120, 0]
        self.num_vec = 50
        self.start_x = 120
        self.start_y = 16
        self.num_decode_layer = 6
        self.is_set_gt_z_as_zero = True

    def process_pred(self, vectors, metadata: dict) -> dict:
        vectors = vectors[0]
        out_k = ['all_cls_scores', 'all_pts_preds', 
                 'all_lane_marking_types_preds', 'all_lane_marking_colors_preds',
                 'all_shape_types_preds', 'all_centerline_types_preds',
                 'all_keypoint_classes_preds', 'all_keypoint_regs_preds']
        # print(ShowDataStruct("vectors", vectors))
        # print(ShowDataStruct("metadata", metadata))
        # print("DRIVING_BEV_STAPostProcessing process")
        # pass

        # print(ShowDataStruct("static_3d_output", vectors))
        # print(ShowDataStruct("_vectors", _vectors))
        outputs = {}
        onnx_path = self.global_config.config['onnx_path'] if 'onnx_path' in self.global_config.config else ''
        if 'onnx_path' in self.global_config.config and ('.bc' in onnx_path  or '.onnx' in onnx_path or '.hbm' in onnx_path):
            outputs['lane_3d_output'] = [vectors[key] for key in out_k]
        else:
            _vectors = {'lane_3d_output': [vectors[key] for key in out_k]}
            if 'lane_3d_output' in _vectors:
                outputs_classes, intermediate_reference_points, lane_marking_types, lane_marking_colors, shape_types, centerline_types, keypoint_classes, keypoint_regs = _vectors['lane_3d_output']
                cls_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, keypoint_cls_pred, keypoint_reg_pred = outputs_classes[self.num_decode_layer - 1], \
                                     intermediate_reference_points[self.num_decode_layer - 1], \
                                     lane_marking_types[self.num_decode_layer - 1], lane_marking_colors[self.num_decode_layer - 1], \
                                     shape_types[self.num_decode_layer - 1], centerline_types[self.num_decode_layer - 1], keypoint_classes[self.num_decode_layer - 1], \
                                     keypoint_regs[self.num_decode_layer - 1]
                # bbox_pred, points_pred = self.lane_map_head.transform_box(points_pred)
                outputs['lane_3d_output'] = cls_pred, points_pred, lane_marking_type_pred, lane_marking_color_pred, shape_type_pred, centerline_type_pred, keypoint_cls_pred, keypoint_reg_pred


        outputs2 = {}
        outputs2['static_3d_pred'] = outputs
        # print(ShowDataStruct("outputs2", outputs2))

        results = []

        outputs2["static_3d_pred"]["lane_3d_output"] = (
            outputs2["static_3d_pred"]["lane_3d_output"][0], None, outputs2["static_3d_pred"]["lane_3d_output"][1], outputs2["static_3d_pred"]["lane_3d_output"][2],
              outputs2["static_3d_pred"]["lane_3d_output"][3], outputs2["static_3d_pred"]["lane_3d_output"][4], outputs2["static_3d_pred"]["lane_3d_output"][5],
              outputs2["static_3d_pred"]["lane_3d_output"][6], outputs2["static_3d_pred"]["lane_3d_output"][7])
        cls_pred, bbox_pred, points_pred, lane_marking_types_pred, lane_marking_colors_pred, shape_types_pred, centerline_types_pred, keypoint_cls_pred, keypoint_reg_pred = outputs2['static_3d_pred']['lane_3d_output']

        for idx in range(len(points_pred)):
            result = dict()
            result['vectors'] = []

            bbox_per_pred = bbox_pred[idx] if bbox_pred is not None else None
            point_per_pred = points_pred[idx]
            cls_per_pred = cls_pred[idx]
            lane_marking_types_per_pred = lane_marking_types_pred[idx]
            lane_marking_colors_per_pred = lane_marking_colors_pred[idx]
            shape_types_per_pred = shape_types_pred[idx]
            centerline_types_per_pred = centerline_types_pred[idx]
            # keypoint_cls_per_pred = keypoint_cls_pred[idx]
            # keypoint_reg_per_pred =keypoint_reg_pred[idx]

            # print(self.pc_range)
            # print(point_per_pred)
            cls_per_pred = cls_pred[idx]
            bbox_per_pred, point_per_pred, score_per_pred, type_per_pred, lane_marking_type_per_pred, lane_marking_color_per_pred, shape_type_per_pred, centerline_type_per_pred, _,_ = decode_pred_with_score(cls_per_pred, bbox_per_pred,
                                                                                   point_per_pred, lane_marking_types_per_pred, lane_marking_colors_per_pred, shape_types_per_pred, centerline_types_per_pred, 
                                                                                   pc_range=self.pc_range,
                                                                                   num_query=self.num_vec)
            point_per_pred = coordinate_transport_local(
                point_per_pred, self.start_x, self.start_y)

            # print(point_per_pred)
            # exit(1)
            score_per_pred = score_per_pred.cpu().numpy()

            for idx, item in enumerate(zip(point_per_pred, type_per_pred, lane_marking_type_per_pred, lane_marking_color_per_pred, shape_type_per_pred, centerline_type_per_pred)):
                static_target = {
                    'pts': item[0][..., :2].tolist(),
                    'confidence_level': score_per_pred[idx],
                    'type': item[1].item(),
                    'cls_name': main_class_name_map[item[1].item()],
                    'lane_marking_type': item[2].item(),
                    'lane_marking_color': item[3].item(),
                    'shape_type': item[4].item(),   
                    'centerline_type': item[5].item(), 
                }

                result['vectors'].append(static_target)
            results.append(result)

        return results

    def process_gt(self, vectors, metadata: dict):
        results = []
        for idx, data in enumerate(vectors):
            result = dict()
            result['gt_vectors'] = []
            points, cls, lane_marking_types, lane_marking_colors, shape_types, centerline_types, is_split_merges, keypoint_norms = pack_polyline_gt_points(data)

            # points = shift_lane_points(points, self.pts_per_vector)
            if self.is_set_gt_z_as_zero == True and isinstance(points, np.ndarray):
                points[..., 2] = 0.0

            for index, item in enumerate(zip(points, cls, lane_marking_types, lane_marking_colors, shape_types, centerline_types)):
                static_target = {
                    'pts': item[0].tolist(),
                    'type': item[1],
                    'cls_name': main_class_name_map[item[1]],
                    'lane_marking_type' : item[2],
                    'lane_marking_color': item[3],
                    'shape_type': item[4], 
                    'centerline_type': item[5],
                }
                result['gt_vectors'].append(static_target)
            results.append(result)
        return results

    def process(self, vectors, metadata: dict, is_gt: bool = False) -> dict:
        if is_gt:
            return self.process_gt(vectors, metadata)
        else:
            return self.process_pred(vectors, metadata)
