from gpal_lightning.neural_network.tasks.builder import POSTPROCESSES
from gpal_lightning.neural_network.tasks.base.postprocesses.postprocess import (
    BasePostProcess,
)
from tools_scripts.data_format_cvt import ShowDataStruct
from gpal_nn.tasks.driving_bev_sta.postprocess.decode_pred import decode_pred_with_score, coordinate_transport_local

# from testing.decode.decode_pred import decode_pred_with_score, coordinate_transport_local
import numpy as np


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
        # print(ShowDataStruct("vectors", vectors))
        # print(ShowDataStruct("metadata", metadata))
        # print("DRIVING_BEV_STAPostProcessing process")
        # pass

        # print(ShowDataStruct("static_3d_output", vectors))
        _vectors = {'lane_3d_output':
                    [
                        vectors['all_cls_scores'],
                        vectors['all_pts_preds']
                    ]
                    }

        # print(ShowDataStruct("_vectors", _vectors))
        outputs = {}
        if 'lane_3d_output' in _vectors:
            outputs_classes, intermediate_reference_points = _vectors['lane_3d_output']
            cls_pred, points_pred = outputs_classes[self.num_decode_layer - 1].softmax(-1), \
                intermediate_reference_points[self.num_decode_layer - 1]
            # cls_pred, points_pred = self.dequant_0(cls_pred), self.dequant_1(points_pred)
            # bbox_pred, points_pred = self.lane_map_head.transform_box(points_pred)
            outputs['lane_3d_output'] = cls_pred, points_pred

        outputs2 = {}
        outputs2['static_3d_pred'] = outputs
        # print(ShowDataStruct("outputs2", outputs2))

        results = []

        outputs2["static_3d_pred"]["lane_3d_output"] = (
            outputs2["static_3d_pred"]["lane_3d_output"][0], None, outputs2["static_3d_pred"]["lane_3d_output"][1])
        cls_pred, bbox_pred, points_pred = outputs2['static_3d_pred']['lane_3d_output']

        for idx in range(len(points_pred)):
            result = dict()
            result['vectors'] = []

            bbox_per_pred = bbox_pred[idx] if bbox_pred is not None else None
            point_per_pred = points_pred[idx]

            # print(self.pc_range)
            # print(point_per_pred)
            cls_per_pred = cls_pred[idx]
            bbox_per_pred, point_per_pred, score_per_pred = decode_pred_with_score(cls_per_pred, bbox_per_pred,
                                                                                   point_per_pred,
                                                                                   pc_range=self.pc_range,
                                                                                   num_query=self.num_vec)
            point_per_pred = coordinate_transport_local(
                point_per_pred, self.start_x, self.start_y)

            # print(point_per_pred)
            # exit(1)
            score_per_pred = score_per_pred.cpu().numpy()

            for idx, item in enumerate(point_per_pred):
                static_target = {
                    'pts': item[..., :2].tolist(),
                    'confidence_level': score_per_pred[idx],
                    'type': 0,
                    'cls_name': 'normal',
                }

                result['vectors'].append(static_target)
            results.append(result)

        return results

    def pack_polyline_gt_points(self, data, bs_idx):
        annos = []
        if 'points' in data['polylines']:
            annos.append(data['polylines']['points'])
        if 'points' in data['edges']:
            annos.append(data['edges']['points'])

        if len(annos) > 0:
            annos = np.concatenate(annos, axis=0)
        return annos

    def process_gt(self, vectors, metadata: dict):

        results = []
        for idx, data in enumerate(vectors):
            # print(ShowDataStruct("data", data))
            result = dict()
            result['gt_vectors'] = []
            points = self.pack_polyline_gt_points(data, idx)

            # points = shift_lane_points(points, self.pts_per_vector)
            if self.is_set_gt_z_as_zero == True and isinstance(points, np.ndarray):
                points[..., 2] = 0.0

            for index, item in enumerate(points):
                static_target = {
                    'pts': item.tolist(),
                    'type': 0,
                    'cls_name': 'normal',
                }
                result['gt_vectors'].append(static_target)
            results.append(result)
        return results

    def process(self, vectors, metadata: dict, is_gt: bool = False) -> dict:
        if is_gt:
            return self.process_gt(vectors, metadata)
        else:
            return self.process_pred(vectors, metadata)
