import torch
from shapely.geometry import Polygon
import numpy as np
from math import sin, cos

# from ...ops.iou3d_nms import iou3d_nms_utils


# def oriented_nms(
#     obj_dicts,
#     front,
#     left,
#     bev_h_resolution,
#     bev_w_resolution,
#     adjust_nms,
#     iou_th=0.00001,
# ):
#     """
#     this function performs nms for oriented bounding boxes (because of yaw)
#     you can refer to the details here: https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles
#     """
#     # sort by confidence
#     obj_dicts = sorted(obj_dicts, key=lambda i: i["conf"], reverse=True)
#     # get polygons
#     polygons = []
#     for obj in obj_dicts:
#         X, Y, L, W, H, yaw, category = (
#             obj["X"],
#             obj["Y"],
#             obj["L"],
#             obj["W"],
#             obj["H"],
#             obj["yaw"],
#             obj["category"],
#         )
#         box = BBOX3D(X, Y, H / 2, L, W, H, yaw)
#         pts = box.bottom_corners()[:2].T
#         pts[:, 0] = int(front / bev_h_resolution) - pts[:, 0] / bev_h_resolution
#         pts[:, 1] = int(left / bev_w_resolution) - pts[:, 1] / bev_w_resolution
#         pts[:, [1, 0]] = pts[:, [0, 1]]
#         polygons.append(
#             Polygon([tuple(pts[0][:2]), tuple(pts[1][:2]), tuple(pts[2][:2]), tuple(pts[3][:2])])
#         )
#     # perform nms
#     selected_polygons = []
#     selected_obj = []
#     while polygons:
#         next_polygon = polygons.pop(0)
#         selected_polygons.append(next_polygon)
#         next_obj = obj_dicts.pop(0)
#         selected_obj.append(next_obj)
#         remaining_boxes = []
#         remaining_obj = []
#         for idx, (rest, rest_obj) in enumerate(zip(polygons, obj_dicts)):
#             # calculate IOU
#             i_o_u = next_polygon.intersection(rest).area / next_polygon.union(rest).area
#             #############################
#             # we can adjust the nms threshold in the given range
#             if adjust_nms:
#                 if abs(next_obj["X"]) < adjust_nms["X"] and abs(next_obj["Y"]) < adjust_nms["Y"]:
#                     if i_o_u < iou_th + adjust_nms["adjusted_threshold"]:
#                         remaining_boxes.append(rest)
#                         remaining_obj.append(rest_obj)
#                         continue
#             ############################

#             if i_o_u < iou_th:
#                 remaining_boxes.append(rest)
#                 remaining_obj.append(rest_obj)
#         polygons = remaining_boxes
#         obj_dicts = remaining_obj
#     return selected_obj


class BBOX3D:
    def __init__(self, X, Y, Z, L, W, H, yaw, pitch=0, roll=0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.L = L
        self.W = W
        self.H = H
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.corners = self.get_corners()

    def get_corners(self):
        x_corners = self.L / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = self.W / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = self.H / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        rotation_matrix = self.euler_to_Rot(self.yaw, self.pitch, self.roll)
        corners = np.dot(rotation_matrix, corners)

        # Translate
        corners[0, :] = corners[0, :] + self.X
        corners[1, :] = corners[1, :] + self.Y
        corners[2, :] = corners[2, :] + self.Z

        return corners

    def euler_to_Rot(self, yaw, pitch, roll):
        P = np.array([[cos(pitch), 0, sin(pitch)], [
                     0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
        R = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)],
                     [0, sin(roll), cos(roll)]])
        Y = np.array([[cos(yaw), -sin(yaw), 0],
                     [sin(yaw), cos(yaw), 0], [0, 0, 1]])
        return np.dot(Y, np.dot(P, R))

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners[:, [2, 3, 7, 6]]

def oriented_nms(
    boxes_for_nms,
    box_scores_nms,
    # front,
    # left,
    # bev_h_resolution,
    # bev_w_resolution,
    # adjust_nms,
    iou_th=0.00001,
):
    # // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    # // params keep: (N)

    """
    this function performs nms for oriented bounding boxes (because of yaw)
    you can refer to the details here: https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles
    """
    # sort by confidence
    
    order = np.argsort(box_scores_nms)[::-1]
    boxes_for_nms = boxes_for_nms[order]
    box_scores_nms = box_scores_nms[order]
    input_idxs = np.array(list(range(len(boxes_for_nms)))).astype(np.int32)
    input_idxs = input_idxs[order]

    # get polygons
    polygons = []
    for obj in boxes_for_nms:
        X, Y, Z, L, W, H, yaw, = obj

        box = BBOX3D(X, Y, H / 2, L, W, H, yaw)
        pts = box.bottom_corners()[:2].T
        polygons.append(
            Polygon([tuple(pts[0][:2]), tuple(pts[1][:2]),
                    tuple(pts[2][:2]), tuple(pts[3][:2])])
        )
    # perform nms
    selected_polygons = []
    selected_idx = []
    idxs = list(range(len(polygons)))
    while polygons:
        next_polygon = polygons.pop(0)
        selected_polygons.append(next_polygon)
        next_idx = idxs.pop(0)
        selected_idx.append(next_idx)
        remaining_boxes = []
        remaining_idx = []
        for idx, (rest, rest_obj) in enumerate(zip(polygons, idxs)):
            # calculate IOU
            i_o_u = next_polygon.intersection(
                rest).area / next_polygon.union(rest).area
            if i_o_u < iou_th:
                remaining_boxes.append(rest)
                remaining_idx.append(rest_obj)
        polygons = remaining_boxes
        idxs = remaining_idx

    return input_idxs[selected_idx], box_scores_nms[selected_idx], box_scores_nms[selected_idx]

def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # import pickle as pkl
    # pkl.dump((box_scores, box_preds, nms_config, score_thresh),
    #          open("class_agnostic_nms.pkl", "wb"))
    # exit(1)


    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config['NMS_PRE_MAXSIZE'], box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]

        boxes_for_nms_np = boxes_for_nms.detach().cpu().numpy()
        box_scores_nms_np = box_scores_nms.detach().cpu().numpy()

        selected_idx, selected_box, selected_scores = oriented_nms(boxes_for_nms_np[:, 0:7],
                                                     box_scores_nms_np, nms_config['NMS_THRESH'])
        
        selected_idx = torch.from_numpy(selected_idx).to(boxes_for_nms.device)
        selected_box = torch.from_numpy(selected_box).to(boxes_for_nms.device)
        selected_scores = torch.from_numpy(
            selected_scores).to(boxes_for_nms.device)

        selected = indices[selected_idx[:nms_config['NMS_POST_MAXSIZE']]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    # return selected_idx, selected_scores
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes


if __name__ == "__main__":
    import pickle as pkl
    inputs = pkl.load(open("class_agnostic_nms.pkl", 'rb'))

    # random.seed(555)
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29501'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    # distributed.init_process_group(backend='nccl')
    train_dataset = class_agnostic_nms(*inputs)

    # print(len(train_dataset))

# [0.93091273 0.88666254 0.8674448  0.8431826  0.80645233 0.79554695
#  0.7930531  0.72659546 0.61026555 0.5698092  0.5651864  0.55626786
#  0.52614665 0.45524606 0.37250292 0.35342386 0.34189236 0.3372056
#  0.334784  ]
# [0.93091273 0.88666254 0.8674448  0.8431826  0.80645233 0.79554695
#  0.7930531  0.72659546 0.61026555 0.5698092  0.5651864  0.55626786
#  0.52614665 0.45524606 0.37250292 0.35342386 0.34189236 0.3372056
#  0.334784  ]
# [0, 1, 2, 7, 8, 12, 13, 16]
# tensor([0.9309, 0.8867, 0.8674, 0.7266, 0.6103, 0.5261, 0.4552, 0.3419],
#        device='cuda:0')