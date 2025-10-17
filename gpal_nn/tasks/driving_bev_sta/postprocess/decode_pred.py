import torch
from gpal_nn.tasks.driving_bev_sta.losses.base_assigner import denormalize_2d_pts, denormalize_2d_bbox


def decode_pred_with_score(cls_score_pred, bbox_pred=None, points_pred=None, shape_types_pred=None, centerline_types_pred=None, 
                           keypoint_cls_pred=None, keypoint_reg_pred=None, num_cls=2, num_query=50, pc_range=None,
                           socre_thr=0.3):
    cls_score_pred = cls_score_pred.squeeze().sigmoid()
    values, cls_pred = cls_score_pred.max(-1)
    # print("values", values)
    # print("cls_pred", cls_pred)
    socre_mask = values > socre_thr
    values = values[socre_mask]
    cls_pred = cls_pred[socre_mask]
    cls_score_pred = values

    if bbox_pred is not None:
        bbox_pred = bbox_pred.squeeze()
        bbox_pred = denormalize_2d_bbox(bbox_pred, pc_range)
        # bbox_pred = bbox_pred[cls_score_pred[..., 0] < cls_score_pred[..., 1]]
        bbox_pred = bbox_pred[socre_mask]
    if points_pred is not None:
        points_pred = points_pred.squeeze()
        points_pred = denormalize_2d_pts(points_pred, pc_range)
        # points_pred = points_pred[cls_score_pred[..., 0] < cls_score_pred[..., 1]]
        points_pred = points_pred[socre_mask]
    if shape_types_pred is not None:
        _, shape_types_pred = shape_types_pred.squeeze().max(-1)
        shape_types_pred = shape_types_pred[socre_mask]
    if centerline_types_pred is not None:
        _, centerline_types_pred = centerline_types_pred.squeeze().max(-1)
        centerline_types_pred = centerline_types_pred[socre_mask]
    if keypoint_cls_pred is not None:
        keypoint_cls_pred = keypoint_cls_pred.squeeze().sigmoid()
        keypoint_cls_pred = keypoint_cls_pred[socre_mask]
    if keypoint_reg_pred is not None:
        keypoint_reg_pred = keypoint_reg_pred.squeeze()
        keypoint_reg_pred = keypoint_reg_pred[socre_mask]

    return bbox_pred, points_pred, cls_score_pred, cls_pred, shape_types_pred, centerline_types_pred, keypoint_cls_pred, keypoint_reg_pred

def coordinate_transport_local(points_pred, start_x, start_y):
    '''
    :param points_pred:
    :param start_x:
    :param start_y:
    :return:
    '''
    if len(points_pred) > 0:
        new_points = torch.zeros((*points_pred.shape[:-1], 3), device=points_pred.device)
        new_points[..., 0] = start_x - points_pred[..., 1]
        new_points[..., 1] = start_y - points_pred[..., 0]
        return new_points
    else:
        return points_pred
