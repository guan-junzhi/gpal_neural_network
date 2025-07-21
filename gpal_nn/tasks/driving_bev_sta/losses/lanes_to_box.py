import torch

'''
in
lane: n * 3; [x, y, z] the world coordinate

out
(box, pts)
box: relate world box (cx, cy, w, h) (0~max)
pts: relate relate box pts n*(x, y) (0~1)
'''


def lane_to_box_vec(lane, start_x=50, start_y=50):
    x = lane[..., 0]
    y = lane[..., 1]
    max_x = x.max()
    max_y = y.max()
    min_x = x.min()
    min_y = y.min()
    h = torch.abs(max_x - min_x)
    w = torch.abs(max_y - min_y)
    pts = torch.zeros((*lane.shape[:-1], 2), device=lane.device)
    pts[..., 1] = (max_x - x) / h
    pts[..., 0] = (max_y - y) / w

    box = torch.zeros(4)  # pxpypwph
    box[0] = start_y - (min_y + max_y) / 2.
    box[1] = start_x - (min_x + max_x) / 2.
    box[2] = w
    box[3] = h

    return box, pts


def cxcywh_to_xyxy(box):
    new_box = torch.zeros_like(box)
    cx = box[..., 0]
    cy = box[..., 1]
    bw = box[..., 2]
    bh = box[..., 3]
    new_box[..., 0] = cx - bw / 2.
    new_box[..., 1] = cy - bh / 2.
    new_box[..., 2] = cx + bw / 2.
    new_box[..., 3] = cy + bh / 2.
    return new_box


def xyxy_to_cxcywh(box):
    new_box = torch.zeros_like(box)
    x1 = box[..., 0]
    y1 = box[..., 1]
    x2 = box[..., 2]
    y2 = box[..., 3]
    new_box[..., 0] = (x1 + x2) / 2.
    new_box[..., 1] = (y1 + y2) / 2.
    new_box[..., 2] = x2 - x1
    new_box[..., 3] = y2 - y1
    return new_box


def box_vec_to_lane(box, vec, start_x=50, start_y=50):
    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]
    max_y = start_y - cx + w / 2.
    max_x = start_x - cy + h / 2.
    lane = torch.zeros((*vec.shape[:-1], 3), device=box.device)
    lane[..., 0] = max_x - vec[..., 1] * h
    lane[..., 1] = max_y - vec[..., 0] * w
    return lane
