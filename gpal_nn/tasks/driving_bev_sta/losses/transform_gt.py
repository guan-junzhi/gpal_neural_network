import numpy as np
import torch


def transform_gt_box(pts, start_x, start_y, num_pts_per_vec=20, y_first=False, device="cpu"):
    """
    Converting the points set into bounding box.

    Args:
        pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
    Returns:
        The bbox [cx, cy, w, h] transformed from points.
    """
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts).to(device)
    if isinstance(pts, list):  # 无真值
        # if len(pts) == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0, num_pts_per_vec, 2), device=device)
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)

    # pts_reshape = torch.tensor(pts[:, :num_pts_per_vec, :2])
    pts_reshape = pts[:, :num_pts_per_vec, :2].clone().detach()
    pts_y = pts_reshape[:, :, 0] if y_first else pts_reshape[:, :, 1]
    pts_x = pts_reshape[:, :, 1] if y_first else pts_reshape[:, :, 0]

    pts_h = start_x - pts_x
    pts_w = start_y - pts_y

    xmin = torch.min(pts_w, dim=1, keepdim=True)[0]
    xmax = torch.max(pts_w, dim=1, keepdim=True)[0]
    ymin = torch.min(pts_h, dim=1, keepdim=True)[0]
    ymax = torch.max(pts_h, dim=1, keepdim=True)[0]

    bbox = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
    _pts = pts_reshape.clone()
    _pts[..., 0] = pts_w
    _pts[..., 1] = pts_h
    return bbox, _pts


def shift_polyline_points(pts, num_pts_per_vec=20):
    assert pts.shape[1] == num_pts_per_vec
    assert pts.shape[2] == 2
    shift_points = []
    shift_points.append(pts)
    shift_points.append(torch.flip(pts, [1]))
    return torch.stack(shift_points).permute(1, 0, 2, 3)


def shift_polygen_points(pts, num_pts_per_vec=20):
    assert pts.shape[1] == num_pts_per_vec
    assert pts.shape[2] == 2

    shift_points = []
    for i in range(num_pts_per_vec):
        shift_pts = torch.roll(pts, i, dims=1)
        shift_points.append(shift_pts)
    return torch.stack(shift_points).permute(1, 0, 2, 3)
