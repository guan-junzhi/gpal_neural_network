import numpy as np
import torch

def point_projection_lane(points, ext, ist, dist=None, bev_real2aug=torch.eye(4, dtype=torch.float32)):
    '''
    :param points: m,20,3 or m,20,2
    :param ext: b,n,3,4
    :param ist: b,n,3,3
    :param dist: b,n,1,5
    :return: b,n,N,2 N=m*20
    '''
    if isinstance(points, list):
        points = torch.tensor(points)
        ext = torch.tensor(ext)
        ist = torch.tensor(ist)
        dist = torch.tensor(dist)
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
        ext = torch.from_numpy(ext)
        ist = torch.from_numpy(ist)
        dist = torch.from_numpy(dist)
    if points.device != ext.device:
        points = points.to(ext.device)

    if points.shape[-1] == 2:  # pred_points
        new_points_pred = torch.concat(
            [points,
             torch.zeros((*points.shape[:2], 1), device=points.device),
             torch.ones((*points.shape[:2], 1), device=points.device)],
            dim=-1).reshape(-1, 4)  # N,4
    elif points.shape[-1] == 3:  # gt_points
        new_points_pred = torch.concat(
            [points,
             torch.ones((*points.shape[:2], 1), device=points.device)],
            dim=-1).reshape(-1, 4)  # N,4
    bev_aug2real = torch.inverse(bev_real2aug).to(new_points_pred.device)
    new_points_pred = new_points_pred.transpose(0, 1)  # 4,N
    new_points_pred = bev_aug2real @ new_points_pred
    new_points_pred = torch.matmul(ext, new_points_pred)  # [b,n,3,4]*[4,N]  [b,n,3,N]
    pixel_point = new_points_pred / new_points_pred[:, :, 2:3, :]

    if dist is not None:
        x = pixel_point[:, :, 0:1, :]
        y = pixel_point[:, :, 1:2, :]
        square_r = x ** 2 + y ** 2
        x_distorted = (x * (1 + dist[:, :, :, 0:1] * square_r +
                            dist[:, :, :, 1:2] * square_r ** 2 +
                            dist[:, :, :, 4:5] * square_r ** 3) +
                       2 * dist[:, :, :, 2:3] * x * y +
                       dist[:, :, :, 3:4] * (square_r + 2 * x ** 2))
        y_distorted = (y * (1 + dist[:, :, :, 0:1] * square_r +
                            dist[:, :, :, 1:2] * square_r ** 2 +
                            dist[:, :, :, 4:5] * square_r ** 3) +
                       2 * dist[:, :, :, 3:4] * y * x +
                       dist[:, :, :, 2:3] * (square_r + 2 * y ** 2))
        pixel_point = torch.concat([x_distorted, y_distorted, torch.ones_like(x)], dim=-2)

    pixel_point = torch.matmul(ist, pixel_point)  # [b,n,3,3]*[b,n,3,N]  [b,n,3,N]
    pixel_point = pixel_point.permute(0, 1, 3, 2)  # [b,n,N,3]

    pixel_point = pixel_point[..., :2]  # [b,n,N,2]

    return pixel_point
