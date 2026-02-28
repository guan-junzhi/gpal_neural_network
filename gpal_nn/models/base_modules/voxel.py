from typing import List, Optional, Union,Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from horizon_plugin_pytorch.nn.quantized.functional_impl import (_voxelization as horizon_voxelization)#c++代码注册机制



class BatchVoxelization(nn.Module):
    """Batch voxelization.

    Args:
        pc_range: Point cloud range.
        voxel_size: voxel size, (x, y, z) scale.
        max_voxels_num: Max voxel number to use. Defaults to 20000.
        max_points_in_voxel: Number of points in per voxel. Defaults to 30.
    """

    def __init__(
        self,
        pc_range: List[float],
        voxel_size: List[float],
        max_voxels_num: Union[tuple, int] = 20000,
        max_points_in_voxel: int = 30,
    ) -> None:
        super(BatchVoxelization, self).__init__()
        self.pc_range = torch.tensor(pc_range)
        self.voxel_size = torch.tensor(voxel_size)
        self.max_points_in_voxel = max_points_in_voxel
        if isinstance(max_voxels_num, tuple):
            self.max_voxels_num = max_voxels_num
        else:
            self.max_voxels_num = (max_voxels_num, max_voxels_num)

    def forward(
        self,
        points_lst: List[torch.Tensor],
        is_deploy=False,
    ):
        """Forward pass.

        Args:
            points_lst: List of point cloud data.
            is_deploy: Whether is deploy pipeline. Defaults to False.

        Returns:
            Voxel features map.
            Coors of voxel feature.
            Number of point in per voxel.
        """
        if self.training:
            max_voxels = self.max_voxels_num[0]
        else:
            max_voxels = self.max_voxels_num[1]
        device = points_lst[0].device
        voxel_lst: List[torch.Tensor] = []
        coors_lst: List[torch.Tensor] = []
        num_points_per_voxel_lst: List[torch.Tensor] = []
        for points in points_lst:
            # voxelize per points, for batch_size > 1
            voxels, coors, num_points_per_voxel = horizon_voxelization(
                points,
                voxel_size=self.voxel_size.to(device),
                pc_range=self.pc_range.to(device),
                max_points_per_voxel=self.max_points_in_voxel,
                max_voxels=max_voxels,
                use_max=is_deploy,
            )
            voxel_lst.append(voxels)
            coors_lst.append(coors)
            num_points_per_voxel_lst.append(num_points_per_voxel)

        voxel_feature = torch.cat(voxel_lst, dim=0)
        num_points_per_voxel = torch.cat(num_points_per_voxel_lst, dim=0)

        # Pad first element of coord according the index in batch_data.
        # Example:
        #   batch_data = [data1, data2], and batch_size = 2,
        #   batch_data.index(data1) = 0, batch_data.index(data2) = 1,
        #   for data1:  coord (z, y, x) --> Pad 0 --> coord (0, z, y, x)
        #   for data2:  coord (z, y, x) --> Pad 1 --> coord (1, z, y, x)
        coors_batch: List[torch.Tensor] = []
        for i, coor in enumerate(coors_lst):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=float(i))
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0).long()

        return voxel_feature, coors_batch, num_points_per_voxel
