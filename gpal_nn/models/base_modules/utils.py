import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def _get_paddings_indicator(
    actual_num: torch.Tensor, max_num: int, axis=0
) -> torch.Tensor:
    """Create boolean mask by actual number of a padded tensor.

    This function helps to identify pillars where there's too little data.

    Example:

    actual_num = [[3,3,3,3,3]] (5 pillars, each contains 3 lidar points)
    max_num: 4 (turns to [[0, 1, 2, 3, 4]])
    will return: [[T, T, T, F, F]]

    Args:
        actual_num (torch.Tensor): NxM tensor, where N is batch size and M is
            total number of pillars. In certain cases N can be omitted.
        max_num (int): max number of points allowed in a pillar.
        axis (int, optional): axis position. Defaults to 0.

    Returns:
        [torch.Tensor]: indicates where the tensor should be padded.
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device
    ).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator