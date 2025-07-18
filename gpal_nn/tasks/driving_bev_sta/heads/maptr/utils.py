from typing import Optional
import torch
from torch import nn
import numpy as np
import copy
from functools import partial



def box_center_to_corner(
    bboxes: torch.Tensor,
    split: Optional[bool] = False,
    legacy_bbox: Optional[bool] = False,
):  # noqa: D205,D400
    """
    Convert bounding box from center format (xcenter, ycenter,
    width, height) to corner format (x_low, y_low, x_high, y_high)

    Args:
        bboxes: Shape is (..., 4) represents bounding boxes.
        split: Whether to split the final output to
            for (..., 1) tensors, or keep the (..., 4) original output.
            Default to False.
        legacy_bbox: Whether the boxes are decoded
            in legacy manner (should add one to bottom or right coordinate
            before using) or not. Default to False.
    """

    border = int(legacy_bbox)
    cx, cy, w, h = torch.split(bboxes, 1, dim=-1)
    x1 = cx - (w - border) * 0.5
    y1 = cy - (h - border) * 0.5
    x2 = x1 + w - border
    y2 = y1 + h - border

    if split:
        return x1, y1, x2, y2
    else:
        return torch.cat([x1, y1, x2, y2], dim=-1)


def box_corner_to_center(
    bboxes: torch.Tensor,
    split: Optional[bool] = False,
    legacy_bbox: Optional[bool] = False,
):  # noqa: D205,D400
    """
    Convert bounding box from corner format (x_low, y_low, x_high, y_high)
    to center format (xcenter, ycenter, width, height)

    Args:
        bboxes: Shape is (..., 4) represents bounding boxes.
        split: Whether to split the final output to
            for (..., 1) tensors, or keep the (..., 4) original output.
            Default to False.
        legacy_bbox: Whether the boxes are decoded
            in legacy manner (should add one to bottom or right coordinate
            before using) or not. Default to False.
    """

    border = int(legacy_bbox)
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
    width = x2 - x1 + border
    height = y2 - y1 + border
    cx = x1 + (width - border) * 0.5
    cy = y1 + (height - border) * 0.5

    if split:
        return cx, cy, width, height
    else:
        return torch.cat([cx, cy, width, height], dim=-1)



def bias_init_with_prob(prior_prob: int) -> float:
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def get_clone_module(module: nn.Module, N: int) -> nn.ModuleList:
    """Get clone nn modules."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Initialize conv/fc bias with constant value."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1,
    bias: float = 0,
    distribution: str = "normal",
):
    """Initialize conv/fc bias with xavier method."""
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)



class QuantStub(nn.Module):
    r"""Refine this docstring in the future.

    Same as torch.nn.QuantStub, with an additional param to
    specify a fixed scale.

    Args:
        scale (float, optional): Pass a number to use as fixed scale.
            If set to None, scale will be computed by observer during forward.
            Defaults to None.
        zero_point (int, optional): Pass a number to use as fixed zero_point.
            Defaults to None.
        qconfig (optional): Quantization configuration for the tensor, if
            qconfig is not provided, we will get qconfig from parent modules.
            Defaults to None.
    """

    def __init__(
        self, scale: float = None, zero_point: int = None, qconfig=None
    ):
        super(QuantStub, self).__init__()
        if scale is not None and zero_point is None:
            zero_point = 0

        self.scale = scale
        self.zero_point = zero_point
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x
