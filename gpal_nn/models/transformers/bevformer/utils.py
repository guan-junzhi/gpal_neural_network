from typing import (Tuple, List, Dict)

import torch
from torch import nn
import copy

class QuantStub(nn.Module):
    def forward(self, x):
        return x

def gen_coords(bev_size: Tuple[int], pc_range: Tuple[float]) -> torch.Tensor:
    """Generate coords."""
    real_w = pc_range[3] - pc_range[0]
    real_h = pc_range[4] - pc_range[1]

    W = bev_size[0]
    H = bev_size[1]

    grid_resolution = (real_w / W, real_h / H)
    real_range = (pc_range[0], pc_range[1], pc_range[3], pc_range[4])

    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(real_range, grid_resolution)

    # Generate a tensor for the x-coordinates of the bird's eye view grid
    x = (torch.linspace(bev_min_x, bev_max_x, W).reshape((1, W)).repeat(H, 1)).double()
    y = (torch.linspace(bev_min_y, bev_max_y, H).reshape((H, 1)).repeat(1, W)).double()
    coords = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords



def get_min_max_coords(
    real_range: List[float], grid_resolution: List[float]) -> Tuple[float]:
    """Get min and max coords."""
    min_x = real_range[0] + grid_resolution[0] / 2
    min_y = real_range[1] + grid_resolution[1] / 2
    max_x = real_range[2] - grid_resolution[0] / 2
    max_y = real_range[3] - grid_resolution[1] / 2
    return min_x, max_x, min_y, max_y


class FFN(nn.Module):
    """The basic structure of FFN.

    Args:
        dim: The inputs dim.
        scale: The scale for inputs dim in hidden layers.
        bias: Whether use bias,
        dropout: Probability of an element to be zeroed.
    """

    def __init__(
        self, dim: int, scale: int = 2, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.ffn1 = nn.Linear(dim, int(dim * scale), bias=bias)
        self.ffn1_act = nn.ReLU(inplace=True)
        self.ffn_dropout1 = nn.Dropout(dropout)
        self.ffn2 = nn.Linear(int(dim * scale), dim, bias=bias)
        self.ffn_dropout2 = nn.Dropout(dropout)
        self.add = FF()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Foward FFN."""
        x = self.ffn1(x)
        x = self.ffn1_act(x)
        x = self.ffn_dropout1(x)
        x = self.ffn2(x)
        x = self.ffn_dropout2(x)
        x = self.add.add(x, skip)
        return x

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        int16_module = [
            self.add,
            self.ffn2,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        from horizon_plugin_pytorch import quantization

        torch.quantization.fuse_modules(
            self,
            ["ffn1", "ffn1_act"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


def batch_apply_distortion(
    norm_coords: torch.Tensor,  # (batch_size, n_cam, num_points, 2)
    dist_coeffs: torch.Tensor   # (batch_size, n_cam, 1, 5)
) -> torch.Tensor:
    """
    params:
        norm_coords: 归一化坐标 [x, y], 形状 (batch_size, n_cam, num_points, 2)
        dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3], 形状 (batch_size, n_cam, 1, 5)
    
    returns:
        distorted_coords: 畸变后的归一化坐标 (batch_size, n_cam, num_points, 2)
    """
    # 分离坐标分量
    x = norm_coords[..., 0]  # (batch_size, n_cam, num_points)
    y = norm_coords[..., 1]
    
    # 计算径向距离及其高次项
    r_sq = x**2 + y**2  # (batch_size, n_cam, num_points)
    r_4 = r_sq**2
    r_6 = r_sq * r_4
    
    # 分离畸变系数（适应新形状）
    k1 = dist_coeffs[..., 0, 0]  # (batch_size, n_cam, 1)
    k2 = dist_coeffs[..., 0, 1]
    p1 = dist_coeffs[..., 0, 2]
    p2 = dist_coeffs[..., 0, 3]
    k3 = dist_coeffs[..., 0, 4]
    
    # 扩展维度用于广播计算
    k1 = k1.unsqueeze(-1)  # (batch_size, n_cam, 1, 1)
    k2 = k2.unsqueeze(-1)
    p1 = p1.unsqueeze(-1)
    p2 = p2.unsqueeze(-1)
    k3 = k3.unsqueeze(-1)
    
    # 径向畸变因子 (1 + k1*r² + k2*r⁴ + k3*r⁶)
    radial_factor = 1 + k1*r_sq + k2*r_4 + k3*r_6
    
    # 切向畸变分量（向量化计算）
    tang_x = 2 * p1 * x * y + p2 * (r_sq + 2 * x**2)
    tang_y = p1 * (r_sq + 2 * y**2) + 2 * p2 * x * y
    
    # 合成畸变坐标
    x_dist = x * radial_factor + tang_x
    y_dist = y * radial_factor + tang_y
    
    return torch.stack([x_dist, y_dist], dim=-1)  # (batch_size, n_cam, num_points, 2)

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



def get_clone_module(module: nn.Module, N: int) -> nn.ModuleList:
    """Get clone nn modules."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
