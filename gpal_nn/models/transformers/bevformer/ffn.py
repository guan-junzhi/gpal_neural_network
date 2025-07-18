import torch
from torch import nn

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
        # self.add = FF()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Foward FFN."""
        x = self.ffn1(x)
        x = self.ffn1_act(x)
        x = self.ffn_dropout1(x)
        x = self.ffn2(x)
        x = self.ffn_dropout2(x)
        x = torch.add(x, skip)
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
