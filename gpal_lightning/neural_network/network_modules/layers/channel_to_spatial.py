from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.network_modules.base_layer import BaseLayer


class Channel2Spatial(BaseLayer):
    """Channel to spatial is essentially the inverse of Spatial2Channel.
    It's now serve as a way to upsample feature map.
    """

    def __init__(
        self,
        global_config: GlobalConfig,
        in_c: int,
        in_h: int,
        in_w: int,
        ratio: int,
    ):
        super().__init__(global_config)
        self.ratio = ratio
        self.in_c = in_c
        self.in_h = in_h
        self.in_w = in_w

        assert self.in_c % (self.ratio**2) == 0, f"in_channel: {self.in_c}, ratio:{self.ratio}"
        self.out_c = self.in_c // (self.ratio**2)
        self.out_h = self.in_h * self.ratio
        self.out_w = self.in_w * self.ratio

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x = x.view(self.batch_size, self.in_h, self.in_w, self.out_c, self.ratio, self.ratio)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(self.batch_size, self.out_c, self.out_h, self.out_w)
        return x
