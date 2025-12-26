import torch.nn as nn
import torch
from typing import Sequence
import os 

from gpal_lightning.neural_network.network_modules.backbones.builder import BACKBONES
from gpal_lightning.neural_network.network_modules.base_module import BaseModule
from gpal_nn.models.base_modules.conv_module import ConvModule2d
from gpal_nn.models.base_modules.basic_henet_module import (
    BasicHENetStageBlock,
    S2DDown,
)


@BACKBONES.register_module()
class HENet(BaseModule):
    """
    Module of HENet.

    Args:
        in_channels: The in_channels for the block.
        block_nums: Number of blocks in each stage.
        embed_dims: Output channels in each stage.
        attention_block_num: Number of attention blocks in each stage.
        mlp_ratios: Mlp expand ratios in each stage.
        mlp_ratio_attn: Mlp expand ratio in attention blocks.
        act_layer: activation layers type.
        use_layer_scale: Use a learnable scale factor in the residual branch.
        layer_scale_init_value: Init value of the learnable scale factor.
        num_classes: Number of classes for a Classifier.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        extra_act: Use extra activation layers in each stage.
        final_expand_channel: Channel expansion before pooling.
        feature_mix_channel: Channel expansion is performed before head.
        block_cls: Basic block types in each stage.
        down_cls: Downsample block types in each stage.
        patch_embed: Stem conv style in the very beginning.
        quant_input: Add a quantization node at the beginning.
        dequant_output: Add a dequantization node at the end.
        stage_out_norm: Add a norm layer to stage outputs.
            Ignored if include_top is True.
        pretrained: Path to pretrained weights or model name
        strict_load: Whether to strictly enforce that the keys in state_dict 
                     match the keys returned by this module's state_dict()
    """

    def __init__(
        self,
        global_config,
        in_channels,
        block_nums,
        embed_dims,
        attention_block_num,
        mlp_ratios,
        mlp_ratio_attn,
        act_layer,
        use_layer_scale,
        extra_act,
        block_cls,
        down_cls,
        layer_scale_init_value: float = 1e-5,
        patch_embed: str = "origin",
        stage_out_norm: bool = False,
        need_neck: bool = False,
        pretrained: str = None,
        strict_load: bool = True,
    ):
        super().__init__(global_config)

        self.stage_out_norm = stage_out_norm
        self.block_cls = block_cls
        self.need_neck = need_neck
        self.pretrained = pretrained
        self.strict_load = strict_load

        if patch_embed in ["origin"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0] // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(embed_dims[0] // 2),
                    act_layer=nn.ReLU(),
                ),
                ConvModule2d(
                    embed_dims[0] // 2,
                    embed_dims[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.ReLU(),
                ),
            )
        elif patch_embed in ["conv_k5_s4"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0],
                    kernel_size=5,
                    stride=4,
                    padding=2,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.GELU(),
                ),
            )
        elif patch_embed in ["conv_4_s4"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.GELU(),
                ),
            )

        stages = []
        downsample_block = []
        for block_idx, block_num in enumerate(tuple(block_nums)):
            stages.append(
                BasicHENetStageBlock(
                    in_dim=embed_dims[block_idx],
                    block_num=block_num,
                    attention_block_num=attention_block_num[block_idx],
                    mlp_ratio=mlp_ratios[block_idx],
                    mlp_ratio_attn=mlp_ratio_attn,
                    act_layer=act_layer[block_idx],
                    use_layer_scale=use_layer_scale[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    extra_act=extra_act[block_idx],
                    block_cls=block_cls[block_idx],
                )
            )
            if block_idx < len(block_nums) - 1:
                assert eval(down_cls[block_idx]) in [S2DDown], down_cls[
                    block_idx
                ]
                downsample_block.append(
                    eval(down_cls[block_idx])(
                        patch_size=2,
                        in_dim=embed_dims[block_idx],
                        out_dim=embed_dims[block_idx + 1],
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.downsample_block = nn.ModuleList(downsample_block)

        
        stage_norm = []
        for embed_dim in embed_dims:
            if self.stage_out_norm is True:
                stage_norm.append(nn.BatchNorm2d(embed_dim))
            else:
                stage_norm.append(nn.Identity())
        self.stage_norm = nn.ModuleList(stage_norm)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Load pretrained weights if specified
        if self.pretrained:
            self.load_pretrained_weights(self.pretrained, self.strict_load)

    def load_pretrained_weights(self, pretrained_path, strict=True):
        """Load pretrained weights from file or model hub"""
        if pretrained_path is None:
            return
            
        print(f"Loading pretrained weights from: {pretrained_path}")
        
        if os.path.isfile(pretrained_path):
            # Load from checkpoint file
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (for DataParallel models)
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            # Filter out incompatible keys
            model_state_dict = self.state_dict()

            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Skip loading parameter {k}, shape mismatch: "
                              f"{v.shape} vs {model_state_dict[k].shape}")
                else:
                    print(f"Skip loading parameter {k}, not found in model")
            
            # Load weights
            self.load_state_dict(filtered_state_dict, strict=strict)
            print(f"Successfully loaded pretrained weights from {pretrained_path}")
            
        elif os.path.isdir(pretrained_path):
            # Try to find the latest checkpoint in directory
            checkpoints = [f for f in os.listdir(pretrained_path) if f.endswith('.pth') or f.endswith('.pt')]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                checkpoint_path = os.path.join(pretrained_path, latest_checkpoint)
                self.load_pretrained_weights(checkpoint_path, strict)
            else:
                print(f"No checkpoint files found in directory: {pretrained_path}")
        else:
            print(f"Pretrained path not found: {pretrained_path}")


    def forward(self, x):
       
        if isinstance(x, Sequence) and len(x) == 1:
            x = x[0]

        x = self.patch_embed(x)
        outs = []
        for idx in range(len(self.stages)):
            x = self.stages[idx](x)
            
            x_normed = self.stage_norm[idx](x)
            
            outs.append(x_normed)
            if idx < len(self.stages) - 1:
                x = self.downsample_block[idx](x)
        if self.need_neck:
            return outs[1], outs[2], outs[3]
        else:
            return [outs[3]]

    def fuse_model(self):
        for module in self.patch_embed:
            module.fuse_model()
        for block in self.downsample_block:
            block.fuse_model()
        for stage in self.stages:
            stage.fuse_model()
        if hasattr(self.final_expand_layer, "fuse_model"):
            self.final_expand_layer.fuse_model()
            block.fuse_model()
        if hasattr(self.feature_mix_layer, "fuse_model"):
            self.feature_mix_layer.fuse_model()



    
    