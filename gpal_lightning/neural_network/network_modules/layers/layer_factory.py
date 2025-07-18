import logging
from typing import Tuple

from torch import nn

from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.network_modules.layers.channel_to_spatial import Channel2Spatial

CUSTOMIZED_OPERATIONS = {Channel2Spatial}


class LayerFactory:
    """
    LayerFactory is used to generate the combination of (conv, norm, activate) layer
    """

    @classmethod
    def build(
        cls,
        global_config: GlobalConfig,
        layer_types: Tuple = ("conv", "norm", "acti"),
        conv_class=nn.Conv2d,
        conv_config: dict = None,
        norm_class=nn.BatchNorm2d,
        norm_config: dict = None,
        acti_class=nn.ReLU,
        acti_config: dict = None,
        pool_class=nn.AvgPool2d,
        pool_config: dict = None,
        dropout_class=nn.Dropout,
        dropout_config: dict = None,
        linear_class=nn.Linear,
        linear_config: dict = None,
        deconv_class=nn.ConvTranspose2d,
        deconv_config: dict = None,
        channel2spatial_config: dict = None,
    ) -> nn.Sequential:
        """

        Args:
            deconv_config:
            deconv_class:
            global_config: instance of global config, required by base layer.
            layer_types: a tuple contains the order of layers to be generated
            conv_class: class object for conv layer, by default it's the nn.Conv2d
            conv_config: dictionary contains the parameters needed for conv layer, it will be passed as
            the kwarg to conv class object. Please note bias parameter is controlled based on
            the existence of norm layer.
            norm_class: class object for norm layer, by default it's the customized nn.Batchnorm2d
            norm_config: dictionary contains the parameters needed for norm layer, it will be passed as
            the kwarg to norm class object
            acti_class: class object for activate layer, by default it's the nn.ReLU
            acti_config:dictionary contains the parameters needed for activate layer, it will be passed as
            the kwarg to activate class object
            pool_class: class object for pooling layer
            pool_config: dict contains pooling parameter
            dropout_class:
            dropout_config:
            linear_class:
            linear_config:
            channel2spatial_config: config for Channel2Spatial class.

        Returns: a nn Squental object contains all the layers in layer_types

        """
        layers = []
        for layer_type in layer_types:
            if layer_type == "conv":
                if "bias" in conv_config and conv_config["bias"]:
                    logging.warning(
                        "bias is included in the args but it should be controlled "
                        "based on the layer types, ignore the args"
                    )
                    conv_config.pop("bias")
                if "norm" in layer_types:
                    bias = False
                else:
                    bias = True
                conv_config = conv_config.copy()
                conv_config["bias"] = bias
                layer = build(global_config, conv_class, conv_config)
            elif layer_type == "norm":
                layer = build(global_config, norm_class, norm_config)
            elif layer_type == "acti":
                layer = build(global_config, acti_class, acti_config)
            elif layer_type == "pool":
                layer = build(global_config, pool_class, pool_config)
            elif layer_type == "dropout":
                layer = build(global_config, dropout_class, dropout_config)
            elif layer_type == "linear":
                layer = build(global_config, linear_class, linear_config)
            elif layer_type == "deconv":
                layer = build(global_config, deconv_class, deconv_config)
            elif layer_type == "channel2spatial":
                layer = build(global_config, Channel2Spatial, channel2spatial_config)
            else:
                raise NotImplementedError(f"Got unexpected layer type: {layer_type}")
            layers.append(layer)
        layers = nn.Sequential(*layers)

        return layers


def build(global_config, layer_class, layer_config):
    if layer_config is None:
        layer_config = {}
    if layer_class in CUSTOMIZED_OPERATIONS:
        return layer_class(global_config, **layer_config)

    return layer_class(**layer_config)
