from gpal_nn.models.backbones.resnet import ResNet, EncoderRes50, EncoderRes34
from gpal_nn.models.backbones.dummy_backbone import DummyBackbone
from gpal_nn.models.backbones.DDRNet_23_slim import DualResShare, DualResNet
__all__ = ['ResNet', 'DummyBackbone',
           'DualResShare', 'DualResNet', 'EncoderRes50', 'EncoderRes34']
