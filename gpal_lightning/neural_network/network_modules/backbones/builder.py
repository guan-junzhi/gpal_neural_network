from gpal_lightning.utils.registry import Registry

BACKBONES = Registry("backbone")


def build(global_config, backbone_type: str, backbone_config: dict):
    """This function is used to build all backbone obj given proper name and config"""
    backbone_cls = BACKBONES.get(backbone_type)
    return backbone_cls(global_config=global_config, **backbone_config)
