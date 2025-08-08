from gpal_lightning.utils.registry import Registry

NECKS = Registry("neck")


def build(global_config, neck_type: str, neck_config: dict):
    """This function is used to build all neck obj given proper name and config"""
    neck_cls = NECKS.get(neck_type)
    return neck_cls(global_config, **neck_config)
