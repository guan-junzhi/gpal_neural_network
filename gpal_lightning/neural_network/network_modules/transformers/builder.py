from gpal_lightning.utils.registry import Registry

TRANSFORMERS = Registry("transformer")


def build(global_config, transformer_type: str, transformer_config: dict):
    """This function is used to build all transformers obj given proper name and config"""
    neck_cls = TRANSFORMERS.get(transformer_type)
    return neck_cls(global_config, **transformer_config)
