import os

from gpal_lightning import const


def get_checkpoint_path(root):
    checkpoint = os.path.join(
        root, const.CHECKPOINT_PATH, const.CHECKPOINT_NAME_LAST + const.FILE_EXTENSION)
    return checkpoint


def get_config_path(root):
    config = os.path.join(root, const.CONFIG_NAME + const.CONFIG_EXTENSION)
    return config
