import copy

import torch

from gpal_lightning.utils.registry import Registry

GRADSCALER = Registry("GradScaler")
GRADSCALER.register_module("AMPScaler", module=torch.cuda.amp.GradScaler)


def build_grad_scaler(grad_scaler_config):
    grad_scaler_config_copy = copy.deepcopy(grad_scaler_config)
    grad_scaler_type = grad_scaler_config_copy.pop("type")
    return GRADSCALER.get(grad_scaler_type)(**grad_scaler_config_copy)
