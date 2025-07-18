from typing import Union


def warmup_lr(
    warmup_epochs: Union[int, float], epoch_percentage: Union[int, float], optimizer, learning_rate: float
) -> float:
    """This function is used to warm up the learning rate in optimizer"""
    lr_scale = min(1.0, epoch_percentage / warmup_epochs)
    actual_lr = lr_scale * learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = actual_lr
    return actual_lr
