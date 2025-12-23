from torch.utils.data import dataloader
import numpy as np
import torch

def gpal_collate(batch):
    """Filter out None samples"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        batch = [{}]
    batch_format = {}

    for k in batch[0]:
        if k in ['label', "meta"] and (not isinstance(batch[0][k], (np.ndarray))):
            batch_format[k] = [ele[k] for ele in batch]
        elif k == 'points':
            batch_format[k] = [torch.from_numpy(ele[k]) for ele in batch]
        else:
            batch_format[k] = dataloader.default_collate(
                [item[k] for item in batch])
    return batch_format
