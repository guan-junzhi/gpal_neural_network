import torch


def pt_to_dict(file):
    data = torch.load(file)
    return data
