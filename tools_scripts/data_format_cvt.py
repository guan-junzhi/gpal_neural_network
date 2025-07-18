import torch
import numpy as np


def ShowDataStruct(head_str, d, indent=0, depth=-1):
    if isinstance(d, (torch.Tensor, np.ndarray)):
        return f"{'    ' * indent}{head_str}{type(d)} : {d.shape}\n"
    elif isinstance(d, (list, set, tuple)):
        str = f"{'    ' * indent}{head_str}{type(d)} len = {len(d)}\n"
        if (depth == -1) or (indent < depth):
            return str + "".join([ShowDataStruct(f"{ele_i}", ele, indent+1, depth) for ele_i, ele in enumerate(d)])
        else:
            return str
    elif isinstance(d, (dict)):
        str = f"{'    ' * indent}{head_str}{type(d)}\n"
        if (depth == -1) or (indent < depth):
            return str + "".join([ShowDataStruct(f"{k}", d[k], indent+1, depth) for k in d])
        else:
            return str
    elif isinstance(d, (int, float, type(""), type(None))):
        return f"{'    ' * indent}{head_str}{type(d)}:{d}\n"
    else:
        return f"{'    ' * indent}unknow type : {head_str}{type(d)}\n"


def ToNumpy(d):
    if isinstance(d, (torch.Tensor)):
        return d.cpu().numpy()
    elif isinstance(d, (np.ndarray)):
        return d
    else:
        raise (f"type err {type(d)}")


def ImgFloatTesnor2Uint8Array(img_tensor):
    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    for i in range(3):
        img[..., i] = img[..., i] * \
            img_norm_cfg["std"][i] + img_norm_cfg["mean"][i]

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
