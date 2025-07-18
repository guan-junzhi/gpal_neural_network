import torch


def grads_norm(params: list):
    sum_ = 0.0
    count = 0.0
    for i in range(len(params)):
        if params[i].grad is not None:
            params[i].grad = torch.nan_to_num(params[i].grad)
            sum_ += torch.sqrt(torch.mean(params[i].grad.detach() ** 2))
            count += 1.0

    norm = sum_ / max(count, 1.0)
    return norm.item() if torch.is_tensor(norm) else norm


def reset_saved_grads(params: list, saved_grads: list):
    for i in range(len(params)):
        saved_grads[i] = None


def save_grads(params: list, saved_grads: list):
    for i in range(len(params)):
        if params[i].grad is not None:
            if saved_grads[i] is not None:
                saved_grads[i] += params[i].grad
            else:
                saved_grads[i] = params[i].grad.clone()
            params[i].grad = None


def restore_saved_grads(params: list, saved_grads: list):
    for i in range(len(params)):
        params[i].grad = saved_grads[i]


def scale_grads(params: list, scale):
    for i in range(len(params)):
        if params[i].grad is not None:
            params[i].grad *= scale
