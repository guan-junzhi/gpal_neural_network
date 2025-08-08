import torch


def convert_tensor_to_fp32(data: [dict, torch.Tensor]):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = convert_tensor_to_fp32(data[key])
    elif isinstance(data, torch.Tensor) and data.dtype == torch.uint8:
        data = data.float() / 255.0
        assert data.dtype == torch.float32, f"[Data] Expect data type to be fp32 got {data.dtype} instead"
    else:
        assert isinstance(data, (dict, torch.Tensor)), "[Data] data is neither a dicitonary or a torch.tensor"
        assert data.dtype in [
            torch.uint8,
            torch.float32,
        ], f"[Data] data to convert to fp32 is neither a uint8 or a float32 type{data.dtype}"

    if isinstance(data, torch.Tensor):
        assert (
            data.dtype == torch.float32
        ), "[Data] data after convert to fp32 needs to be of fp32 dtype and within [0, 1]"

    return data

def convert_half_to_single_precision(data: [dict, list, tuple, torch.Tensor]):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [value.float() for value in data]
    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = data[key].float()
    elif isinstance(data, torch.Tensor):
        data = data.float()
    else:
        assert 0, "data with {0} is not supported".format(type(data))

    return data

def _data2tensor(data):
    """

    Args:
        data: numpy array of dict of numpy array after preprocessing

    Returns: tensor vector

    """
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = _data2tensor(val)
    elif isinstance(data, list) or isinstance(data, tuple):
        data = [_data2tensor(itm) for itm in data]
    elif isinstance(data, torch.Tensor):
        pass
    else:
        data = torch.from_numpy(data)
    return data

def tensor2numpy(fun):
    """This decorator turns all tensor to numpy in cpu."""

    def _tensor2numpy(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.detach()
                arg = arg.cpu()
                arg = arg.numpy()

            new_args.append(arg)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                value = value.cpu()
                value = value.numpy()
            new_kwargs[key] = value
        return fun(*new_args, **new_kwargs)

    return _tensor2numpy
