import torch
from typing import Any, Dict, Union


def convert_dict_to_numpy(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归处理字典,将PyTorch张量转换为numpy数组
    
    Args:
        data: 包含PyTorch张量的字典
        
    Returns:
        转换后的字典,张量已转为numpy数组
    """
    converted_data = {}
    
    for key, value in data.items():
        if torch.is_tensor(value):
            # 检查是否在CUDA上
            if value.is_cuda:
                print(f"Key '{key}': CUDA张量 -> 转移到CPU并转换为numpy")
                converted_data[key] = value.cpu().detach().numpy()
            else:
                print(f"Key '{key}': CPU张量 -> 直接转换为numpy")
                converted_data[key] = value.detach().numpy()
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            print(f"Key '{key}': 嵌套字典 -> 递归处理")
            converted_data[key] = convert_dict_to_numpy(value)
        elif isinstance(value, (list, tuple)):
            # 处理列表或元组中的张量
            converted_data[key] = convert_sequence_to_numpy(value)
        else:
            # 其他类型直接保留
            converted_data[key] = value
            print(f"Key '{key}': {type(value).__name__} -> 保持不变")
    
    return converted_data

def convert_sequence_to_numpy(sequence: Union[list, tuple]) -> Union[list, tuple]:
    """
    处理列表或元组中的张量
    """
    converted = []
    for item in sequence:
        if torch.is_tensor(item):
            if item.is_cuda:
                converted.append(item.cpu().detach().numpy())
            else:
                converted.append(item.detach().numpy())
        elif isinstance(item, dict):
            converted.append(convert_dict_to_numpy(item))
        elif isinstance(item, (list, tuple)):
            converted.append(convert_sequence_to_numpy(item))
        else:
            converted.append(item)
    
    return type(sequence)(converted)


