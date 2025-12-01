import onnx
from onnx import helper
import numpy as np

def safe_add_prefix(model_path, output_path, prefix, verify=True):
    """
    安全地添加前缀，确保输出一致性
    
    Args:
        model_path: 输入模型路径
        output_path: 输出模型路径
        prefix: 要添加的前缀
        verify: 是否验证输出一致性
    """
    # 加载模型
    model = onnx.load(model_path)
    original_model = onnx.load(model_path)  # 保留原始副本用于验证
    
    # 记录原始输入输出名称
    original_input_names = [input.name for input in model.graph.input]
    original_output_names = [output.name for output in model.graph.output]
    
    print(f"原始输入名称: {original_input_names}")
    print(f"原始输出名称: {original_output_names}")
    
    # 创建名称映射（保护输入输出名称）
    name_map = {}
    
    # 1. 收集所有需要重命名的张量名称（不包括输入输出）
    all_tensor_names = set()
    
    # 收集节点输出
    for node in model.graph.node:
        for output_name in node.output:
            if output_name and output_name not in original_output_names:
                all_tensor_names.add(output_name)
    
    # 收集初始器名称
    for initializer in model.graph.initializer:
        if initializer.name not in original_input_names:
            all_tensor_names.add(initializer.name)
    
    # 收集值信息名称
    for value_info in model.graph.value_info:
        if value_info.name not in original_output_names:
            all_tensor_names.add(value_info.name)
    
    # 2. 重命名所有中间张量
    for tensor_name in all_tensor_names:
        if tensor_name not in original_input_names and tensor_name not in original_output_names:
            new_name = f"{prefix}_{tensor_name}"
            name_map[tensor_name] = new_name
    
    # 3. 重命名所有节点（如果有名称）
    for node in model.graph.node:
        if node.name:  # 确保节点有名称
            original_name = node.name
            new_name = f"{prefix}_{original_name}"
            name_map[original_name] = new_name
            node.name = new_name
    
    # 4. 更新所有引用
    # 更新节点输入
    for node in model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] in name_map:
                node.input[i] = name_map[node.input[i]]
    
    # 更新节点输出
    for node in model.graph.node:
        for i in range(len(node.output)):
            if node.output[i] in name_map:
                node.output[i] = name_map[node.output[i]]
    
    # 更新初始器名称
    for initializer in model.graph.initializer:
        if initializer.name in name_map:
            initializer.name = name_map[initializer.name]
    
    # 更新值信息名称
    for value_info in model.graph.value_info:
        if value_info.name in name_map:
            value_info.name = name_map[value_info.name]
    
    # 保存模型
    onnx.save(model, output_path)
    print(f"模型已保存到: {output_path}")
    
    # 验证输出一致性
    if verify:
        try:
            is_consistent = verify_output_consistency(original_model, model, prefix)
            if not is_consistent:
                print("警告: 添加前缀后输出可能存在不一致")
            else:
                print("验证通过: 输出保持一致")
        except Exception as e:
            print(f"验证过程中出现错误: {e}")
            print("建议检查重命名后的模型是否有效")
    
    return model

def verify_output_consistency(original_model, modified_model, prefix, rtol=1e-5, atol=1e-8):
    """验证两个模型的输出一致性"""
    import onnxruntime as ort
    import numpy as np
    
    # 创建推理会话
    orig_session = ort.InferenceSession(original_model.SerializeToString())
    mod_session = ort.InferenceSession(modified_model.SerializeToString())
    
    # 获取原始输入输出名称
    orig_input_names = [input.name for input in orig_session.get_inputs()]
    orig_output_names = [output.name for output in orig_session.get_outputs()]
    
    # 获取修改后模型的输入输出名称
    mod_input_names = [input.name for input in mod_session.get_inputs()]
    mod_output_names = [output.name for output in mod_session.get_outputs()]
    
    print(f"原始模型输入: {orig_input_names}")
    print(f"原始模型输出: {orig_output_names}")
    print(f"修改后模型输入: {mod_input_names}")
    print(f"修改后模型输出: {mod_output_names}")
    
    # 检查输入输出名称是否匹配
    if set(orig_input_names) != set(mod_input_names):
        print("警告: 输入名称不匹配，尝试自动映射")
    
    if set(orig_output_names) != set(mod_output_names):
        print("警告: 输出名称不匹配，尝试自动映射")
    
    # 创建相同输入
    input_data = {}
    for input in orig_session.get_inputs():
        shape = [dim if dim > 0 else 1 for dim in input.shape]
        input_data[input.name] = np.random.rand(*shape).astype(np.float32)
    
    # 原始模型推理
    orig_outputs = orig_session.run(None, input_data)
    
    # 修改后模型推理（使用正确的输入名称）
    mod_input_data = {}
    for orig_name, data in input_data.items():
        # 优先使用原始名称
        if orig_name in mod_input_names:
            mod_input_data[orig_name] = data
        else:
            # 尝试添加前缀
            prefixed_name = f"{prefix}_{orig_name}"
            if prefixed_name in mod_input_names:
                mod_input_data[prefixed_name] = data
            else:
                # 使用第一个可用的输入名称
                if mod_input_names:
                    mod_input_data[mod_input_names[0]] = data
                    print(f"使用映射: {orig_name} -> {mod_input_names[0]}")
                else:
                    raise ValueError("修改后模型没有可用的输入")
    
    mod_outputs = mod_session.run(None, mod_input_data)
    
    # 比较输出
    if len(orig_outputs) != len(mod_outputs):
        print(f"输出数量不一致: 原始模型 {len(orig_outputs)} 个输出, 修改后模型 {len(mod_outputs)} 个输出")
        return False
    
    all_consistent = True
    for i, (orig_out, mod_out) in enumerate(zip(orig_outputs, mod_outputs)):
        if orig_out.shape != mod_out.shape:
            print(f"输出 {i} 形状不一致: 原始 {orig_out.shape}, 修改后 {mod_out.shape}")
            all_consistent = False
            continue
            
        if not np.allclose(orig_out, mod_out, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(orig_out - mod_out))
            print(f"输出 {i} 数值不一致，最大差异: {max_diff}")
            all_consistent = False
        else:
            print(f"输出 {i} 一致")
    
    return all_consistent

# 使用示例
if __name__ == "__main__":
    safe_add_prefix(".vscode/workspace_huiquyang/20251120_03_33_17_onnx/checkpoint/epoch=24-step=81000_checkpoint_sim.onnx", "prefixed_model.onnx", "fisheye")
