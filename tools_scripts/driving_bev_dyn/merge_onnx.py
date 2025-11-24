import onnx
import onnxruntime as ort
import numpy as np


def enhanced_merge_onnx_models(fisheye_path, pinhole_path, output_path):
    """简化的ONNX模型合并方法：重命名鱼眼模型后直接拼接"""
    
    print("=== 加载模型 ===")
    fisheye_model = onnx.load(fisheye_path)
    pinhole_model = onnx.load(pinhole_path)
    
    print("=== 检查原始模型graph名称 ===")
    print(f"鱼眼模型graph名称: {fisheye_model.graph.name}")
    print(f"针孔模型graph名称: {pinhole_model.graph.name}")
    
    # 确保原始模型有graph名称
    if not fisheye_model.graph.name:
        fisheye_model.graph.name = "fisheye_original_graph"
    if not pinhole_model.graph.name:
        pinhole_model.graph.name = "pinhole_original_graph"
    
    
    print("=== 创建合并的计算图 ===")
    merged_graph = onnx.GraphProto()
    
    # 设置graph名称
    merged_graph.name = "merged_fisheye_pinhole_graph"
    
    # 1. 直接拼接所有节点（保持原有顺序）
    print("直接拼接鱼眼模型节点...")
    merged_graph.node.extend(fisheye_model.graph.node)
    
    print("直接拼接针孔模型节点...")
    merged_graph.node.extend(pinhole_model.graph.node)
    
    # 2. 合并所有输入（不去重，保持所有输入）
    print("合并所有输入...")
    for input_tensor in fisheye_model.graph.input:
        merged_graph.input.extend([input_tensor])
    
    for input_tensor in pinhole_model.graph.input:
        merged_graph.input.extend([input_tensor])
    
    # 3. 合并所有输出（不去重，保持所有输出）
    print("合并所有输出...")
    for output_tensor in fisheye_model.graph.output:
        merged_graph.output.extend([output_tensor])
    
    for output_tensor in pinhole_model.graph.output:
        merged_graph.output.extend([output_tensor])
    
    # 4. 合并所有初始值（不去重，保持所有初始值）
    print("合并所有初始值...")
    for initializer in fisheye_model.graph.initializer:
        merged_graph.initializer.extend([initializer])
    
    for initializer in pinhole_model.graph.initializer:
        merged_graph.initializer.extend([initializer])
    
    # 5. 合并所有value_info（中间张量信息）
    print("合并中间张量信息...")
    for value_info in fisheye_model.graph.value_info:
        merged_graph.value_info.extend([value_info])
    
    for value_info in pinhole_model.graph.value_info:
        merged_graph.value_info.extend([value_info])
    
    print("=== 创建合并模型 ===")
    # 创建合并模型
    merged_model = onnx.helper.make_model(
        merged_graph,
        producer_name='simplified-onnx-merge-tool',
        opset_imports=fisheye_model.opset_import
    )
    
    print("=== 验证合并模型 ===")
    # 验证模型
    try:
        # 使用临时文件验证
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
            onnx.save(merged_model, temp_file.name)
            
            # 检查模型是否有效
            onnx.checker.check_model(merged_model)
            
            # 创建推理会话验证
            session = ort.InferenceSession(temp_file.name)
            
            print("✅ 模型验证通过!")
            print(f"输入数量: {len(session.get_inputs())}")
            print(f"输出数量: {len(session.get_outputs())}")
            print(f"计算节点数量: {len(merged_model.graph.node)}")
            print(f"Graph名称: {merged_model.graph.name}")
            
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        
    
    print("=== 保存合并模型 ===")
    onnx.save(merged_model, output_path)
    print(f"✅ 合并模型已保存至: {output_path}")
    
    return merged_model




def create_test_inputs_for_models(fisheye_model_path, pinhole_model_path, seed=42):
    """为两个模型创建测试输入数据（使用固定随机种子确保一致性）"""
    print("=== 创建测试输入数据（使用固定种子确保一致性） ===")
    
    # 设置随机种子确保每次生成相同的数据
    np.random.seed(seed)
    
    # 加载模型以获取输入信息
    fisheye_model = onnx.load(fisheye_model_path)
    pinhole_model = onnx.load(pinhole_model_path)
    
    test_inputs = {}
    
    # 为鱼眼模型创建测试输入
    fisheye_inputs = {}
    for input_tensor in fisheye_model.graph.input:
        # 获取输入的形状和数据类型
        tensor_type = input_tensor.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)  # 对于动态维度，使用1作为测试值
        
        # 根据数据类型创建测试数据
        if tensor_type.elem_type == onnx.TensorProto.FLOAT:
            # 创建固定随机浮点数数据
            data = np.random.randn(*shape).astype(np.float32)
        elif tensor_type.elem_type == onnx.TensorProto.INT32:
            # 创建固定随机整数数据
            data = np.random.randint(0, 10, size=shape).astype(np.int32)
        elif tensor_type.elem_type == onnx.TensorProto.INT64:
            # 创建固定随机整数数据
            data = np.random.randint(0, 10, size=shape).astype(np.int64)
        else:
            # 默认使用浮点数
            data = np.random.randn(*shape).astype(np.float32)
        
        fisheye_inputs[input_tensor.name] = data
        print(f"鱼眼模型输入 {input_tensor.name}: 形状 {shape}, 类型 {tensor_type.elem_type}")
    
    # 为针孔模型创建测试输入
    pinhole_inputs = {}
    for input_tensor in pinhole_model.graph.input:
        # 获取输入的形状和数据类型
        tensor_type = input_tensor.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)  # 对于动态维度，使用1作为测试值
        
        # 根据数据类型创建测试数据
        if tensor_type.elem_type == onnx.TensorProto.FLOAT:
            # 创建固定随机浮点数数据
            data = np.random.randn(*shape).astype(np.float32)
        elif tensor_type.elem_type == onnx.TensorProto.INT32:
            # 创建固定随机整数数据
            data = np.random.randint(0, 10, size=shape).astype(np.int32)
        elif tensor_type.elem_type == onnx.TensorProto.INT64:
            # 创建固定随机整数数据
            data = np.random.randint(0, 10, size=shape).astype(np.int64)
        else:
            # 默认使用浮点数
            data = np.random.randn(*shape).astype(np.float32)
        
        pinhole_inputs[input_tensor.name] = data
        print(f"针孔模型输入 {input_tensor.name}: 形状 {shape}, 类型 {tensor_type.elem_type}")
    
    test_inputs['fisheye'] = fisheye_inputs
    test_inputs['pinhole'] = pinhole_inputs
    
    # 重置随机种子，避免影响其他代码
    np.random.seed(None)
    
    return test_inputs

def run_model_inference(model_path, input_data):
    """运行模型推理并返回输出"""
    try:
        session = ort.InferenceSession(model_path)
        
        # 准备输入数据
        feed_dict = {}
        for input_info in session.get_inputs():
            input_name = input_info.name
            if input_name in input_data:
                feed_dict[input_name] = input_data[input_name]
            else:
                # 如果输入数据中没有对应的名称，创建默认数据
                shape = input_info.shape
                # 将动态维度(-1)替换为1
                shape = [1 if dim == -1 else dim for dim in shape]
                feed_dict[input_name] = np.random.randn(*shape).astype(np.float32)
        
        # 运行推理
        outputs = session.run(None, feed_dict)
        
        # 获取输出名称
        output_names = [output.name for output in session.get_outputs()]
        result = dict(zip(output_names, outputs))
        
        return result
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        return None

def compare_model_outputs_with_renaming(original_outputs, merged_outputs, prefix, tolerance=1e-5):
    """比较重命名后的模型输出是否一致"""
    print(f"=== 比较{prefix}模型输出（考虑重命名） ===")
    
    if original_outputs is None or merged_outputs is None:
        print(f"❌ 无法比较{prefix}模型输出，其中一个为None")
        return False
    
    all_match = True
    
    # 遍历原始模型的每个输出
    for original_output_name, original_output in original_outputs.items():
        # 计算合并后的输出名称（添加前缀）
        merged_output_name = f"{prefix}_output_{original_output_name}"
        merged_output_name = original_output_name
        
        if merged_output_name not in merged_outputs:
            print(f"❌ 重命名后的输出 {merged_output_name} 在合并模型中不存在")
            all_match = False
            continue
        
        merged_output = merged_outputs[merged_output_name]
        
        # 检查形状是否一致
        if original_output.shape != merged_output.shape:
            print(f"❌ 输出 {original_output_name} -> {merged_output_name} 形状不一致: {original_output.shape} vs {merged_output.shape}")
            all_match = False
            continue
        
        # 检查数值是否一致（在容忍度范围内）
        if np.allclose(original_output, merged_output, rtol=tolerance, atol=tolerance):
            print(f"✅ 输出 {original_output_name} -> {merged_output_name} 匹配 (形状: {original_output.shape})")
        else:
            # 计算差异统计
            diff = np.abs(original_output - merged_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            print(f"❌ 输出 {original_output_name} -> {merged_output_name} 不匹配")
            print(f"   最大差异: {max_diff:.6e}")
            print(f"   平均误差: {mean_diff:.6e}")
            print(f"   标准差: {std_diff:.6e}")
            print(f"   形状: {original_output.shape}")
            all_match = False
    
    return all_match
    

def test_merged_model_with_inputs(fisheye_path, pinhole_path, merged_path, test_inputs):
    """使用测试输入数据验证合并模型"""
    print("=== 使用测试输入验证合并模型 ===")
    
    # 运行原始鱼眼模型推理
    print("运行鱼眼模型推理...")
    fisheye_outputs = run_model_inference(fisheye_path, test_inputs['fisheye'])
    
    # 运行原始针孔模型推理
    print("运行针孔模型推理...")
    pinhole_outputs = run_model_inference(pinhole_path, test_inputs['pinhole'])
    
    # 运行合并模型推理
    print("运行合并模型推理...")
    
    # 准备合并模型的输入数据（两个模型的输入合并）
    merged_inputs = {}
    merged_inputs.update(test_inputs['fisheye'])
    merged_inputs.update(test_inputs['pinhole'])
    
    merged_outputs = run_model_inference(merged_path, merged_inputs)
    
    if fisheye_outputs is None or pinhole_outputs is None or merged_outputs is None:
        print("❌ 模型推理失败，无法进行比较")
        return False
    
    # 正确比较重命名后的输出
    print("比较鱼眼模型输出（考虑重命名）...")
    fisheye_match = compare_model_outputs_with_renaming(fisheye_outputs, merged_outputs, "fisheye")
    
    print("比较针孔模型输出（考虑重命名）...")
    pinhole_match = compare_model_outputs_with_renaming(pinhole_outputs, merged_outputs, "pinhole")
    
    if fisheye_match and pinhole_match:
        print("✅ 所有模型输出匹配！合并模型功能正确。")
        return True
    else:
        print("❌ 模型输出不匹配！合并模型可能存在功能问题。")
        return False


if __name__ == "__main__":
    
    # 使用增强合并方法
    fisheye_path = "prefixed_model.onnx"
    pinhole_path = ".vscode/workspace_huiquyang/20251116_07_00_57_onnx/checkpoint/epoch=12-step=80000_checkpoint_sim.onnx"
    merged_path = "merged_validated.onnx"
    
    merged_model = enhanced_merge_onnx_models(fisheye_path, pinhole_path, merged_path)
    
    # 创建测试输入数据并进行真正的功能验证
    print("\n" + "="*50)
    print("开始真正的输入测试验证")
    print("="*50)
    
    # 创建测试输入数据
    test_inputs = create_test_inputs_for_models(fisheye_path, pinhole_path)
    
    # 使用测试输入验证合并模型
    test_passed = test_merged_model_with_inputs(fisheye_path, pinhole_path, merged_path, test_inputs)
    
    if test_passed:
        print("\n🎉 所有测试通过！合并模型功能完整且正确。")
    else:
        print("\n⚠️ 测试未通过，请检查合并模型的功能完整性。")