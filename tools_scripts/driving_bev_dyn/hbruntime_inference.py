#!/usr/bin/env python3
"""
使用HBRuntime进行ONNX模型推理的工具
专门处理horizon自定义算子问题
"""

import numpy as np
import onnx
from horizon_tc_ui import HBRuntime
from pathlib import Path
import json

def create_hbruntime_session(model_path):
    """
    创建HBRuntime推理会话
    
    Args:
        model_path: ONNX模型路径
    
    Returns:
        HBRuntime: HBRuntime对象
    """
    try:
        # 根据正确的使用方式创建HBRuntime会话
        # HBRuntime构造函数直接接受模型路径
        session = HBRuntime(model_path)
        
        print("✅ HBRuntime会话创建成功")
        return session
        
    except Exception as e:
        print(f"❌ 创建HBRuntime会话失败: {e}")
        raise

def get_model_io_info(model_path):
    """
    通过分析ONNX模型文件获取输入输出信息
    
    Args:
        model_path: ONNX模型路径
    
    Returns:
        tuple: (输入名称列表, 输出名称列表)
    """
    try:
        # 加载ONNX模型来分析输入输出
        model = onnx.load(model_path)
        
        # 获取输入名称
        input_names = [input.name for input in model.graph.input]
        
        # 获取输出名称
        output_names = [output.name for output in model.graph.output]
        
        # 过滤掉初始值（initializer），它们不是真正的输入
        initializer_names = [init.name for init in model.graph.initializer]
        input_names = [name for name in input_names if name not in initializer_names]
        
        print("📊 模型输入输出信息:")
        print("   输入:")
        for i, name in enumerate(input_names):
            print(f"     {i}: {name}")
        
        print("   输出:")
        for i, name in enumerate(output_names):
            print(f"     {i}: {name}")
        
        return input_names, output_names
        
    except Exception as e:
        print(f"❌ 通过ONNX分析获取IO信息失败: {e}")
        
        # 如果分析失败，根据文件名猜测模型类型
        model_name = Path(model_path).stem.lower()
        
        if 'fisheye' in model_name or 'fish' in model_name:
            # 鱼眼模型输入
            input_names = ['img_front_fisheye', 'img_right_fisheye', 'img_rear_fisheye', 
                          'img_left_fisheye', 'fish_images_grid', 'fish_vt_grid',
                          'fish_prev_feats', 'fish_prev_feats_grid']
            output_names = ['fish_head_conv', 'fish_hm_center', 'fish_prev_feats_output']
        else:
            # 针孔模型输入
            input_names = ['img_front_120', 'img_front_30', 'img_back', 'img_front_left',
                          'img_front_right', 'img_rear_left', 'img_rear_right', 'images_grid', 
                          'vt_grid', 'prev_feats', 'prev_feats_grid']
            output_names = ['head_conv', 'hm_center', 'prev_feats_output']
        
        print("⚠️  使用基于文件名的猜测IO信息:")
        print("   输入:")
        for i, name in enumerate(input_names):
            print(f"     {i}: {name}")
        
        print("   输出:")
        for i, name in enumerate(output_names):
            print(f"     {i}: {name}")
        
        return input_names, output_names

def prepare_input_data(model_path, input_names):
    """
    准备输入数据，直接从ONNX模型读取真实的输入维度和数据类型
    
    Args:
        model_path: ONNX模型路径
        input_names: 输入名称列表
    
    Returns:
        dict: 输入数据字典
    """
    input_data = {}
    
    try:
        # 加载ONNX模型来获取真实的输入信息
        model = onnx.load(model_path)
        
        # 创建输入名称到形状和数据类型的映射
        input_info_map = {}
        for input_node in model.graph.input:
            input_name = input_node.name
            
            # 跳过初始值（initializer）
            initializer_names = [init.name for init in model.graph.initializer]
            if input_name in initializer_names:
                continue
            
            # 获取输入的形状
            shape = []
            for dim in input_node.type.tensor_type.shape.dim:
                dim_value = dim.dim_value
                if dim_value == 0:  # 动态维度
                    # 对于动态维度，使用合理的默认值
                    dim_value = 1
                shape.append(dim_value)
            
            # 获取输入的数据类型
            elem_type = input_node.type.tensor_type.elem_type
            
            input_info_map[input_name] = {
                'shape': shape,
                'dtype': elem_type
            }
        
        print("📊 从ONNX模型读取的输入信息:")
        for name, info in input_info_map.items():
            dtype_name = get_onnx_type_name(info['dtype'])
            print(f"   {name}: 形状 {info['shape']}, 类型 {info['dtype']} ({dtype_name})")
        
        # 为每个输入名称创建数据
        for name in input_names:
            if name in input_info_map:
                # 使用模型中的真实形状和数据类型
                info = input_info_map[name]
                shape = info['shape']
                dtype = info['dtype']
                
                # 根据数据类型创建合适的数据
                data = create_data_by_type(shape, dtype)
                input_data[name] = data
                
                dtype_name = get_onnx_type_name(dtype)
                print(f"   准备输入 '{name}': 形状 {shape}, 类型 {dtype_name}")
            else:
                # 如果输入名称不在模型中，使用智能猜测
                shape = guess_input_shape_from_name(name)
                # 默认使用float32，但根据名称猜测类型
                guessed_dtype = guess_input_dtype_from_name(name)
                data = create_data_by_type(shape, guessed_dtype)
                input_data[name] = data
                
                dtype_name = get_onnx_type_name(guessed_dtype)
                print(f"   准备输入 '{name}': 猜测形状 {shape}, 类型 {dtype_name} (未在模型中找到)")
        
    except Exception as e:
        print(f"⚠️  无法从ONNX模型读取输入信息: {e}")
        print("   使用备用方法准备输入数据...")
        
        # 备用方法：使用智能猜测
        for name in input_names:
            shape = guess_input_shape_from_name(name)
            guessed_dtype = guess_input_dtype_from_name(name)
            data = create_data_by_type(shape, guessed_dtype)
            input_data[name] = data
            
            dtype_name = get_onnx_type_name(guessed_dtype)
            print(f"   准备输入 '{name}': 猜测形状 {shape}, 类型 {dtype_name}")
    
    return input_data

def get_onnx_type_name(elem_type):
    """获取ONNX数据类型的名称"""
    type_mapping = {
        1: 'float32',
        2: 'uint8',
        3: 'int8',
        4: 'uint16',
        5: 'int16',
        6: 'int32',
        7: 'int64',
        8: 'string',
        9: 'bool',
        10: 'float16',
        11: 'double',
        12: 'uint32',
        13: 'uint64',
        14: 'complex64',
        15: 'complex128',
        16: 'bfloat16'
    }
    return type_mapping.get(elem_type, f'unknown({elem_type})')

def create_data_by_type(shape, elem_type):
    """根据数据类型创建合适的数据"""
    if elem_type == 1:  # float32
        return np.random.randn(*shape).astype(np.float32)
    elif elem_type == 6:  # int32
        return np.random.randint(0, 100, shape).astype(np.int32)
    elif elem_type == 7:  # int64
        return np.random.randint(0, 100, shape).astype(np.int64)
    elif elem_type == 2:  # uint8
        return np.random.randint(0, 255, shape).astype(np.uint8)
    elif elem_type == 3:  # int8
        return np.random.randint(-128, 127, shape).astype(np.int8)
    elif elem_type == 10:  # float16
        return np.random.randn(*shape).astype(np.float16)
    elif elem_type == 11:  # double
        return np.random.randn(*shape).astype(np.float64)
    elif elem_type == 9:  # bool
        return np.random.choice([True, False], shape).astype(np.bool_)
    else:
        # 默认使用float32
        return np.random.randn(*shape).astype(np.float32)

def guess_input_dtype_from_name(input_name):
    """
    根据输入名称猜测合适的数据类型
    
    Args:
        input_name: 输入名称
    
    Returns:
        int: ONNX数据类型代码
    """
    input_name_lower = input_name.lower()
    
    # 根据关键词猜测数据类型
    if any(keyword in input_name_lower for keyword in ['coord', 'coor', 'point', 'voxel', 'index', 'id']):
        return 7  # int64 (坐标、索引通常用int64)
    
    elif any(keyword in input_name_lower for keyword in ['image', 'img', 'feature', 'feat', 'embedding']):
        return 1  # float32 (图像和特征通常用float32)
    
    elif any(keyword in input_name_lower for keyword in ['grid', 'transform', 'calib']):
        return 1  # float32 (变换和标定通常用float32)
    
    elif any(keyword in input_name_lower for keyword in ['meta', 'info', 'label']):
        return 6  # int32 (元数据和标签通常用int32)
    
    else:
        return 1  # 默认使用float32

def guess_input_shape_from_name(input_name):
    """
    根据输入名称智能猜测合适的形状（备用方法）
    
    Args:
        input_name: 输入名称
    
    Returns:
        list: 猜测的形状
    """
    input_name_lower = input_name.lower()
    
    # 根据关键词猜测形状
    if any(keyword in input_name_lower for keyword in ['feature', 'feat', 'embedding']):
        return [1, 64, 100, 100]  # 特征数据
    
    elif any(keyword in input_name_lower for keyword in ['coord', 'coor', 'point', 'voxel']):
        return [1, 4, 10000]  # 坐标或点云数据
    
    elif any(keyword in input_name_lower for keyword in ['image', 'img']):
        return [1, 3, 320, 640]  # 图像数据
    
    elif any(keyword in input_name_lower for keyword in ['grid', 'transform']):
        return [1, 6, 200, 200, 2]  # 网格或变换数据
    
    elif any(keyword in input_name_lower for keyword in ['calib', 'intrinsic', 'extrinsic']):
        return [1, 3, 4]  # 标定数据
    
    elif any(keyword in input_name_lower for keyword in ['meta', 'info']):
        return [1, 10]  # 元数据
    
    else:
        return [1, 64, 100, 100]  # 通用特征形状

def run_hbruntime_inference(session, input_data, output_names):
    """
    运行HBRuntime推理
    
    Args:
        session: HBRuntime会话
        input_data: 输入数据字典
        output_names: 输出名称列表
    
    Returns:
        list: 推理结果列表
    """
    try:
        print("🚀 开始HBRuntime推理...")
        
        # 根据正确的使用方式运行推理
        # HBRuntime.run()需要输出名称列表和输入数据字典
        outputs = session.run(output_names, input_data)
        
        print("✅ 推理完成!")
        
        # HBRuntime通常返回字典格式的结果
        if isinstance(outputs, dict):
            # 按输出名称顺序返回结果列表
            output_list = [outputs[name] for name in output_names]
            return output_list
        elif isinstance(outputs, list):
            return outputs
        else:
            return [outputs]
            
    except Exception as e:
        print(f"❌ HBRuntime推理失败: {e}")
        raise

def test_hbruntime_model(model_path):
    """
    测试HBRuntime模型
    
    Args:
        model_path: 模型路径
    """
    print(f"🧪 测试HBRuntime模型: {model_path}")
    
    try:
        # 先分析模型获取IO信息
        input_names, output_names = get_model_io_info(model_path)
        
        # 创建会话
        session = create_hbruntime_session(model_path)
        
        # 准备输入数据（需要模型路径来读取真实维度）
        input_data = prepare_input_data(model_path, input_names)
        
        # 验证输入数据是否包含所有必需的输入
        missing_inputs = []
        for name in input_names:
            if name not in input_data:
                missing_inputs.append(name)
        
        if missing_inputs:
            print(f"⚠️  缺少以下输入: {missing_inputs}")
            # 尝试补充缺失的输入
            for name in missing_inputs:
                if name in input_data:
                    continue
                # 使用默认形状
                shape = [1, 3, 224, 224]
                input_data[name] = np.random.randn(*shape).astype(np.float32)
                print(f"   补充输入 '{name}': 形状 {shape}")
        
        # 运行推理
        outputs = run_hbruntime_inference(session, input_data, output_names)
        
        # 显示结果
        print("📊 推理结果:")
        for i, output in enumerate(outputs):
            print(f"   输出{i}: 形状 {output.shape}, 类型 {output.dtype}")
            print(f"       值范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"       均值: {output.mean():.4f}, 标准差: {output.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        
        # 提供更详细的错误信息
        if "img_front_fisheye" in str(e):
            print("💡 提示: 这是一个鱼眼模型，需要鱼眼相关的输入")
            print("   尝试使用鱼眼模型配置重新测试...")
            
            # 尝试使用鱼眼配置重新测试
            try:
                input_names = ['img_front_fisheye', 'img_right_fisheye', 'img_rear_fisheye', 
                              'img_left_fisheye', 'fish_images_grid', 'fish_vt_grid',
                              'fish_prev_feats', 'fish_prev_feats_grid']
                output_names = ['fish_head_conv', 'fish_hm_center', 'fish_prev_feats_output']
                
                session = create_hbruntime_session(model_path)
                input_data = prepare_input_data(model_path, input_names)
                outputs = run_hbruntime_inference(session, input_data, output_names)
                
                print("✅ 使用鱼眼配置测试成功!")
                return True
                
            except Exception as e2:
                print(f"❌ 鱼眼配置测试也失败: {e2}")
        
        return False

def compare_models_outputs(fisheye_path, pinhole_path, merged_path, tolerance=1e-5):
    """
    对比合并前后模型的输出是否一致
    
    Args:
        fisheye_path: 鱼眼模型路径
        pinhole_path: 针孔模型路径
        merged_path: 合并模型路径
        tolerance: 数值容忍度
    
    Returns:
        bool: 输出是否一致
    """
    print("=" * 70)
    print("🔍 对比合并前后模型输出")
    print("=" * 70)
    
    try:
        # 1. 准备相同的输入数据
        print("\n📊 准备相同的输入数据...")
        
        # 获取合并模型的输入信息
        merged_input_names, merged_output_names = get_model_io_info(merged_path)
        
        # 准备输入数据（使用合并模型的输入）
        input_data = prepare_input_data(merged_path, merged_input_names)
        
        # 2. 分别运行三个模型的推理
        print("\n🚀 运行鱼眼模型推理...")
        fisheye_session = create_hbruntime_session(fisheye_path)
        fisheye_input_names, fisheye_output_names = get_model_io_info(fisheye_path)
        
        # 准备鱼眼模型的输入（只包含鱼眼相关的输入）
        fisheye_input_data = {}
        for name in fisheye_input_names:
            if name in input_data:
                fisheye_input_data[name] = input_data[name]
            else:
                print(f"⚠️  鱼眼模型缺少输入: {name}")
        
        fisheye_outputs = run_hbruntime_inference(fisheye_session, fisheye_input_data, fisheye_output_names)
        
        print("\n🚀 运行针孔模型推理...")
        pinhole_session = create_hbruntime_session(pinhole_path)
        pinhole_input_names, pinhole_output_names = get_model_io_info(pinhole_path)
        
        # 准备针孔模型的输入（只包含针孔相关的输入）
        pinhole_input_data = {}
        for name in pinhole_input_names:
            if name in input_data:
                pinhole_input_data[name] = input_data[name]
            else:
                print(f"⚠️  针孔模型缺少输入: {name}")
        
        pinhole_outputs = run_hbruntime_inference(pinhole_session, pinhole_input_data, pinhole_output_names)
        
        print("\n🚀 运行合并模型推理...")
        merged_session = create_hbruntime_session(merged_path)
        merged_outputs = run_hbruntime_inference(merged_session, input_data, merged_output_names)
        
        # 3. 对比输出结果
        print("\n📊 对比输出结果...")
        
        all_match = True
        
        # 对比鱼眼模型输出
        print("🔍 对比鱼眼模型输出:")
        for i, (orig_name, orig_output) in enumerate(zip(fisheye_output_names, fisheye_outputs)):
            # 在合并模型中查找对应的输出
            merged_output = None
            for j, merged_name in enumerate(merged_output_names):
                if orig_name in merged_name or merged_name in orig_name:
                    merged_output = merged_outputs[j]
                    break
            
            if merged_output is not None:
                if compare_tensors(orig_output, merged_output, tolerance, f"鱼眼输出{i}"):
                    print(f"   ✅ 鱼眼输出 {orig_name} 匹配")
                else:
                    print(f"   ❌ 鱼眼输出 {orig_name} 不匹配")
                    all_match = False
            else:
                print(f"   ⚠️  在合并模型中找不到对应的鱼眼输出: {orig_name}")
        
        # 对比针孔模型输出
        print("\n🔍 对比针孔模型输出:")
        for i, (orig_name, orig_output) in enumerate(zip(pinhole_output_names, pinhole_outputs)):
            # 在合并模型中查找对应的输出
            merged_output = None
            for j, merged_name in enumerate(merged_output_names):
                if orig_name == merged_name:
                    merged_output = merged_outputs[j]
                    break
            
            if merged_output is not None:
                if compare_tensors(orig_output, merged_output, tolerance, f"针孔输出{i}"):
                    print(f"   ✅ 针孔输出 {orig_name} 匹配")
                else:
                    print(f"   ❌ 针孔输出 {orig_name} 不匹配")
                    all_match = False
            else:
                print(f"   ⚠️  在合并模型中找不到对应的针孔输出: {orig_name}")
        
        # 4. 总结结果
        print("\n" + "=" * 70)
        if all_match:
            print("🎉 所有模型输出匹配！合并模型功能正确。")
        else:
            print("⚠️  部分模型输出不匹配，请检查合并模型的功能完整性。")
        print("=" * 70)
        
        return all_match
        
    except Exception as e:
        print(f"❌ 对比模型输出失败: {e}")
        return False

def compare_tensors(tensor1, tensor2, tolerance=1e-5, name=""):
    """
    对比两个张量是否一致
    
    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        tolerance: 数值容忍度
        name: 张量名称（用于调试）
    
    Returns:
        bool: 是否一致
    """
    # 检查形状是否一致
    if tensor1.shape != tensor2.shape:
        print(f"   {name}: 形状不一致 {tensor1.shape} vs {tensor2.shape}")
        
        # 分析形状差异
        analyze_shape_difference(tensor1.shape, tensor2.shape, name)
        return False
    
    # 检查数据类型是否一致
    if tensor1.dtype != tensor2.dtype:
        print(f"   {name}: 数据类型不一致 {tensor1.dtype} vs {tensor2.dtype}")
        return False
    
    # 检查数值是否一致（在容忍度范围内）
    if np.allclose(tensor1, tensor2, rtol=tolerance, atol=tolerance):
        return True
    else:
        # 计算差异统计
        diff = np.abs(tensor1 - tensor2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   {name}: 数值不匹配 (最大差异: {max_diff:.6e}, 平均差异: {mean_diff:.6e})")
        return False

def analyze_shape_difference(shape1, shape2, tensor_name):
    """分析形状差异并提供诊断信息"""
    if len(shape1) != len(shape2):
        print(f"     维度数量不同: {len(shape1)} vs {len(shape2)}")
        return
    
    print(f"     形状差异分析:")
    for i, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != dim2:
            diff = dim2 - dim1
            print(f"       维度{i}: {dim1} -> {dim2} (差异: {diff})")
            
            # 提供可能的解释
            if i == 1:  # 通道维度
                if "head_conv" in tensor_name:
                    print(f"         可能是类别数量配置不同")
            elif i == 2:  # 高度维度
                if diff == -120:  # 240 -> 120
                    print(f"         可能是BEV网格分辨率配置不同 (240x240 vs 120x120)")
            elif i == 3:  # 宽度维度
                if diff == -120:  # 240 -> 120
                    print(f"         可能是BEV网格分辨率配置不同 (240x240 vs 120x120)")

def diagnose_model_differences(fisheye_path, pinhole_path, merged_path):
    """
    诊断模型之间的差异
    
    Args:
        fisheye_path: 鱼眼模型路径
        pinhole_path: 针孔模型路径
        merged_path: 合并模型路径
    """
    print("=" * 70)
    print("🔍 模型差异诊断")
    print("=" * 70)
    
    try:
        # 加载三个模型
        fisheye_model = onnx.load(fisheye_path)
        pinhole_model = onnx.load(pinhole_path)
        merged_model = onnx.load(merged_path)
        
        print("\n📊 模型基本信息:")
        print(f"   鱼眼模型: {len(fisheye_model.graph.node)} 个节点")
        print(f"   针孔模型: {len(pinhole_model.graph.node)} 个节点")
        print(f"   合并模型: {len(merged_model.graph.node)} 个节点")
        
        # 分析输入差异
        print("\n📊 输入信息对比:")
        analyze_inputs_differences(fisheye_model, pinhole_model, merged_model)
        
        # 分析输出差异
        print("\n📊 输出信息对比:")
        analyze_outputs_differences(fisheye_model, pinhole_model, merged_model)
        
        # 分析网络结构差异
        print("\n📊 网络结构差异:")
        analyze_network_structure_differences(fisheye_model, pinhole_model, merged_model)
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")

def analyze_inputs_differences(fisheye_model, pinhole_model, merged_model):
    """分析输入差异"""
    fisheye_inputs = [input.name for input in fisheye_model.graph.input]
    pinhole_inputs = [input.name for input in pinhole_model.graph.input]
    merged_inputs = [input.name for input in merged_model.graph.input]
    
    print("   鱼眼模型输入:", fisheye_inputs)
    print("   针孔模型输入:", pinhole_inputs)
    print("   合并模型输入:", merged_inputs)
    
    # 检查输入形状
    print("\n   输入形状对比:")
    for input_name in set(fisheye_inputs + pinhole_inputs):
        shapes = {}
        for model, name in [(fisheye_model, "鱼眼"), (pinhole_model, "针孔"), (merged_model, "合并")]:
            for input_node in model.graph.input:
                if input_node.name == input_name:
                    shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
                    shapes[name] = shape
                    break
        
        if len(set(str(s) for s in shapes.values())) > 1:
            print(f"     {input_name}: {shapes}")

def analyze_outputs_differences(fisheye_model, pinhole_model, merged_model):
    """分析输出差异"""
    fisheye_outputs = [output.name for output in fisheye_model.graph.output]
    pinhole_outputs = [output.name for output in pinhole_model.graph.output]
    merged_outputs = [output.name for output in merged_model.graph.output]
    
    print("   鱼眼模型输出:", fisheye_outputs)
    print("   针孔模型输出:", pinhole_outputs)
    print("   合并模型输出:", merged_outputs)
    
    # 检查输出形状
    print("\n   输出形状对比:")
    shape_found = False
    for output_name in set(pinhole_outputs):  # 重点关注针孔模型输出
        shapes = {}
        for model, name in [(pinhole_model, "针孔"), (merged_model, "合并")]:
            for output_node in model.graph.output:
                if output_node.name == output_name:
                    shape = []
                    for dim in output_node.type.tensor_type.shape.dim:
                        dim_value = dim.dim_value
                        if dim_value == 0:  # 动态维度
                            dim_value = -1  # 标记为动态
                        shape.append(dim_value)
                    shapes[name] = shape
                    break
        
        if shapes.get("针孔") != shapes.get("合并"):
            print(f"     {output_name}: {shapes}")
            shape_found = True
        else:
            # 即使形状相同也显示，但标记为相同
            if shapes.get("针孔") and shapes.get("针孔") != [-1, -1, -1, -1]:  # 排除全动态情况
                print(f"     {output_name}: {shapes} (ONNX定义相同)")
                shape_found = True
    
    if not shape_found:
        print("     (ONNX模型中输出形状信息缺失或全为动态维度)")

def get_actual_output_shapes(model_path):
    """通过实际推理获取真实的输出形状"""
    print(f"\n🔍 通过实际推理获取真实输出形状: {Path(model_path).name}")
    
    try:
        # 创建会话
        session = create_hbruntime_session(model_path)
        
        # 获取输入输出信息
        input_names, output_names = get_model_io_info(model_path)
        
        # 准备输入数据
        input_data = prepare_input_data(model_path, input_names)
        
        # 运行推理
        outputs = run_hbruntime_inference(session, input_data, output_names)
        
        # 显示实际形状
        actual_shapes = {}
        for i, (name, output) in enumerate(zip(output_names, outputs)):
            print(f"     {name}: {output.shape}")
            actual_shapes[name] = output.shape
        
        return actual_shapes
        
    except Exception as e:
        print(f"❌ 无法获取实际输出形状: {e}")
        return {}

def deep_diagnose_model_shapes(fisheye_path, pinhole_path, merged_path):
    """深度诊断模型输出形状差异"""
    print("=" * 70)
    print("🔍 深度诊断模型输出形状差异")
    print("=" * 70)
    
    # 获取实际推理的输出形状
    fisheye_actual = get_actual_output_shapes(fisheye_path)
    pinhole_actual = get_actual_output_shapes(pinhole_path)
    merged_actual = get_actual_output_shapes(merged_path)
    
    # 对比针孔模型和合并模型的输出形状
    print("\n📊 实际输出形状对比 (针孔 vs 合并):")
    
    all_match = True
    for output_name in pinhole_actual.keys():
        pinhole_shape = pinhole_actual.get(output_name)
        merged_shape = merged_actual.get(output_name)
        
        if pinhole_shape is not None and merged_shape is not None:
            if pinhole_shape == merged_shape:
                print(f"   ✅ {output_name}: {pinhole_shape} (匹配)")
            else:
                print(f"   ❌ {output_name}: 针孔{pinhole_shape} vs 合并{merged_shape} (不匹配)")
                all_match = False
                
                # 分析具体差异
                analyze_shape_difference(pinhole_shape, merged_shape, output_name)
        else:
            print(f"   ⚠️  {output_name}: 无法获取形状信息")
    
    # 分析可能的原因
    print("\n🔧 形状不匹配的可能原因分析:")
    
    # 检查BEV网格配置
    if any(shape and len(shape) >= 4 for shape in pinhole_actual.values()):
        # 分析空间维度差异
        for name, shape in pinhole_actual.items():
            if len(shape) >= 4:  # 有空间维度的输出
                h, w = shape[2], shape[3]
                merged_shape = merged_actual.get(name, [])
                if len(merged_shape) >= 4:
                    merged_h, merged_w = merged_shape[2], merged_shape[3]
                    if h != merged_h or w != merged_w:
                        print(f"   - {name}: BEV网格分辨率不同 ({h}x{w} vs {merged_h}x{merged_w})")
    
    # 检查类别数量差异
    for name, shape in pinhole_actual.items():
        if len(shape) >= 2:  # 有通道维度的输出
            channels = shape[1]
            merged_shape = merged_actual.get(name, [])
            if len(merged_shape) >= 2:
                merged_channels = merged_shape[1]
                if channels != merged_channels:
                    print(f"   - {name}: 类别/通道数不同 ({channels} vs {merged_channels})")
    
    # 检查特征图下采样比例
    print("\n💡 建议解决方案:")
    if not all_match:
        print("   1. 检查两个模型的训练配置文件，确保BEV网格分辨率一致")
        print("   2. 检查类别数量配置是否相同")
        print("   3. 检查特征提取网络的下采样比例是否一致")
        print("   4. 考虑重新训练模型，使用相同的配置参数")
    else:
        print("   🎉 所有输出形状匹配，合并成功!")
    
    return all_match

def analyze_network_structure_differences(fisheye_model, pinhole_model, merged_model):
    """分析网络结构差异"""
    # 统计不同类型的节点
    def count_node_types(model):
        types = {}
        for node in model.graph.node:
            op_type = node.op_type
            types[op_type] = types.get(op_type, 0) + 1
        return types
    
    fisheye_types = count_node_types(fisheye_model)
    pinhole_types = count_node_types(pinhole_model)
    merged_types = count_node_types(merged_model)
    
    print("   节点类型统计:")
    print(f"     鱼眼模型: {fisheye_types}")
    print(f"     针孔模型: {pinhole_types}")
    print(f"     合并模型: {merged_types}")
    
    # 检查horizon自定义算子
    horizon_ops = [op for op in fisheye_types.keys() if 'horizon' in op.lower() or 'Hz' in op]
    horizon_ops.extend([op for op in pinhole_types.keys() if 'horizon' in op.lower() or 'Hz' in op])
    
    if horizon_ops:
        print(f"\n   Horizon自定义算子: {set(horizon_ops)}")

def compare_models_outputs(fisheye_path, pinhole_path, merged_path, tolerance=1e-5):
    """
    对比合并前后模型的输出是否一致
    
    Args:
        fisheye_path: 鱼眼模型路径
        pinhole_path: 针孔模型路径
        merged_path: 合并模型路径
        tolerance: 数值容忍度
    
    Returns:
        bool: 输出是否一致
    """
    print("=" * 70)
    print("🔍 对比合并前后模型输出")
    print("=" * 70)
    
    try:
        # 1. 准备相同的输入数据
        print("\n📊 准备相同的输入数据...")
        
        # 获取合并模型的输入信息
        merged_input_names, merged_output_names = get_model_io_info(merged_path)
        
        # 准备输入数据（使用合并模型的输入）
        input_data = prepare_input_data(merged_path, merged_input_names)
        
        # 2. 分别运行三个模型的推理
        print("\n🚀 运行鱼眼模型推理...")
        fisheye_session = create_hbruntime_session(fisheye_path)
        fisheye_input_names, fisheye_output_names = get_model_io_info(fisheye_path)
        
        # 准备鱼眼模型的输入（只包含鱼眼相关的输入）
        fisheye_input_data = {}
        for name in fisheye_input_names:
            if name in input_data:
                fisheye_input_data[name] = input_data[name]
            else:
                print(f"⚠️  鱼眼模型缺少输入: {name}")
        
        fisheye_outputs = run_hbruntime_inference(fisheye_session, fisheye_input_data, fisheye_output_names)
        
        print("\n🚀 运行针孔模型推理...")
        pinhole_session = create_hbruntime_session(pinhole_path)
        pinhole_input_names, pinhole_output_names = get_model_io_info(pinhole_path)
        
        # 准备针孔模型的输入（只包含针孔相关的输入）
        pinhole_input_data = {}
        for name in pinhole_input_names:
            if name in input_data:
                pinhole_input_data[name] = input_data[name]
            else:
                print(f"⚠️  针孔模型缺少输入: {name}")
        
        pinhole_outputs = run_hbruntime_inference(pinhole_session, pinhole_input_data, pinhole_output_names)
        
        print("\n🚀 运行合并模型推理...")
        merged_session = create_hbruntime_session(merged_path)
        merged_outputs = run_hbruntime_inference(merged_session, input_data, merged_output_names)
        
        # 3. 对比输出结果
        print("\n📊 对比输出结果...")
        
        all_match = True
        
        # 对比鱼眼模型输出
        print("🔍 对比鱼眼模型输出:")
        for i, (orig_name, orig_output) in enumerate(zip(fisheye_output_names, fisheye_outputs)):
            # 在合并模型中查找对应的输出
            merged_output = None
            for j, merged_name in enumerate(merged_output_names):
                if orig_name in merged_name or merged_name in orig_name:
                    merged_output = merged_outputs[j]
                    break
            
            if merged_output is not None:
                if compare_tensors(orig_output, merged_output, tolerance, f"鱼眼输出{i}"):
                    print(f"   ✅ 鱼眼输出 {orig_name} 匹配")
                else:
                    print(f"   ❌ 鱼眼输出 {orig_name} 不匹配")
                    all_match = False
            else:
                print(f"   ⚠️  在合并模型中找不到对应的鱼眼输出: {orig_name}")
        
        # 对比针孔模型输出
        print("\n🔍 对比针孔模型输出:")
        for i, (orig_name, orig_output) in enumerate(zip(pinhole_output_names, pinhole_outputs)):
            # 在合并模型中查找对应的输出
            merged_output = None
            for j, merged_name in enumerate(merged_output_names):
                if orig_name == merged_name:
                    merged_output = merged_outputs[j]
                    break
            
            if merged_output is not None:
                if compare_tensors(orig_output, merged_output, tolerance, f"针孔输出{i}"):
                    print(f"   ✅ 针孔输出 {orig_name} 匹配")
                else:
                    print(f"   ❌ 针孔输出 {orig_name} 不匹配")
                    all_match = False
            else:
                print(f"   ⚠️  在合并模型中找不到对应的针孔输出: {orig_name}")
        
        # 4. 总结结果
        print("\n" + "=" * 70)
        if all_match:
            print("🎉 所有模型输出匹配！合并模型功能正确。")
        else:
            print("⚠️  部分模型输出不匹配，请检查合并模型的功能完整性。")
        print("=" * 70)
        
        return all_match
        
    except Exception as e:
        print(f"❌ 对比模型输出失败: {e}")
        return False

def compare_tensors(tensor1, tensor2, tolerance=1e-5, name=""):
    """
    对比两个张量是否一致
    
    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        tolerance: 数值容忍度
        name: 张量名称（用于调试）
    
    Returns:
        bool: 是否一致
    """
    # 检查形状是否一致
    if tensor1.shape != tensor2.shape:
        print(f"   {name}: 形状不一致 {tensor1.shape} vs {tensor2.shape}")
        return False
    
    # 检查数据类型是否一致
    if tensor1.dtype != tensor2.dtype:
        print(f"   {name}: 数据类型不一致 {tensor1.dtype} vs {tensor2.dtype}")
        return False
    
    # 检查数值是否一致（在容忍度范围内）
    if np.allclose(tensor1, tensor2, rtol=tolerance, atol=tolerance):
        return True
    else:
        # 计算差异统计
        diff = np.abs(tensor1 - tensor2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   {name}: 数值不匹配 (最大差异: {max_diff:.6e}, 平均差异: {mean_diff:.6e})")
        return False

def merge_models_with_hbruntime(fisheye_path, pinhole_path, output_path):
    """
    使用HBRuntime友好的方式合并模型
    
    Args:
        fisheye_path: 鱼眼模型路径
        pinhole_path: 针孔模型路径
        output_path: 输出模型路径
    """
    print("=" * 60)
    print("🚀 使用HBRuntime友好的方式合并模型")
    print("=" * 60)
    
    try:
        # 加载两个模型
        fisheye_model = onnx.load(fisheye_path)
        pinhole_model = onnx.load(pinhole_path)
        
        print("📊 模型信息:")
        print(f"   鱼眼模型: {len(fisheye_model.graph.node)} 个节点")
        print(f"   针孔模型: {len(pinhole_model.graph.node)} 个节点")
        
        # 检查horizon自定义算子
        horizon_ops_fisheye = [node for node in fisheye_model.graph.node 
                              if node.domain == 'horizon']
        horizon_ops_pinhole = [node for node in pinhole_model.graph.node 
                              if node.domain == 'horizon']
        
        print(f"🔍 Horizon自定义算子:")
        print(f"   鱼眼模型: {len(horizon_ops_fisheye)} 个")
        print(f"   针孔模型: {len(horizon_ops_pinhole)} 个")
        
        # 创建合并的计算图
        merged_graph = onnx.GraphProto()
        merged_graph.name = "merged_hbruntime_compatible"
        
        # 合并所有组件
        merged_graph.node.extend(fisheye_model.graph.node)
        merged_graph.node.extend(pinhole_model.graph.node)
        
        merged_graph.input.extend(fisheye_model.graph.input)
        merged_graph.input.extend(pinhole_model.graph.input)
        
        merged_graph.output.extend(fisheye_model.graph.output)
        merged_graph.output.extend(pinhole_model.graph.output)
        
        merged_graph.initializer.extend(fisheye_model.graph.initializer)
        merged_graph.initializer.extend(pinhole_model.graph.initializer)
        
        merged_graph.value_info.extend(fisheye_model.graph.value_info)
        merged_graph.value_info.extend(pinhole_model.graph.value_info)
        
        # 合并opset导入，确保包含horizon域
        merged_opset_imports = []
        opset_domains = set()
        
        for opset in fisheye_model.opset_import:
            domain = opset.domain if opset.domain else ""
            if domain not in opset_domains:
                merged_opset_imports.append(opset)
                opset_domains.add(domain)
        
        for opset in pinhole_model.opset_import:
            domain = opset.domain if opset.domain else ""
            if domain not in opset_domains:
                merged_opset_imports.append(opset)
                opset_domains.add(domain)
        
        # 确保包含horizon域
        if 'horizon' not in opset_domains:
            horizon_opset = onnx.helper.make_opsetid('horizon', 1)
            merged_opset_imports.append(horizon_opset)
            print("✅ 已添加horizon域opset导入")
        
        # 创建合并模型
        merged_model = onnx.helper.make_model(
            merged_graph,
            producer_name='hbruntime-merge-tool',
            opset_imports=merged_opset_imports
        )
        
        # 保存合并模型
        onnx.save(merged_model, output_path)
        print(f"💾 合并模型已保存至: {output_path}")
        
        # 使用HBRuntime测试合并模型
        print("\n🧪 使用HBRuntime测试合并模型...")
        if test_hbruntime_model(output_path):
            print("🎉 合并模型HBRuntime测试通过!")
        else:
            print("⚠️  合并模型HBRuntime测试失败，但模型已保存")
        
        return True
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False

def compare_runtimes(model_path):
    """
    比较ONNX Runtime和HBRuntime的性能
    
    Args:
        model_path: 模型路径
    """
    print("📊 运行时性能比较")
    print("=" * 40)
    
    # 测试HBRuntime
    print("\n🧪 测试HBRuntime...")
    try:
        import time
        
        # HBRuntime测试
        hb_start = time.time()
        session_hb = create_hbruntime_session(model_path)
        input_info, _ = get_model_io_info(session_hb)
        input_data = prepare_input_data(input_info)
        
        # 预热
        _ = run_hbruntime_inference(session_hb, input_data)
        
        # 正式测试
        hb_times = []
        for i in range(10):
            start = time.time()
            _ = run_hbruntime_inference(session_hb, input_data)
            hb_times.append(time.time() - start)
        
        hb_avg = np.mean(hb_times) * 1000  # 转换为毫秒
        print(f"✅ HBRuntime平均推理时间: {hb_avg:.2f}ms")
        
    except Exception as e:
        print(f"❌ HBRuntime测试失败: {e}")
        hb_avg = float('inf')
    
    # ONNX Runtime测试（如果可能）
    print("\n🧪 测试ONNX Runtime...")
    try:
        import onnxruntime as ort
        import time
        
        session_ort = ort.InferenceSession(model_path)
        
        # 准备输入
        ort_inputs = {}
        for input in session_ort.get_inputs():
            shape = [1 if dim == -1 else dim for dim in input.shape]
            ort_inputs[input.name] = np.random.randn(*shape).astype(np.float32)
        
        # 预热
        _ = session_ort.run(None, ort_inputs)
        
        # 正式测试
        ort_times = []
        for i in range(10):
            start = time.time()
            _ = session_ort.run(None, ort_inputs)
            ort_times.append(time.time() - start)
        
        ort_avg = np.mean(ort_times) * 1000  # 转换为毫秒
        print(f"✅ ONNX Runtime平均推理时间: {ort_avg:.2f}ms")
        
    except Exception as e:
        print(f"❌ ONNX Runtime测试失败: {e}")
        ort_avg = float('inf')
    
    print("\n📈 性能比较结果:")
    if hb_avg < ort_avg:
        print(f"🏆 HBRuntime更快: {hb_avg:.2f}ms vs {ort_avg:.2f}ms")
    else:
        print(f"🏆 ONNX Runtime更快: {ort_avg:.2f}ms vs {hb_avg:.2f}ms")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HBRuntime推理工具')
    parser.add_argument('--test', type=str, help='测试HBRuntime模型')
    parser.add_argument('--merge', nargs=3, metavar=('FISHEYE', 'PINHOLE', 'OUTPUT'), 
                       help='使用HBRuntime友好的方式合并模型')
    parser.add_argument('--compare', type=str, help='比较ONNX Runtime和HBRuntime性能')
    parser.add_argument('--compare-models', nargs=3, metavar=('FISHEYE', 'PINHOLE', 'MERGED'), 
                       help='对比合并前后模型的输出是否一致')
    parser.add_argument('--diagnose', nargs=3, metavar=('FISHEYE', 'PINHOLE', 'MERGED'), 
                       help='诊断模型之间的差异')
    parser.add_argument('--deep-diagnose', nargs=3, metavar=('FISHEYE', 'PINHOLE', 'MERGED'), 
                       help='深度诊断模型输出形状差异')
    parser.add_argument('--compare-outputs', nargs=3, metavar=('FISHEYE', 'PINHOLE', 'MERGED'), 
                       help='对比合并前后模型的输出是否一致')
    
    args = parser.parse_args()
    
    if args.test:
        test_hbruntime_model(args.test)
    elif args.merge:
        fisheye_path, pinhole_path, output_path = args.merge
        merge_models_with_hbruntime(fisheye_path, pinhole_path, output_path)
    elif args.compare:
        compare_runtimes(args.compare)
    elif args.compare_models:
        fisheye_path, pinhole_path, merged_path = args.compare_models
        compare_models_outputs(fisheye_path, pinhole_path, merged_path)
    elif args.diagnose:
        fisheye_path, pinhole_path, merged_path = args.diagnose
        diagnose_model_differences(fisheye_path, pinhole_path, merged_path)
    elif args.deep_diagnose:
        fisheye_path, pinhole_path, merged_path = args.deep_diagnose
        deep_diagnose_model_shapes(fisheye_path, pinhole_path, merged_path)
    elif args.compare_outputs:
        fisheye_path, pinhole_path, merged_path = args.compare_outputs
        compare_models_outputs(fisheye_path, pinhole_path, merged_path)
    else:
        print("请指定 --test、--merge 或 --compare 参数")
        print("示例:")
        print("  测试模型: python hbruntime_inference.py --test model.onnx")
        print("  合并模型: python hbruntime_inference.py --merge fisheye.onnx pinhole.onnx merged.onnx")
        print("  性能比较: python hbruntime_inference.py --compare model.onnx")
        print("  对比输出: python hbruntime_inference.py --compare-models fisheye.onnx pinhole.onnx merged.onnx")
        print("  诊断差异: python hbruntime_inference.py --diagnose fisheye.onnx pinhole.onnx merged.onnx")
        print("  深度诊断: python hbruntime_inference.py --deep-diagnose fisheye.onnx pinhole.onnx merged.onnx")
        print("  对比输出: python hbruntime_inference.py --compare-outputs fisheye.onnx pinhole.onnx merged.onnx")