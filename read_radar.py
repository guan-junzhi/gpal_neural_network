import os
import numpy as np

def read_lidar_point_cloud_from_hesai_pcd(lidar_path):
    """支持ASCII格式的PCD文件读取函数"""
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"文件不存在: {lidar_path}")

    try:
        with open(lidar_path, 'r') as f:
            # 读取并解析头部
            header = {}
            while True:
                line = f.readline().strip()
                
                if line.startswith('DATA'):
                    header['data_type'] = line.split()[1]
                    # 记录数据开始位置
                    data_start = f.tell()
                    break
                
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        value = parts[1:] if len(parts) > 2 else parts[1]
                        header[key] = value
            
            # 解析字段信息
            fields = header['FIELDS'].split() if isinstance(header['FIELDS'], str) else header['FIELDS']
            sizes = [int(s) for s in header['SIZE'].split()] if isinstance(header['SIZE'], str) else [int(s) for s in header['SIZE']]
            types = header['TYPE'].split() if isinstance(header['TYPE'], str) else header['TYPE']
            points_count = int(header['POINTS'][0] if isinstance(header['POINTS'], list) else header['POINTS'])
            
            # 检查数据格式
            if header['data_type'] != 'ascii':
                raise ValueError(f"不支持的数据格式: {header['data_type']}，仅支持ascii格式")
            
            # 读取ASCII数据
            f.seek(data_start)
            lines = f.readlines()
            
            # 解析ASCII数据
            points_data = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 分割数据行
                    values = line.split()
                    if len(values) == len(fields):
                        # 转换为对应的数据类型
                        point = []
                        for i, (field, size, type_char) in enumerate(zip(fields, sizes, types)):
                            try:
                                if type_char == 'F':  # Float类型
                                    point.append(float(values[i]))
                                elif type_char == 'U':  # Unsigned整数类型
                                    point.append(int(values[i]))
                                elif type_char == 'I':  # Signed整数类型
                                    point.append(int(values[i]))
                                else:
                                    # 默认按浮点数处理
                                    point.append(float(values[i]))
                            except (ValueError, IndexError):
                                point.append(0.0)  # 转换失败时使用默认值
                        points_data.append(point)
            
            # 转换为numpy数组
            if points_data:
                points_array = np.array(points_data, dtype=np.float32)
                
                # 过滤包含NaN的点
                valid_mask = ~np.isnan(points_array).any(axis=1)
                points_array = points_array[valid_mask]
                
                print(f"成功读取 {len(points_array)} 个点云数据点")
                print(f"字段: {fields}")
                print(f"数据形状: {points_array.shape}")
                
                return points_array
            else:
                print("警告: 未读取到有效数据")
                return np.array([])
            
    except Exception as e:
        print(f"读取点云文件 {lidar_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])

def read_lidar_point_cloud_from_hesai_pcd_binary(lidar_path):
    """二进制格式的PCD文件读取函数（保留原功能）"""
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"文件不存在: {lidar_path}")

    try:
        with open(lidar_path, 'rb') as f:
            # 读取并解析头部
            header = {}
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith('DATA'):
                    header['data_type'] = line.split()[1]
                    # 记录二进制数据开始位置
                    data_start = f.tell()
                    break
                
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        value = parts[1:] if len(parts) > 2 else parts[1]
                        header[key] = value
            
            # 解析字段信息
            fields = header['FIELDS'].split() if isinstance(header['FIELDS'], str) else header['FIELDS']
            sizes = [int(s) for s in header['SIZE'].split()] if isinstance(header['SIZE'], str) else [int(s) for s in header['SIZE']]
            types = header['TYPE'].split() if isinstance(header['TYPE'], str) else header['TYPE']
            points_count = int(header['POINTS'][0] if isinstance(header['POINTS'], list) else header['POINTS'])
            
            # 检查数据格式
            if header['data_type'] != 'binary':
                raise ValueError(f"不支持的数据格式: {header['data_type']}，仅支持binary格式")
            
            # 计算每个点的字节数
            point_step = sum(sizes)
            
            # 读取所有二进制数据
            f.seek(data_start)
            binary_data = f.read()
            
            # 检查数据长度
            expected_size = points_count * point_step
            if len(binary_data) < expected_size:
                print(f"警告: 数据长度不足，期望 {expected_size} 字节，实际 {len(binary_data)} 字节")
                points_count = min(points_count, len(binary_data) // point_step)
            
            # 优化：使用结构化数组直接映射二进制数据
            # 构建数据类型描述
            dtype_list = []
            type_mapping = {
                'F': {4: '<f4', 8: '<f8'},  # Float
                'U': {1: '<u1', 2: '<u2', 4: '<u4'},  # Unsigned
                'I': {1: '<i1', 2: '<i2', 4: '<i4'},  # Signed
            }
            
            for i, field in enumerate(fields):
                size = sizes[i]
                type_char = types[i]
                if type_char in type_mapping and size in type_mapping[type_char]:
                    dtype_list.append((field, type_mapping[type_char][size]))
                else:
                    # 默认使用对应大小的无符号整数
                    dtype_list.append((field, f'<u{size}'))
            
            # 创建结构化数据类型
            dtype = np.dtype(dtype_list)
            
            # 批量读取所有点数据
            points_structured = np.frombuffer(binary_data[:points_count * point_step], dtype=dtype, count=points_count)
            
            # 转换为常规numpy数组（只保留需要的字段）
            available_fields = fields
            
            # 创建结果数组
            points_array = np.zeros((len(points_structured), len(fields)), dtype=np.float32)
            
            for i, field in enumerate(available_fields):
                points_array[:, i] = points_structured[field].astype(np.float32)
            
            # 过滤包含NaN的点
            valid_mask = ~np.isnan(points_array).any(axis=1)
            points_array = points_array[valid_mask]
            return points_array
            
    except Exception as e:
        print(f"读取点云文件 {lidar_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])

def read_pcd_file(lidar_path):
    """自动检测PCD文件格式并读取"""
    try:
        # 先尝试ASCII格式
        return read_lidar_point_cloud_from_hesai_pcd(lidar_path)
    except Exception as e:
        print(f"ASCII格式读取失败，尝试二进制格式: {e}")
        # 再尝试二进制格式
        return read_lidar_point_cloud_from_hesai_pcd_binary(lidar_path)

if __name__ == '__main__':
    lidar_path = '1754483208.500329.pcd'
    points = read_pcd_file(lidar_path)
    print(f"最终读取的点云数据形状: {points.shape}")
    if points.size > 0:
        print("前5个点云数据:")
        print(points[:5])