import os
from typing import Dict, Any
import numpy as np
import re
def parse_file(file_path: str) -> Dict[str, Any]:
        """解析单个pbtxt文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析文件内容
            parsed_data = _parse_content(content)
            parsed_data['file_name'] = os.path.basename(file_path)
            parsed_data['timestamp'] = _extract_timestamp(file_path)
            
            return parsed_data
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return {}
def _parse_content(content: str) -> Dict[str, Any]:
        """解析pbtxt内容"""
        data = {}
        lines = content.strip().split('\n')
        
        current_section = None
        current_data = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是新的section开始
            if line.endswith('{'):
                section_name = line.split()[0]
                current_section = section_name
                current_data = {}
                continue
            
            # 检查是否是section结束
            if line == '}':
                if current_section:
                    data[current_section] = current_data
                    current_section = None
                continue
            
            # 解析键值对
            if ':' in line and current_section:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 处理不同类型的值
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif _is_float(value):
                    value = float(value)
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                current_data[key] = value
        
        return data
    
def _is_float(value: str) -> bool:
    """检查字符串是否可以转换为浮点数"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def _extract_timestamp(file_path: str) -> float:
    """从文件名中提取时间戳"""
    filename = os.path.basename(file_path)
    # 文件名格式如: 13545.729685380.pb.txt
    match = re.search(r'(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return 0.0


def quaternion_to_rotation_matrix(w, x, y, z):
    """将四元数转换为3x3旋转矩阵"""
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def read_pbtxt_file(file_path):
    """读取pbtxt文件并解析为字典"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    result = {}
    current_section = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '{' in line and '}' not in line:
                current_section = line.split('{')[0].strip()
                result[current_section] = {}
            elif line == '}':
                current_section = None
            elif ':' in line and current_section:
                key, value = line.split(':', 1)
                key, value = key.strip(), value.strip()
                
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
                
                result[current_section][key] = value
    
    return result

def get_vector(data, section, keys, default=0):
    """从字典中获取向量值"""
    if section not in data:
        return [default] * len(keys)
    
    return [data[section].get(key, default) for key in keys]

def create_extrinsic_matrix(extrinsics_data):
    """创建相机外参矩阵"""
    # 获取平移向量
    x, y, z = get_vector(extrinsics_data, 'translation', ['x', 'y', 'z'])
    
    # 获取四元数并计算旋转矩阵
    if 'rotation' in extrinsics_data:
        w, rx, ry, rz = get_vector(extrinsics_data, 'rotation', ['w', 'x', 'y', 'z'], 0)
        rotation_matrix = quaternion_to_rotation_matrix(w, rx, ry, rz)
    else:
        rotation_matrix = np.eye(3)
    
    # 构建4x4外参矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = [x, y, z]
    
    return extrinsic_matrix, np.linalg.inv(extrinsic_matrix)

def create_intrinsic_matrix(intrinsics_data):
    """创建相机内参矩阵和畸变系数"""
    intrinsic_matrix = np.eye(3)
    
    if 'intrinsic' in intrinsics_data:
        fx = intrinsics_data['intrinsic'].get('fx', 0)
        fy = intrinsics_data['intrinsic'].get('fy', 0)
        cx = intrinsics_data['intrinsic'].get('cx', 0)
        cy = intrinsics_data['intrinsic'].get('cy', 0)
        
        intrinsic_matrix[0, 0] = fx
        intrinsic_matrix[1, 1] = fy
        intrinsic_matrix[0, 2] = cx
        intrinsic_matrix[1, 2] = cy
    
    # 获取畸变系数
    distortion_coeffs = np.zeros(5)
    if 'pinhole_coeffs' in intrinsics_data:
        coeffs = intrinsics_data['pinhole_coeffs']
        distortion_coeffs[0] = coeffs.get('k1', 0)
        distortion_coeffs[1] = coeffs.get('k2', 0)
        distortion_coeffs[2] = coeffs.get('p1', 0)
        distortion_coeffs[3] = coeffs.get('p2', 0)
        distortion_coeffs[4] = coeffs.get('k3', 0)
    
    return intrinsic_matrix, distortion_coeffs