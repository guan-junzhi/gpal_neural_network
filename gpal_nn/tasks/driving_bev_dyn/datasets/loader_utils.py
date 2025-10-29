import cv2
import numpy as np
import os
import json

import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

from tqdm import tqdm

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_lidar_point_cloud_from_hesai_pcd(lidar_path):
    # 检查文件是否存在
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"文件不存在: {lidar_path}")

    try:
        with open(lidar_path, 'rb') as f:
            data = f.read()
            # 定义新的数据类型，匹配 PCD 文件字段
            data_type = np.dtype([
                ('x', '<f4'), 
                ('y', '<f4'), 
                ('z', '<f4'), 
                ('intensity', '<f4'),
                ('timestamp_offset', '<f4'),
                ('ring', '<u4'),
                ('echo_number', '<u4')
            ])
            # 定位二进制数据起始位置
            binary_start_index = data.find(b"DATA binary")
            if binary_start_index == -1:
                raise ValueError("未找到 'DATA binary' 标记")
            data_binary = data[binary_start_index + 12:]

            # 解析点云数量
            header_text = data[:binary_start_index].decode('utf-8')
            lines = header_text.split('\n')
            num_points_line = lines[-2]
            num_points = int(num_points_line.split(' ')[-1])

            # 从二进制数据中读取点云
            points = np.frombuffer(data_binary, dtype=data_type, count=num_points)
            points_array = points.view(np.float32).reshape(-1, len(points.dtype))
            # 过滤包含 NaN 的点
            points_array = points_array[~np.isnan(points_array).any(axis=1)]

        return points_array
    except Exception as e:
        print(f"读取点云文件 {lidar_path} 时出错: {e}")
        return np.array([])


@dataclass
class CameraIntrinsic:
    """相机内参类"""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    
    def __init__(self, cam_K: List[List[float]], cam_dist: List[float]):
        """从K矩阵和畸变系数初始化"""
        self.fx = cam_K[0][0]
        self.fy = cam_K[1][1]
        self.cx = cam_K[0][2]
        self.cy = cam_K[1][2]
        self.k1 = cam_dist[0]
        self.k2 = cam_dist[1]
        self.p1 = cam_dist[2]
        self.p2 = cam_dist[3]
        self.k3 = cam_dist[4] if len(cam_dist) > 4 else 0.0
    
    def to_matrix(self) -> np.ndarray:
        """转换为3x3内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """获取畸变系数"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

@dataclass
class CameraExtrinsic:
    """相机外参类"""
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    
    def __init__(self, extrinsic_matrix: List[List[float]]):
        """从4x4外参矩阵初始化"""
        matrix = np.array(extrinsic_matrix)
        self.rotation_matrix = matrix[:3, :3]
        self.translation_vector = matrix[:3, 3]
    
    def to_matrix(self) -> np.ndarray:
        """转换为4x4外参矩阵"""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.translation_vector
        return matrix

class CameraInfo:
    """相机信息类"""
    
    def __init__(self, name: str, camera_data: Dict):
        """从相机数据字典初始化"""
        self.name = name
        self.intrinsic = CameraIntrinsic(
            camera_data['intrinsic']['cam_K'],
            camera_data['intrinsic']['cam_dist']
        )
        self.extrinsic = CameraExtrinsic(camera_data['extrinsic'])
        self.image_size = tuple(camera_data['image_size'])
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """将3D点投影到2D图像平面"""
        # 确保输入是正确的形状
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # 转换到相机坐标系
        points_camera = self.extrinsic.rotation_matrix @ points_3d.T + self.extrinsic.translation_vector.reshape(-1, 1)
        
        # 避免除零错误
        valid_mask = points_camera[2] > 1e-6
        
        # 投影到图像平面
        K = self.intrinsic.to_matrix()
        points_2d_homogeneous = K @ points_camera
        points_2d = np.zeros((2, points_camera.shape[1]))
        points_2d[:, valid_mask] = points_2d_homogeneous[:2, valid_mask] / points_2d_homogeneous[2, valid_mask]
        
        return points_2d.T

class BoundingBox3D:
    """3D边界框类"""
    
    def __init__(self, bbox_data: Dict):
        """从边界框数据字典初始化"""
        self.track_id = bbox_data.get('track_id', '-1')
        self.main_id = bbox_data.get('main_id', '-1')
        self.object_type = bbox_data.get('type', '')
        self.type_name = bbox_data.get('typeName', '')
        
        # 位置信息
        self.position = np.array([
            float(bbox_data['position']['x']),
            float(bbox_data['position']['y']),
            float(bbox_data['position']['z'])
        ])
        
        # 旋转信息
        self.rotation = np.array([
            float(bbox_data['rotation']['roll']),
            float(bbox_data['rotation']['pitch']),
            float(bbox_data['rotation']['yaw'])
        ])
        
        # 尺寸信息
        self.size = np.array([
            float(bbox_data['size']['length']),
            float(bbox_data['size']['width']),
            float(bbox_data['size']['height'])
        ])
        
        # 速度信息
        try:
            self.velocity = np.array([
                float(bbox_data['velocity']['x']),
                float(bbox_data['velocity']['y']),
                float(bbox_data['velocity']['z'])
            ])
        except:
            self.velocity = np.array([0, 0, 0])
        
        # 其他属性
        self.occlusion = bool(bbox_data.get('occlusion', False))
        self.truncated = bool(bbox_data.get('truncated', False))
        self.roi = bool(bbox_data.get('roi', True))
        self.point_num = int(bbox_data.get('pointNum', -1))
        self.radar_state = int(bbox_data.get('radar_state', -1))
        self.image_visible = int(bbox_data.get('image_visible', -1))
        self.scene_type = str(bbox_data.get('sceneType', 'HSAI_lidar'))
        self.yawrate = float(bbox_data.get('yawrate', -1))
    
    def get_8_corners(self) -> np.ndarray:
        """获取3D框的8个角点坐标"""
        l, w, h = self.size
        
        # 定义8个角点（相对于中心点）
        corners = np.array([
            [-l/2, -w/2, -h/2],  # 0: 后右下
            [l/2, -w/2, -h/2],   # 1: 前右下
            [l/2, w/2, -h/2],    # 2: 前左下
            [-l/2, w/2, -h/2],   # 3: 后左下
            [-l/2, -w/2, h/2],   # 4: 后右上
            [l/2, -w/2, h/2],    # 5: 前右上
            [l/2, w/2, h/2],     # 6: 前左上
            [-l/2, w/2, h/2]     # 7: 后左上
        ])
        
        # 应用旋转（只考虑yaw角）
        yaw = self.rotation[2]
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        
        # 平移到世界坐标
        world_corners = rotated_corners + self.position
        
        return world_corners

class GpalDrivingJsonAnnosDataLoader:
    """Json 标注数据加载器"""
    
    def __init__(self):
        pass
    
    def load_data(self, data: Union[str, Dict]):
        
        """
        初始化数据加载器
        Args:
            data: JSON文件路径(str) 或 已解析的字典数据(Dict)
        """
        self.cameras: Dict[str, CameraInfo] = {}
        self.bounding_boxes: List[BoundingBox3D] = []
        self.meta_info = {}
        self.special_labels = []
        
        # 根据输入类型加载数据
        if isinstance(data, str):
            self._load_from_file(data)
        elif isinstance(data, dict):
            self._load_from_dict(data)
        else:
            raise ValueError("数据输入必须是文件路径(str)或字典(dict)")
    
    def _load_from_file(self, json_file_path: str):
        """从JSON文件加载数据"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._load_from_dict(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")
    
    def _load_from_dict(self, data: Dict):
        """从字典数据加载"""
        # 加载元信息
        self.meta_info = data.get('meta_infos', {})
        
        # 加载相机信息
        camera_infos = self.meta_info.get('camera_infos', {})
        for camera_name, camera_data in camera_infos.items():
            try:
                self.cameras[camera_name] = CameraInfo(camera_name, camera_data)
            except Exception as e:
                print(f"警告: 加载相机 {camera_name} 失败: {e}")
        
        # 加载3D边界框
        for i, bbox_data in enumerate(data.get('3d_attributes', [])):
            try:
                bbox = BoundingBox3D(bbox_data)
                self.bounding_boxes.append(bbox)
            except Exception as e:
                print(f"警告: 加载第{i+1}个边界框失败: {e}")
        
        # 加载特殊标签
        special_labels_data = data.get('special_labels', {})
        self.special_labels = special_labels_data.get('special_labels', [])
    
    def get_camera_by_name(self, camera_name: str) -> Optional[CameraInfo]:
        """根据名称获取相机信息"""
        return self.cameras.get(camera_name)
    
    def get_cameras_by_type(self, camera_type: str) -> List[CameraInfo]:
        """根据类型获取相机列表"""
        return [cam for name, cam in self.cameras.items() if camera_type in name]
    
    def get_bounding_boxes_by_type(self, object_type: str) -> List[BoundingBox3D]:
        """根据类型获取边界框列表"""
        return [bbox for bbox in self.bounding_boxes if object_type in bbox.object_type]
    
    def print_summary(self, is_print=True):
    
        if is_print:
            self._get_statistics()
            self._print_camera_info()
            self._print_box_info()

    def _get_statistics(self):
        """打印数据摘要"""
        print("=" * 50)
        print("自动驾驶数据摘要")
        print("=" * 50)
        print(f"时间戳: {self.meta_info.get('current_timestamp', 'Unknown')}")
        print(f"车辆速度: {self.meta_info.get('car_v', 'Unknown')} m/s")
        print(f"标签版本: {self.meta_info.get('labelVersion', 'Unknown')}")
        print()
        
        print(f"相机数量: {len(self.cameras)}")
        for name, camera in self.cameras.items():
            print(f"  - {name.replace('img_', '').replace('_', ' ').title():<15}: "
                  f"{camera.image_size[0]}x{camera.image_size[1]}")
        print()
        
        

        if self.special_labels:
            print(f"\n特殊标签: {len(self.special_labels)}")
            for label in self.special_labels:
                print(f"  - {label}")
        
        print(f"3D对象数量: {len(self.bounding_boxes)}")
        type_counts = {}
        for bbox in self.bounding_boxes:
            type_counts[bbox.object_type] = type_counts.get(bbox.object_type, 0) + 1
        
        for obj_type, count in type_counts.items():
            print(f"  - {obj_type.replace('_', ' ').title():<20}: {count}")
        
        """获取数据统计信息"""
        stats = {
            'cameras': len(self.cameras),
            'bounding_boxes': len(self.bounding_boxes),
            'object_types': {},
            'camera_types': {}
        }
        
        # 统计对象类型
        for bbox in self.bounding_boxes:
            stats['object_types'][bbox.object_type] = stats['object_types'].get(bbox.object_type, 0) + 1
        
        # 统计相机类型
        for name in self.cameras.keys():
            cam_type = name.replace('img_', '')
            stats['camera_types'][cam_type] = stats['camera_types'].get(cam_type, 0) + 1
        
        print(f"\n统计信息: {stats}")
    
    def _print_box_info(self):
        print(f"\n所有3D边界框信息, 数量 {len(self.bounding_boxes)}:")
        print(f"{'range':^5} | {'object type':^20} | {'position(xyz)':^22} | {'size(lwh)':^19} | {'yaw':5}")
        for i, bbox in enumerate(self.bounding_boxes):
            pos_str = f"{bbox.position[0]:6.2f}, {bbox.position[1]:6.2f}, {bbox.position[2]:6.2f}"
            size_str = f"{bbox.size[0]:5.2f}, {bbox.size[1]:5.2f}, {bbox.size[2]:5.2f}"
            print(f"{i:^5} | {bbox.object_type:^20} | {pos_str:^22} | {size_str:^19} | {bbox.rotation[2]:5.2f}")
        
    def _print_camera_info(self):
        print("\n所有相机信息:")
        print(f"{'camera name':<20} {'fx':^10} {'fy':^10} {'cx':^10} {'cy':^10} {'extrinsic.trans(xyz)':<30}")

        # 遍历所有相机
        for camera in loader.cameras.values():
            # 提取相机参数
            fx = camera.intrinsic.fx
            fy = camera.intrinsic.fy
            cx = camera.intrinsic.cx
            cy = camera.intrinsic.cy
            pos = camera.extrinsic.translation_vector
            # 格式化位置向量为字符串
            pos_str = f"{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}"
            # 按列对齐打印
            print(f"{str(camera.name):<20} {fx:^10.2f} {fy:^10.2f} {cx:^10.2f} {cy:^10.2f} {pos_str:<30}")

    def set_pts_range(self, pts_range: List[float] = [-102.4, -102.4, -2, 102.4, 102.4, 4]):
        self.pts_range = pts_range

    def set_class_names(self, class_names: List[str]=[], class_colors: List[str]=[]):
        if not len(class_names):
            class_names = ['vehicle_car', 
                           'vehicle_truck', 
                           'vehicle_construction_vehicle',
                           'vehicle_cyclist',
                           'vehicle_tricycle',
                           'human_pedestrian'
                           ]
        self.class_names = class_names
        self.set_class_colors(class_colors)

    def set_class_colors(self, class_colors: List[str] = []):
        if not len(class_colors):
            class_colors = ['red', 
                            'blue', 
                            'green', 
                            'yellow', 
                            'purple', 
                            'orange', 
                            'pink', 
                            'brown', 
                            'cyan'
                            ]
        
        # 定义类别颜色
        self.class_colors = {
            i:j for i, j in zip(self.class_names, class_colors[:len(self.class_names)])
        }

    def visualize_bev_and_side_box_with_lw_z_dis(self, 
                                        points=None, 
                                        timestamp=''
                                        ):
        """可视化BEV场景"""
        
        # 创建图形布局
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 左侧大图：鸟瞰图 (BEV)
        ax_bev = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
        
        x_min, y_min, z_min, x_max, y_max, z_max = self.pts_range
        CLASS_COLORS = self.class_colors
        
        if points is not None:
            points_vis = points[:, :3]  # 假设前3列是xyz坐标
            # 应用点云范围过滤
            mask = ((points_vis[:, 0] >= x_min) & (points_vis[:, 0] <= x_max) &
                    (points_vis[:, 1] >= y_min) & (points_vis[:, 1] <= y_max) &
                    (points_vis[:, 2] >= z_min) & (points_vis[:, 2] <= z_max))
            points_vis = points_vis[mask]
            # 绘制点云
            scatter = ax_bev.scatter(points_vis[:, 0], points_vis[:, 1], 
                                    c=points_vis[:, 2], 
                                    marker=',',
                                    cmap='viridis', 
                                    s=0.05, alpha=0.6)
            
            # 添加颜色条
            # plt.colorbar(scatter, ax=ax_bev, label='Z (m)', shrink=0.8)
            
        # 绘制3D框的鸟瞰投影
        for i, bbox in enumerate(self.bounding_boxes):
            center  = bbox.position
            l, w, h = bbox.size
            heading = bbox.rotation[2]
            object_type = bbox.object_type
            trackid = bbox.track_id
            
            # 计算矩形的四个角点
            corners = bbox.get_8_corners()[4:]
            
            color = CLASS_COLORS.get(object_type, 'gray')
            ax_bev.plot(corners[:, 0], corners[:, 1], color=color, linewidth=0.5)
            ax_bev.plot(corners[[0,-1]][:, 0], corners[[0,-1]][:, 1], color='k', linewidth=0.5)
            
            # ax_bev.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)  # 填充颜色
            
            # 标注trackid
            ax_bev.text(center[0], center[1], str(trackid), 
                        ha='center', va='center', fontsize=3, 
                        #bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
                        )
        
        ax_bev.set_xlim(x_min, x_max)
        ax_bev.set_ylim(y_min, y_max)
        ax_bev.set_xlabel('X (m)', fontsize=12)
        ax_bev.set_ylabel('Y (m)', fontsize=12)
        ax_bev.set_title(f"Bird's Eye View (BEV) {timestamp}", fontsize=14, fontweight='bold')
        ax_bev.set_aspect('equal')
        ax_bev.grid(True, alpha=0.3)
        
        
        
        # 2. 右上左图：长宽尺寸散点分布
        ax_lw = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=1)
        
        # 收集所有有效标签的长宽数据
        lengths = []
        widths = []
        colors_lw = []
        
        for i, bbox in enumerate(self.bounding_boxes):
            l, w, h = bbox.size
            object_type = bbox.object_type
            lengths.append(l)
            widths.append(w)
            colors_lw.append(CLASS_COLORS.get(object_type, 'gray'))
        
        if lengths and widths:
            ax_lw.scatter(lengths, widths, c=colors_lw, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax_lw.set_xlabel('Length (m)', fontsize=10)
        ax_lw.set_ylabel('Width (m)', fontsize=10)
        ax_lw.set_title('L-W Distribution', fontsize=12, fontweight='bold')
        ax_lw.set_aspect('equal')
        ax_lw.grid(True, alpha=0.3)
        
        # 3. 右上右图：类别统计
        ax_class = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1)
        
        # 统计类别
        class_counts = Counter()
        for i, bbox in enumerate(self.bounding_boxes):
            object_type = bbox.object_type
            class_counts[object_type] += 1
        
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors_bar = [CLASS_COLORS.get(clss, 'gray') for clss in classes]
            
            bars = ax_class.bar(range(len(classes)), counts, color=colors_bar, alpha=0.7, edgecolor='black')
            ax_class.set_xticks(range(len(classes)))
            ax_class.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
            
            # 在柱子上添加数值
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax_class.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{count}', ha='center', va='bottom', fontsize=9)
        
        ax_class.set_ylabel('Count', fontsize=10)
        ax_class.set_title('Class Statistics', fontsize=12, fontweight='bold')
        ax_class.grid(True, alpha=0.3, axis='y')
        
        # 4. 右下图：侧视图 (从Y轴方向看)
        ax_side = plt.subplot2grid((2, 4), (1, 2), colspan=2, rowspan=1)
        
        # 绘制点云侧视图 (X-Z平面)
        if points is not None:
            scatter_side = ax_side.scatter(points_vis[:, 0], points_vis[:, 2], 
                                            c=points_vis[:, 1], cmap='plasma', 
                                            s=0.3, alpha=0.6)
        # 绘制3D框的侧视投影
        
        for bbox in self.bounding_boxes:
            center = bbox.position
            l, w, h = bbox.size
            heading = bbox.rotation[2]
            object_type = bbox.object_type
                
            # 侧视图中的矩形 (在X-Z平面)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            
            # 计算在X-Z平面的投影角点
            x_corners = np.array([-l/2, l/2, l/2, -l/2, -l/2])
            z_corners = np.array([-h/2, -h/2, h/2, h/2, -h/2])
            
            # 应用旋转到X坐标
            x_rotated = x_corners * cos_h + center[0]
            z_absolute = z_corners + center[2]
            
            color = CLASS_COLORS.get(object_type, 'gray')
            ax_side.plot(x_rotated, z_absolute, color=color, linewidth=1)
            # ax_side.fill(x_rotated, z_absolute, color=color, alpha=0.2)
                
            # 标注类别
            ax_side.text(center[0], center[2], 
                         '', 
                         #ha='center', va='center', fontsize=8,
                         #bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
                        )
    
            ax_side.set_xlim(x_min, x_max)
            ax_side.set_ylim(z_min, z_max)
            ax_side.set_xlabel('X (m)', fontsize=12)
            ax_side.set_ylabel('Z (m)', fontsize=12)
            ax_side.set_title('Side View (Y-axis perspective)', fontsize=14, fontweight='bold')
            ax_side.set_aspect('equal')
            ax_side.grid(True, alpha=0.3)
        
        
        plt.savefig('bev.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def visualize_bev_and_side_box(self,
                                   points=None, 
                                   timestamp='',
                                   save_path='bev.png'
                                   ):
        """可视化BEV和侧视图场景"""
        
        # 创建图形布局
        x_min, y_min, z_min, x_max, y_max, z_max = self.pts_range
        CLASS_COLORS = self.class_colors
        
        atios = 15
        W, H = int((x_max-x_min)/atios), (int(y_max-y_min)/atios)
        fig = plt.figure(figsize=(W, H))
        
        
        # 修改gridspec设置
        x_range = (x_max - x_min)
        y_range = (y_max - y_min)
        z_range = (z_max - z_min)

        # 根据实际数据范围计算高度比例
        height_ratio_bev = y_range
        height_ratio_side = z_range

        gs = fig.add_gridspec(2, hspace=0, wspace=0, 
                              height_ratios=[height_ratio_side, height_ratio_bev])
        ax_side, ax_bev = gs.subplots()
        
        if points is not None:
            points_vis = points[:, :3]  # 假设前3列是xyz坐标
            # 应用点云范围过滤
            mask = ((points_vis[:, 0] >= x_min) & (points_vis[:, 0] <= x_max) &
                    (points_vis[:, 1] >= y_min) & (points_vis[:, 1] <= y_max) &
                    (points_vis[:, 2] >= z_min) & (points_vis[:, 2] <= z_max))
            points_vis = points_vis[mask]
            
            # BEV: 绘制 X-Y 平面
            vmin_val = np.percentile(points_vis[:, 2], 10)  # 取第10百分位数
            vmax_val = np.percentile(points_vis[:, 2], 90)  # 取第90百分位数
            scatter_bev = ax_bev.scatter(points_vis[:, 0], points_vis[:, 1], 
                                         c=points_vis[:, 2], 
                                         marker='.',
                                        #  cmap='viridis', 
                                        # vmin=vmin_val,
                                        # vmax=vmax_val,
                                        cmap='gray',
                                        s=0.01, 
                                         )
            # 侧视图: 绘制 X-Z 平面
            scatter_side = ax_side.scatter(points_vis[:, 0], points_vis[:, 2], 
                                           c=points_vis[:, 1], 
                                           marker=',',
                                           cmap='plasma', 
                                           s=0.05, alpha=0.6)
            # plt.colorbar(scatter, ax=ax_bev, label='Z (m)', shrink=0.8)  # 添加颜色条
        
        for i, bbox in enumerate(self.bounding_boxes):
            center  = bbox.position
            
            exist_in = ((center[0] >= x_min) & (center[0] <= x_max) &
                       (center[1] >= y_min) & (center[1] <= y_max) &
                       (center[2] >= z_min) & (center[2] <= z_max))
            if not exist_in:
                continue
            
            l, w, h = bbox.size
            heading = bbox.rotation[2]
            object_type = bbox.object_type
            trackid = bbox.track_id
            
            # 计算矩形的四个角点
            corners = bbox.get_8_corners()
            color = CLASS_COLORS.get(object_type, 'gray')
            
            # BEV: 绘制 X-Y 平面
            bev_corners      = corners[[6,7,4,5]]
            bev_corners_head = corners[[5,6]]
            ax_bev.plot(bev_corners[:, 0], bev_corners[:, 1], color=color, linewidth=1.0)
            ax_bev.plot(bev_corners_head[:, 0], bev_corners_head[:, 1], color='k', linewidth=1.0)

            # 中心朝线
            front_center = (corners[5] + corners[6]) / 2
            f1_center = (front_center + center) / 2
            f2_center = front_center * 2 - f1_center
            ax_bev.plot([f2_center[0], front_center[0]], [f2_center[1], front_center[1]], color='k', linewidth=1.0)
            # ax_bev.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)  # 填充颜色
            
            # track_id
            ax_bev.text(center[0], center[1], str(trackid), 
                        ha='center', va='center', fontsize=6, 
                        #bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
                        )
                        
            # 侧视图: 绘制 X-Z 平面
            xz_corners      = corners[[5,4,0,1]]
            xz_corners_head = corners[[1,5]]
            ax_side.plot(xz_corners[:, 0], xz_corners[:, 2], color=color, linewidth=0.5)
            ax_side.plot(xz_corners_head[:, 0], xz_corners_head[:, 2], color='k', linewidth=0.5)
        
        # === 设置BEV图属性 ===
        ax_bev.set_xlim(x_min, x_max)
        ax_bev.set_ylim(y_min, y_max)
        ax_bev.set_xlabel('X (m)', fontsize=12)
        ax_bev.set_ylabel('Y (m)', fontsize=12)
        ax_bev.set_aspect('equal')
        # ax_bev.set_title(f"Bird's Eye View (BEV)(Y-axis) {timestamp}", fontsize=14, fontweight='bold')
        ax_bev.grid(True, alpha=0.3)
        
        # === 设置侧视图属性 ===
        ax_side.set_xlim(x_min, x_max)
        ax_side.set_ylim(z_min, z_max)
        # ax_side.set_xlabel('X (m)', fontsize=12)
        ax_side.set_ylabel('Z (m)', fontsize=12)
        # ax_side.set_title('Side View (Y-axis perspective)', fontsize=14, fontweight='bold')
        ax_side.set_title(f"Bird's Eye View (BEV)(Y-axis) {timestamp}", fontsize=14, fontweight='bold')
        ax_side.set_aspect('equal')
        ax_side.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
        plt.close()
        
        
    def visualize_3d_scene(self, save_path: Optional[str] = None, show_cameras: bool = True):
        """可视化3D场景"""
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 定义颜色映射
        color_map = {
            'vehicle_car': 'blue',
            'vehicle_truck': 'green', 
            'vehicle_bus': 'purple',
            'human_pedestrian': 'red',
            'bicycle': 'orange',
            'motorcycle': 'yellow'
        }
        
        # 绘制每个3D框
        for i, bbox in enumerate(self.bounding_boxes):
            corners = bbox.get_8_corners()
            color = color_map.get(bbox.object_type, 'gray')
            
            # 绘制底面 (0-1-2-3-0)
            bottom_indices = [0, 1, 2, 3, 0]
            bottom_corners = corners[bottom_indices]
            ax.plot(bottom_corners[:, 0], bottom_corners[:, 1], bottom_corners[:, 2], 
                   color=color, linewidth=2, alpha=0.8)
            
            # 绘制顶面 (4-5-6-7-4)
            top_indices = [4, 5, 6, 7, 4]
            top_corners = corners[top_indices]
            ax.plot(top_corners[:, 0], top_corners[:, 1], top_corners[:, 2], 
                   color=color, linewidth=2, alpha=0.8)
            
            # 绘制垂直边
            for j in range(4):
                ax.plot([corners[j, 0], corners[j+4, 0]], 
                       [corners[j, 1], corners[j+4, 1]], 
                       [corners[j, 2], corners[j+4, 2]], 
                       color=color, linewidth=2, alpha=0.8)
            
            # 添加对象标签
            label_text = f'{bbox.object_type.replace("_", " ").title()}\nID:{bbox.track_id}'
            ax.text(bbox.position[0], bbox.position[1], bbox.position[2] + bbox.size[2]/2 + 0.5,
                   label_text, fontsize=8, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # 绘制相机位置
        if show_cameras:
            for name, camera in self.cameras.items():
                if 'fisheye' not in name:  # 只显示主要相机
                    pos = camera.extrinsic.translation_vector
                    ax.scatter(pos[0], pos[1], pos[2], c='orange', s=150, marker='^', 
                             edgecolors='black', linewidth=1)
                    
                    # 简化相机名称
                    cam_label = name.replace('img_', '').replace('_', ' ').title()
                    ax.text(pos[0], pos[1], pos[2] + 1.5, cam_label, 
                           fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='orange', alpha=0.7))
        
        # 设置图形属性
        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_zlabel('Z (米)', fontsize=12)
        ax.set_title('3D场景可视化 - 自动驾驶感知数据', fontsize=14, fontweight='bold')
        
        # 计算合适的显示范围
        if self.bounding_boxes:
            all_positions = np.array([bbox.position for bbox in self.bounding_boxes])
            x_range = [all_positions[:, 0].min() - 10, all_positions[:, 0].max() + 10]
            y_range = [all_positions[:, 1].min() - 10, all_positions[:, 1].max() + 10]
            z_range = [0, max(all_positions[:, 2].max() + 5, 10)]
        else:
            x_range = [-50, 50]
            y_range = [-50, 50]
            z_range = [0, 10]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = []
        for obj_type, color in color_map.items():
            if any(bbox.object_type == obj_type for bbox in self.bounding_boxes):
                legend_elements.append(Line2D([0], [0], color=color, lw=3, 
                                            label=obj_type.replace('_', ' ').title()))
        
        if show_cameras:
            legend_elements.append(Line2D([0], [0], marker='^', color='orange', lw=0, 
                                        markersize=12, label='相机'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D场景图已保存到: {save_path}")
        
        plt.show()



class InitJsonFile:
    def __init__(self, class_names, pts_range):
        self.class_names = class_names
        self.pts_range = pts_range
        
        self._init()
        # 其它: 定义类别颜色
        class_colors = ['red', 
                        'blue', 
                        'green', 
                        'yellow', 
                        'purple', 
                        'orange', 
                        'pink', 
                        'brown', 
                        'cyan'
                        ]
        
        # 定义类别颜色
        self.class_colors = {
            i:j for i, j in zip(self.class_names, class_colors[:len(self.class_names)])
        }
    
    def _init(self):
        self.cameras: Dict[str, CameraInfo] = {}
        # self.bounding_boxes: List[BoundingBox3D] = []
        self.freespace = []
        self.meta_info = {}
        self.special_labels = []

    def load(self, json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        return json_data

    def parse_json(self, json_data):
        
        self._init()  # 重置
        
        # 加载元信息
        self.meta_info = json_data.get('meta_infos', {})
        
        # 加载相机信息
        camera_infos = self.meta_info.get('camera_infos', {})
        for camera_name, camera_data in camera_infos.items():
            try:
                self.cameras[camera_name] = CameraInfo(camera_name, camera_data)
            except Exception as e:
                print(f"警告: 加载相机 {camera_name} 失败: {e}")
        
      
            freespace_str = json_data["3d_attributes"]["position"]
            self.freespace = np.array(freespace_str, dtype=np.float32)


        
        # 加载特殊标签
        # special_labels_data = json_data.get('special_labels', {})
        # self.special_labels = json_data.get('special_labels', [])
        
        return self.meta_info, self.cameras, self.freespace
    
    def parse_json_freespace(self, json_data):
        
        self._init()  # 重置
        
        # 加载元信息
        self.meta_info = json_data.get('meta_infos', {})
        
        # 加载相机信息
        camera_infos = self.meta_info.get('camera_infos', {})
        for camera_name, camera_data in camera_infos.items():
            try:
                self.cameras[camera_name] = CameraInfo(camera_name, camera_data)
            except Exception as e:
                print(f"警告: 加载相机 {camera_name} 失败: {e}")
        
        # 加载3D边界框
        for i, bbox_data in enumerate(json_data.get('3d_attributes', [])):
            try:
                bbox = BoundingBox3D(bbox_data)
                self.bounding_boxes.append(bbox)
            except Exception as e:
                print(f"警告: 加载第{i+1}个边界框失败: {e}")
        
        # 加载特殊标签
        # special_labels_data = json_data.get('special_labels', {})
        self.special_labels = json_data.get('special_labels', [])
        
        return self.meta_info, self.cameras, self.bounding_boxes, self.special_labels

    def visualize_bev_and_side_box(self,
                                   points=None, 
                                   timestamp='',
                                   save_path='bev.png'
                                   ):
        """可视化BEV和侧视图场景"""
        
        # 创建图形布局
        x_min, y_min, z_min, x_max, y_max, z_max = self.pts_range
        CLASS_COLORS = self.class_colors
        
        atios = 15
        W, H = int((x_max-x_min)/atios), (int(y_max-y_min)/atios)
        fig = plt.figure(figsize=(W, H))
        
        # 修改gridspec设置
        x_range = (x_max - x_min)
        y_range = (y_max - y_min)
        z_range = (z_max - z_min)

        # 根据实际数据范围计算高度比例
        height_ratio_bev = y_range
        height_ratio_side = z_range

        gs = fig.add_gridspec(2, hspace=0, wspace=0, 
                              height_ratios=[height_ratio_side, height_ratio_bev])
        ax_side, ax_bev = gs.subplots()
        
        if points is not None:
            points_vis = points[:, :3]  # 假设前3列是xyz坐标
            # 应用点云范围过滤
            mask = ((points_vis[:, 0] >= x_min) & (points_vis[:, 0] <= x_max) &
                    (points_vis[:, 1] >= y_min) & (points_vis[:, 1] <= y_max) &
                    (points_vis[:, 2] >= z_min) & (points_vis[:, 2] <= z_max))
            points_vis = points_vis[mask]
            
            # BEV: 绘制 X-Y 平面
            vmin_val = np.percentile(points_vis[:, 2], 10)  # 取第10百分位数
            vmax_val = np.percentile(points_vis[:, 2], 90)  # 取第90百分位数
            scatter_bev = ax_bev.scatter(points_vis[:, 0], points_vis[:, 1], 
                                         c=points_vis[:, 2], 
                                         marker='.',
                                        #  cmap='viridis', 
                                        # vmin=vmin_val,
                                        # vmax=vmax_val,
                                        cmap='gray',
                                        s=0.01, 
                                         )
            # 侧视图: 绘制 X-Z 平面
            scatter_side = ax_side.scatter(points_vis[:, 0], points_vis[:, 2], 
                                           c=points_vis[:, 1], 
                                           marker=',',
                                           cmap='plasma', 
                                           s=0.05, alpha=0.6)
            # plt.colorbar(scatter, ax=ax_bev, label='Z (m)', shrink=0.8)  # 添加颜色条
        
        for i, bbox in enumerate(self.bounding_boxes):
            center  = bbox.position
            
            exist_in = ((center[0] >= x_min) & (center[0] <= x_max) &
                       (center[1] >= y_min) & (center[1] <= y_max) &
                       (center[2] >= z_min) & (center[2] <= z_max))
            if not exist_in:
                continue
            
            l, w, h = bbox.size
            heading = bbox.rotation[2]
            object_type = bbox.object_type
            trackid = bbox.track_id
            
            # 计算矩形的四个角点
            corners = bbox.get_8_corners()
            color = CLASS_COLORS.get(object_type, 'gray')
            
            # BEV: 绘制 X-Y 平面
            bev_corners      = corners[[6,7,4,5]]
            bev_corners_head = corners[[5,6]]
            ax_bev.plot(bev_corners[:, 0], bev_corners[:, 1], color=color, linewidth=1.0)
            ax_bev.plot(bev_corners_head[:, 0], bev_corners_head[:, 1], color='k', linewidth=1.0)

            # 中心朝线
            front_center = (corners[5] + corners[6]) / 2
            f1_center = (front_center + center) / 2
            f2_center = front_center * 2 - f1_center
            ax_bev.plot([f2_center[0], front_center[0]], [f2_center[1], front_center[1]], color='k', linewidth=1.0)
            # ax_bev.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)  # 填充颜色
            
            # track_id
            ax_bev.text(center[0], center[1], str(trackid), 
                        ha='center', va='center', fontsize=6, 
                        #bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
                        )
                        
            # 侧视图: 绘制 X-Z 平面
            xz_corners      = corners[[5,4,0,1]]
            xz_corners_head = corners[[1,5]]
            ax_side.plot(xz_corners[:, 0], xz_corners[:, 2], color=color, linewidth=0.5)
            ax_side.plot(xz_corners_head[:, 0], xz_corners_head[:, 2], color='k', linewidth=0.5)
        
        # === 设置BEV图属性 ===
        ax_bev.set_xlim(x_min, x_max)
        ax_bev.set_ylim(y_min, y_max)
        ax_bev.set_xlabel('X (m)', fontsize=12)
        ax_bev.set_ylabel('Y (m)', fontsize=12)
        ax_bev.set_aspect('equal')
        # ax_bev.set_title(f"Bird's Eye View (BEV)(Y-axis) {timestamp}", fontsize=14, fontweight='bold')
        ax_bev.grid(True, alpha=0.3)
        
        # === 设置侧视图属性 ===
        ax_side.set_xlim(x_min, x_max)
        ax_side.set_ylim(z_min, z_max)
        # ax_side.set_xlabel('X (m)', fontsize=12)
        ax_side.set_ylabel('Z (m)', fontsize=12)
        # ax_side.set_title('Side View (Y-axis perspective)', fontsize=14, fontweight='bold')
        ax_side.set_title(f"Bird's Eye View (BEV)(Y-axis) {timestamp}", fontsize=14, fontweight='bold')
        ax_side.set_aspect('equal')
        ax_side.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
        plt.close()
        
def read_camera_yaml_to_dict(yaml_file):

    # 初始化数据字典
    yaml_dict = {}

    # 读取YAML文件
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)

    camera_matrix = fs.getNode("camera_matrix").mat()
    distortion_coefficients = fs.getNode("distortion_coefficients").mat()
    r_mat = fs.getNode("r_mat").mat()
    t_vec = fs.getNode("t_vec").mat()

    # 关闭文件
    fs.release()

    yaml_dict['camera_matrix'] = camera_matrix
    yaml_dict['distortion_coefficients'] = distortion_coefficients
    yaml_dict['r_mat'] = r_mat
    yaml_dict['t_vec'] = t_vec

    return yaml_dict

if __name__ == "__main__":

    
    root_data_dir = "/data/dp_group/process-prod-bucket/business_datasets/obstacle_data"
    
    root_json_day_dir_list = [
        '/data/ai_group/workdirs/od_occ_group/qiyingli/ManualLabelPost/EKART_ID4001_2025-07-01-13-18-12',
    ]
    
    
    loader = GpalDrivingJsonAnnosDataLoader()
    class_names = ['vehicle_car', 'vehicle_truck', 'vehicle_construction_vehicle','vehicle_cyclist','vehicle_tricycle','human_pedestrian']
    
    loader.set_pts_range(pts_range=[-102.4, -80, -2, 102.4, 80, 6])
    loader.set_class_names(class_names)
    loader.set_class_colors()
    
    for curr_daydir in root_json_day_dir_list:
        
        curr_day = os.path.basename(curr_daydir)
        
        subday_dirname_list = sorted(os.listdir(curr_daydir))
        for curr_subdirname in subday_dirname_list:
            curr_sub_daydir = os.path.join(curr_daydir, curr_subdirname)
            if not os.path.isdir(curr_sub_daydir):
                continue
            curr_sub_data_dir = os.path.join(root_data_dir, curr_day, curr_subdirname)
            
            curr_json_list = sorted(os.listdir(curr_sub_daydir))
            for curr_jsonname in tqdm(curr_json_list):
                curr_jsonfile = os.path.join(curr_sub_daydir, curr_jsonname)
                if not curr_jsonname.endswith('.json'):
                    continue
                
                # 数据读取完成
                json_data = read_json_file(curr_jsonfile)
                
                curr_timestamp = curr_jsonname.replace('.json', '')
                lidar_point_path = f'{curr_sub_data_dir}/rslidar_aligned/{curr_timestamp}.pcd'
                lidar_pts = read_lidar_point_cloud_from_hesai_pcd(lidar_point_path)  # xyzi
                
                loader.load_data(json_data)
                
                loader.print_summary(is_print=False)
                # loader.visualize_bev_and_side_box_with_lw_z_dis(points=lidar_pts)
                
                save_dir = f'./vis/{curr_day}/{curr_subdirname}'
                os.makedirs(f'{save_dir}', exist_ok=True)
                loader.visualize_bev_and_side_box(points=lidar_pts, save_path=f'{save_dir}/{curr_timestamp}.png')
                # loader.visualize_3d_scene('3d_scene.png')
                
                a= 1            