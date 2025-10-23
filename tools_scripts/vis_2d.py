import os
import numpy as np
import cv2
import copy


class Vis2D():
    # u -> Y-
    # v -> X-
    def __init__(self, map_x, map_y, resolution):
        self.resolution = resolution
        map_h = int((map_x[1] - map_x[0]) / resolution)
        map_w = int((map_y[1] - map_y[0]) / resolution)

        self.map_center_v = int((map_x[1] - 0) / resolution)
        self.map_center_u = int((map_y[1] - 0) / resolution)

        # print("map_h = ", map_h, "  map_w = ", map_w)
        # print("map_center_v = ", self.map_center_v, "  map_center_u = ", self.map_center_u)

        self.base_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)

        for ele in range(int(map_x[0] / 10.0)-1, int(map_x[1] / 10.0)+1):
            self.DrawLineWithMap(self.base_map, [
                                 ele * 10.0, map_y[0]], [ele * 10.0, map_y[1]], color=(50, 50, 50), line_width=2)
        for ele in range(int(map_y[0] / 10.0)-1, int(map_y[1] / 10.0)+1):
            self.DrawLineWithMap(self.base_map, [
                                 map_x[0], ele * 10.0], [map_x[1], ele * 10.0], color=(50, 50, 50), line_width=2)

        self.DrawLineWithMap(self.base_map, [0.0, 0.0], [
                             5.0, 0.0], color=(0, 0, 255), line_width=5)
        self.DrawLineWithMap(self.base_map, [0.0, 0.0], [
                             0.0, 5.0], color=(0, 255, 0), line_width=5)

        self.map = copy.deepcopy(self.base_map)

    def XyToUv(self, x, y):
        v = self.map_center_v - x / self.resolution
        u = self.map_center_u - y / self.resolution
        return (int(u), int(v))

    def XysToUvs(self, xs, ys):
        # print(xs, ys)
        vs = self.map_center_v - xs / self.resolution
        us = self.map_center_u - ys / self.resolution
        return us.astype(np.int32), vs.astype(np.int32)

    def DrawLineWithMap(self, map, xy1, xy2, color, line_width=1):
        # print(xy1[0], xy1[1], self.XyToUv(xy1[0], xy1[1]))
        cv2.line(map, self.XyToUv(xy1[0], xy1[1]), self.XyToUv(
            xy2[0], xy2[1]), color, line_width)

    # def DrawLine(self, xy1, xy2, color, line_width=1):
    #     # print(xy1[0], xy1[1], self.XyToUv(xy1[0], xy1[1]))
    #     cv2.line(self.map, self.XyToUv(xy1[0], xy1[1]), self.XyToUv(
    #         xy2[0], xy2[1]), color, line_width)

    # def DrawPolyline(self, pts, color, line_width=1):
    #     # pts N*2
    #     if (len(pts.shape) != 2) or (pts.shape[0] < 2) or (pts.shape[0] < 2):
    #         return
    #     for p, q in zip(pts[:-1], pts[1:]):
    #         self.DrawLine([p[0], p[1]], [q[0], q[1]],
    #                       color, line_width=line_width)

    def DrawLine(self, xy1, xy2, color, line_width=1, line_type='solid', dash_length=10):
        """
        绘制直线，可以是实线或虚线
        
        参数:
            xy1: 起点坐标 (x, y)
            xy2: 终点坐标 (x, y)
            color: 线条颜色
            line_width: 线条宽度
            line_type: 线条类型，'solid' 为实线，'dashed' 为虚线
            dash_length: 虚线中每段实线的长度，仅在line_type为'dashed'时有效
        """
        # 转换坐标
        uv1 = self.XyToUv(xy1[0], xy1[1])
        uv2 = self.XyToUv(xy2[0], xy2[1])
        
        if line_type == 'solid':
            # 绘制实线
            cv2.line(self.map, uv1, uv2, color, line_width, cv2.LINE_AA)
        elif line_type == 'dashed':
            # 绘制虚线
            # 计算线段总长度
            dx = uv2[0] - uv1[0]
            dy = uv2[1] - uv1[1]
            distance = (dx**2 + dy**2)**0.5
            
            # 计算单位向量
            if distance > 0:
                unit_dx = dx / distance
                unit_dy = dy / distance
                
                # 绘制虚线分段
                current = 0
                while current < distance:
                    # 计算当前段的终点
                    end = min(current + dash_length, distance)
                    x1 = int(uv1[0] + unit_dx * current)
                    y1 = int(uv1[1] + unit_dy * current)
                    x2 = int(uv1[0] + unit_dx * end)
                    y2 = int(uv1[1] + unit_dy * end)
                    
                    # 绘制当前段（实线部分）
                    cv2.line(self.map, (x1, y1), (x2, y2), color, line_width, cv2.LINE_8)
                    
                    # 跳过空白部分（长度等于实线部分）
                    current += 2 * dash_length

    # def DrawPolyline(self, pts, color, line_width=1, line_type='solid', dash_length=10):
    #     """
    #     绘制多边形线条，可以是实线或虚线
        
    #     参数:
    #         pts: 点集，形状为N*2
    #         color: 线条颜色
    #         line_width: 线条宽度
    #         line_type: 线条类型，'solid' 为实线，'dashed' 为虚线
    #         dash_length: 虚线中每段实线的长度，仅在line_type为'dashed'时有效
    #     """
    #     # 检查输入点集的有效性
    #     if len(pts.shape) != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
    #         return
        
    #     # 依次绘制每一段线
    #     for p, q in zip(pts[:-1], pts[1:]):
    #         self.DrawLine(
    #             [p[0], p[1]], 
    #             [q[0], q[1]], 
    #             color, 
    #             line_width=line_width,
    #             line_type=line_type,
    #             dash_length=dash_length
    #         )
    def DrawPolyline(self, pts, color, line_width=1, shape_type=0, dash_length=10, 
                    line_thickness='normal', color2=(128, 128, 128), dot_size=2):
        """
        绘制扩展车道线（车体系坐标适配版），支持11种类型的多边形线条
        
        参数说明：
            pts: 点集，形状为N*2（numpy数组或列表，坐标为车体系X、Y）
            其他参数同前，新增：车体系下偏移方向以X向前、Y向左为正
        """
        # 定义支持的线条类型集合
        line_types_dict = ['solid', 'dashed', 'double_left_solid', 'double_right_solid',
            'thick_solid', 'thick_dashed', 'colorful_three_solid', 'reversible_line',
            'variable_lane', 'point_line', 'other']
        if isinstance(shape_type, int) and 0 <= shape_type < len(line_types_dict):
            line_type = line_types_dict[shape_type]
        else:
            # 无效索引时默认使用'other'
            line_type = 'other'
        color2 = color
        # 检查输入有效性
        if not isinstance(pts, np.ndarray):
            pts = np.array(pts)
        if len(pts.shape) != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            return
        
        # 处理"其他"类型：强制灰色实线
        if line_type == 'other':
            color = (128, 128, 128)
            line_type = 'solid'
            line_width = 1
        
        # 调整粗线宽度（基础线宽*3）
        actual_line_width = line_width * 3 if line_type in ['thick_solid', 'thick_dashed'] else line_width
        
        # 分类型绘制逻辑（车体系坐标适配）
        if line_type in ['solid', 'dashed', 'thick_solid', 'thick_dashed', 'other']:
            # 基础类型：单条线
            for p, q in zip(pts[:-1], pts[1:]):
                self.DrawLine(
                    [p[0], p[1]], [q[0], q[1]],
                    color=color,
                    line_width=actual_line_width,
                    line_type=line_type.replace('thick_', ''),
                    dash_length=dash_length
                )
        
        elif line_type in ['double_left_solid', 'double_right_solid']:
            # 双实线（左/右侧重）：双线并行，车体系下左偏移向Y正方向
            line_spacing = line_width * 2  # 双线间距
            for p, q in zip(pts[:-1], pts[1:]):
                dx = q[0] - p[0]  # 线段X方向变化（车体系：向前为正）
                dy = q[1] - p[1]  # 线段Y方向变化（车体系：向左为正）
                length = np.sqrt(dx**2 + dy**2)
                if length < 1e-3:
                    continue
                
                # 车体系下的左偏移向量（垂直于线段，指向Y正方向）
                # 推导：原方向向量(dx, dy)，垂直向量为(dy, -dx)（点积dx*dy + dy*(-dx)=0），确保向左偏移
                vx = (dy / length) * line_spacing  # X方向偏移量
                vy = (-dx / length) * line_spacing  # Y方向偏移量（左为正）
                
                # 主线条（根据类型确定左右位置）
                if line_type == 'double_left_solid':
                    # 左线为主实线，右线为副实线（向右偏移=左偏移向量取反）
                    self.DrawLine([p[0], p[1]], [q[0], q[1]],
                                color=color, line_width=actual_line_width,
                                line_type='solid', dash_length=dash_length)
                    # 右线（偏移后）
                    p_sub = [p[0] - vx, p[1] - vy]  # 右偏移=左偏移*-1
                    q_sub = [q[0] - vx, q[1] - vy]
                    self.DrawLine(p_sub, q_sub,
                                color=color2, line_width=actual_line_width,
                                line_type='dashed', dash_length=dash_length)
                else:  # double_right_solid
                    # 右线为主实线，左线为副实线
                    self.DrawLine([p[0] - vx, p[1] - vy], [q[0] - vx, q[1] - vy],
                                color=color, line_width=actual_line_width,
                                line_type='dashed', dash_length=dash_length)
                    # 左线（原位置）
                    self.DrawLine([p[0], p[1]], [q[0], q[1]],
                                color=color2, line_width=actual_line_width,
                                line_type='solid', dash_length=dash_length)
        
        elif line_type == 'colorful_three_solid':
            # 彩色三实线：三条并行，车体系下左/右偏移对称
            line_spacing = line_width * 1.5  # 线间距
            for p, q in zip(pts[:-1], pts[1:]):
                dx = q[0] - p[0]
                dy = q[1] - p[1]
                length = np.sqrt(dx**2 + dy**2)
                if length < 1e-3:
                    continue
                
                # 车体系左偏移向量（垂直向左）
                vx = (dy / length) * line_spacing
                vy = (-dx / length) * line_spacing
                
                # 左线（主色）：向左偏移
                p_left = [p[0] + vx, p[1] + vy]
                q_left = [q[0] + vx, q[1] + vy]
                self.DrawLine(p_left, q_left, color=color, line_width=actual_line_width, line_type='solid')
                
                # 中线（副色）：原位置
                self.DrawLine([p[0], p[1]], [q[0], q[1]], color=color2, line_width=actual_line_width, line_type='solid')
                
                # 右线（主色）：向右偏移（左偏移向量取反）
                p_right = [p[0] - vx, p[1] - vy]
                q_right = [q[0] - vx, q[1] - vy]
                self.DrawLine(p_right, q_right, color=color, line_width=actual_line_width, line_type='solid')
        
        elif line_type == 'reversible_line':
            # 潮汐车道线：双虚线（车体系下双线左右分布）
            line_spacing = line_width * 3
            short_line_length = dash_length * 0.8
            for p, q in zip(pts[:-1], pts[1:]):
                dx = q[0] - p[0]
                dy = q[1] - p[1]
                length = np.sqrt(dx**2 + dy**2)
                if length < 1e-3:
                    continue
                
                # 车体系左偏移向量（垂直向左）
                vx = (dy / length) * line_spacing
                vy = (-dx / length) * line_spacing
                
                # 左线（主色）
                p_left = [p[0] + vx/2, p[1] + vy/2]  # 左半间距
                q_left = [q[0] + vx/2, q[1] + vy/2]
                self.DrawLine(p_left, q_left, color=color, line_width=actual_line_width, line_type='dashed')
                
                # 右线（主色）
                p_right = [p[0] - vx/2, p[1] - vy/2]  # 右半间距
                q_right = [q[0] - vx/2, q[1] - vy/2]
                self.DrawLine(p_right, q_right, color=color, line_width=actual_line_width, line_type='dashed')

        
        elif line_type == 'variable_lane': #紫色
            for p, q in zip(pts[:-1], pts[1:]):
                self.DrawLine(
                    [p[0], p[1]], [q[0], q[1]],
                    color=[250, 51, 153],
                    line_width=actual_line_width,
                    line_type='solid',
                    dash_length=dash_length
                )
        
        elif line_type == 'point_line': #粉色
            # 波特点线：车体系下按点集绘制连续点
            for p, q in zip(pts[:-1], pts[1:]):
                self.DrawLine(
                    [p[0], p[1]], [q[0], q[1]],
                    color=[203, 192, 255],
                    line_width=actual_line_width,
                    line_type='solid',
                    dash_length=dash_length
                )

    def DrawKeypoint(self, xy, r, color, thickness=1):
        cv2.circle(self.map, self.XyToUv(xy[0], xy[1]), int(r), color, -1)

    def DrawText(self, map, xy1, txt, color, scale, line_width=1):
        # print(xy1[0], xy1[1], self.XyToUv(xy1[0], xy1[1]))\
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(map, txt, self.XyToUv(
            xy1[0], xy1[1]), font, scale, color, line_width)

    def DrawCircle(self, xy, r, color, thickness=1):
        cv2.circle(self.map, self.XyToUv(xy[0], xy[1]), int(
            r/self.resolution), color, thickness)

    def Clear(self):
        self.map = copy.deepcopy(self.base_map)

    def DrawBbox(self, tf, size, vel, color, color_head, line_width=1):
        hsx = size[0] * 0.5
        hsy = size[1] * 0.5

        p = np.matrix([[-hsx, -hsy, 0, 1],
                       [-hsx, hsy, 0, 1],
                       [hsx, hsy, 0, 1],
                       [hsx, -hsy, 0, 1],
                       ])
        p_w = np.array(np.matrix(tf) * p.T)
        p_w = p_w[:2].T

        self.DrawLineWithMap(self.map, p_w[0], p_w[1], color, line_width)
        self.DrawLineWithMap(self.map, p_w[1], p_w[2], color, line_width)
        self.DrawLineWithMap(self.map, p_w[2], p_w[3], color_head, line_width)
        self.DrawLineWithMap(self.map, p_w[3], p_w[0], color, line_width)

        if vel is not None:
            p0 = [tf[0, 3], tf[1, 3]]
            p1 = [tf[0, 3] + vel[0], tf[1, 3] + vel[1]]
            self.DrawLineWithMap(self.map, p0, p1, color, line_width)

            txt = "v: {:.2f} {:.2f}".format(vel[0], vel[1])
            self.DrawText(self.map, p1, txt, color, 0.5, 1)

    def DrawPointcloud(self, pts, colors):
        us, vs = self.XysToUvs(pts[:, 0], pts[:, 1])
        masks = (us >= 0) * (vs >= 0) * \
            (us < self.base_map.shape[1]) * (vs < self.base_map.shape[0])
        us = us[masks == True]
        vs = vs[masks == True]

        self.map[vs, us, :] = np.array(colors)
        return

    def Draw(self):

        return self.map
