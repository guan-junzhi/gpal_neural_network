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

    def DrawPolyline(self, pts, color, line_width=1, line_type='solid', dash_length=10):
        """
        绘制多边形线条，可以是实线或虚线
        
        参数:
            pts: 点集，形状为N*2
            color: 线条颜色
            line_width: 线条宽度
            line_type: 线条类型，'solid' 为实线，'dashed' 为虚线
            dash_length: 虚线中每段实线的长度，仅在line_type为'dashed'时有效
        """
        # 检查输入点集的有效性
        if len(pts.shape) != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            return
        
        # 依次绘制每一段线
        for p, q in zip(pts[:-1], pts[1:]):
            self.DrawLine(
                [p[0], p[1]], 
                [q[0], q[1]], 
                color, 
                line_width=line_width,
                line_type=line_type,
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
