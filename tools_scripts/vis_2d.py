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

    def DrawLine(self, xy1, xy2, color, line_width=1):
        # print(xy1[0], xy1[1], self.XyToUv(xy1[0], xy1[1]))
        cv2.line(self.map, self.XyToUv(xy1[0], xy1[1]), self.XyToUv(
            xy2[0], xy2[1]), color, line_width)

    def DrawPolyline(self, pts, color, line_width=1):
        # pts N*2
        if (len(pts.shape) != 2) or (pts.shape[0] < 2) or (pts.shape[0] < 2):
            return
        for p, q in zip(pts[:-1], pts[1:]):
            self.DrawLine([p[0], p[1]], [q[0], q[1]],
                          color, line_width=line_width)

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
