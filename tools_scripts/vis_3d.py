import open3d as o3d
import numpy as np
import time
import open3d.visualization.rendering as rendering


class LocalViz3D:
    def __init__(self, origin_coord_size=-1):
        self.to_show = []
        if origin_coord_size > 0:
            FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=origin_coord_size, origin=[0, 0, 0])
            self.to_show.append(FOR1)

    def AddPointcloud(self, points, color=None):

        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(points[:, :3])
        if color is not None:
            color = np.array(color)
            if len(color.shape) == 1:
                color = np.expand_dims(color, 0).repeat(points.shape[0], 0)
            # print(color.shape)
            # print(points.shape)
            pts.colors = o3d.utility.Vector3dVector(color[:, :3])
        self.to_show.append(pts)

    def AddPointcloudOct(self, points, color=None):
        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(points[:, :3])
        if color is not None:
            color = np.array(color)
            if len(color.shape) == 1:
                color = np.expand_dims(color, 0).repeat(points.shape[0], 0)
            # print(color.shape)
            # print(points.shape)
            pts.colors = o3d.utility.Vector3dVector(color[:, :3])

        octree = o3d.geometry.Octree(max_depth=8)
        octree.convert_from_point_cloud(pts, size_expand=0.4)

        self.to_show.append(octree)

    def AddCam(self, tf=np.identity(4), color=[1, 0, 0], f=1500, w=1920, h=1080, scale=1e-4):
        # tf = odo_base_inv * odos.GetPose(ts)
        w_2 = w / 2
        h_2 = h / 2
        p = np.matrix([[0, 0, 0, 1 / scale],
                       [w_2, h_2, f, 1 / scale],
                       [-w_2, h_2, f, 1 / scale],
                       [-w_2, -h_2, f, 1 / scale],
                       [w_2, -h_2, f, 1 / scale]]) * scale
        # print(tf.shape, p.shape, p, tf)
        # print(p[0])

        p_w = np.array(np.matrix(tf) * p.T)
        p_w = p_w[:3]
        # print(p_w[0])
        # p_w[2, :] *= 10.0
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(p_w.T),
            lines=o3d.utility.Vector2iVector(
                [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
        )
        colors = [color] * 8
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.to_show.append(line_set)

    def AddCoord(self, tf=np.identity(4), size=0.1):
        # tf = odo_base_inv * odos.GetPose(ts)

        p = np.matrix([[0, 0, 0, 1],
                       [size, 0, 0, 1],
                       [0, size, 0, 1],
                       [0, 0, size, 1]])

        p_w = np.array(np.matrix(tf) * p.T)
        p_w = p_w[:3]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(p_w.T),
            lines=o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        )
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.to_show.append(line_set)

    # lines n * 2 * 3
    def AddLines(self, lines, color=[0, 1.0, 0]):
        # tf = odo_base_inv * odos.GetPose(ts)
        lines = np.array(lines)
        line_num = lines.shape[0]
        # print(lines.shape)
        pts = np.concatenate([lines[:, 0, :], lines[:, 1, :]], 0)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(
                [[i, i + line_num] for i in range(line_num)])
        )

        color = np.array(color)
        if len(color.shape) == 1:
            color = np.expand_dims(color, 0).repeat(line_num, 0)
        line_set.colors = o3d.utility.Vector3dVector(color)
        self.to_show.append(line_set)

    def AddBbox(self, tf=np.identity(4), size=np.array([1.0, 1.0, 1.0]), colorA=[1, 0, 0], colorB=[0, 1, 0]):
        # tf = odo_base_inv * odos.GetPose(ts)
        hsx = size[0] * 0.5
        hsy = size[1] * 0.5
        hsz = size[2] * 0.5

        p = np.matrix([[-hsx, -hsy, -hsz, 1],
                       [-hsx, hsy, -hsz, 1],
                       [hsx, hsy, -hsz, 1],
                       [hsx, -hsy, -hsz, 1],

                       [-hsx, -hsy, hsz, 1],
                       [-hsx, hsy, hsz, 1],
                       [hsx, hsy, hsz, 1],
                       [hsx, -hsy, hsz, 1],
                       ])

        p_w = np.array(np.matrix(tf) * p.T)
        p_w = p_w[:3]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(p_w.T),
            lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0],
                                              [4, 5], [5, 6], [6, 7], [7, 4],
                                              [0, 4], [1, 5], [2, 6], [3, 7]
                                              ])
        )
        colors = []
        for i in range(12):
            if i in [0, 1, 3, 4, 5, 7, 8, 9]:
                colors.append(colorA)
            else:
                colors.append(colorB)

        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.to_show.append(line_set)

    def Show(self, config=None, dump_config=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # vis.add_geometry(pts)
        for ele in self.to_show:
            # print(ele)
            vis.add_geometry(ele)

        vis.get_render_option().point_size = 1
        # images = []
        if config is not None:
            pinholeCamera = o3d.io.read_pinhole_camera_parameters(config)
            ctr = vis.get_view_control()

            ctr.convert_from_pinhole_camera_parameters(pinholeCamera)
        vis.poll_events()
        vis.update_renderer()
        # img = vis.capture_screen_float_buffer()
        # img_np = np.asarray(img)
        # images.append(img_np)
        # time.sleep(5) # Set frame Time
        vis.run()

        vis.destroy_window()

        if dump_config:
            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters("params.json", params)

    def DumpImg(self, render_config, f_name):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for ele in self.to_show:
            vis.add_geometry(ele)
        vis.get_render_option().point_size = 1
        pinholeCamera = o3d.io.read_pinhole_camera_parameters(render_config)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(pinholeCamera)
        vis.poll_events()
        time.sleep(0.5)
        # vis.update_renderer()
        img = vis.capture_screen_float_buffer()
        img_np = (np.asarray(img)*255).astype(np.uint8)[..., ::-1]

        # print(np.max(img_np), img_np.dtype)
        import cv2
        cv2.imwrite(f_name, img_np)

    # open3d version >= 0.14.1
    def DumpImgProject(self, render_config, f_name):
        pinholeCamera = o3d.io.read_pinhole_camera_parameters(render_config)
        render = rendering.OffscreenRenderer(
            pinholeCamera.intrinsic.width, pinholeCamera.intrinsic.height)

        yellow = rendering.MaterialRecord()
        yellow.shader = "unlitLine"
        yellow.line_width = 2.0
        yellow.point_size = 0.1
        for ele_i, ele in enumerate(self.to_show):
            render.scene.add_geometry(str(ele_i), ele, yellow, False)
        render.setup_camera(pinholeCamera.intrinsic, pinholeCamera.extrinsic)
        # render.point_size(0.1)
        img = render.render_to_image()
        # print("Saving image at ", f_name)
        o3d.io.write_image(f_name, img)


if __name__ == "__main__":

    v3d = LocalViz3D(1)
    a = np.random.random([1000, 3])
    b = np.random.random([1000, 3])
    v3d.AddPointcloud(a, b)
    v3d.AddCam(tf=np.identity(4))
    v3d.Show()
