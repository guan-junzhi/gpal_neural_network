import cv2
import numpy as np
import random


def letterbox_image(image_src, dst_size, K=None, pad_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    """
    :param image_src:       original image（numpy）
    :param dst_size:        target image size（h，w）
    :param pad_color:       padding color, default is black
    :param interpolation:   inter mode, default is linear inter
    :return:
    """
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if image_src.shape[0:2] != (pad_w, pad_h):
        image_dst = cv2.resize(image_src, (pad_w, pad_h), interpolation=interpolation)
    else:
        image_dst = image_src

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    # add border
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    x_offset, y_offset = max(left, right), max(top, down)

    # pad_src_h, pad_src_w = int(round(dst_h / scale)), int(round(dst_w / scale))
    # src_top = int((pad_src_h - src_h) / 2)
    # src_down = int((pad_src_h - src_h + 1) / 2)
    # src_left = int((pad_src_w - src_w) / 2)
    # src_right = int((pad_src_w - src_w + 1) / 2)
    # src_x_offset, src_y_offset = max(src_left, src_right), max(src_top, src_down)

    if K is not None:
        K = np.array([
            [scale, 0., x_offset],
            [0., scale, y_offset],
            [0., 0., 1.]], dtype=K.dtype) @ K
    return image_dst, K


def random_scale_and_translate(image, in_shape, K=None, scale=0.1, offset=0.1):
    """
    对图像和内参矩阵 K 进行缩放和平移变换

    参数:
        image: 输入图像 (H, W, C)
        K: 相机内参矩阵 (3x3)
        scale: 缩放比例
        offset: 平移比例

    返回:
        transformed_image: 变换后的图像
        K_new: 变换后的内参矩阵
    """
    dst_h, dst_w = in_shape
    h, w = image.shape[:2]
    sx = 1 + random.uniform(-scale, scale)
    sy = 1 + random.uniform(-scale, scale)
    tx = random.uniform(-offset, offset)
    ty = random.uniform(-offset * 0.1, offset) #涉及到h方向的裁剪，所以ty向上的变化尺度小一点

    dx = int(tx * w)
    dy = int(ty * h)

    new_w = int(w * sx)
    new_h = int(h * sy)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 平移图像（用边缘填充）
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(image, M, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    if K is not None:
        K = np.array([
            [sx, 0., dx],
            [0., sy, dy],
            [0., 0., 1.]], dtype=K.dtype) @ K

    return image, K
