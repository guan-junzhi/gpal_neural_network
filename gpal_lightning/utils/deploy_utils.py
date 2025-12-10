import numpy as np
import cv2

def DistGridMap(src_w, src_h, dist, intrins, tgt_w, tgt_h, top_crop_len, top_crop_bgn, norm=True):
    if norm:
        ws = np.linspace(-1.0, 1.0, src_w,
                         endpoint=True)[np.newaxis, :, np.newaxis].repeat(src_h, 0)
        hs = np.linspace(-1.0, 1.0, src_h,
                         endpoint=True)[:, np.newaxis, np.newaxis].repeat(src_w, 1)
    else:
        ws = np.linspace(0.0, src_w-1.0, src_w,
                         endpoint=True)[np.newaxis, :, np.newaxis].repeat(src_h, 0)
        hs = np.linspace(0.0, src_h-1.0, src_h,
                         endpoint=True)[:, np.newaxis, np.newaxis].repeat(src_w, 1)
    # cv2.imwrite("ws.jpg", (ws * 127+128).astype(np.uint8))
    # cv2.imwrite("hs.jpg", (hs * 127+128).astype(np.uint8))
    src_map = np.concatenate([ws, hs], axis=-1)
    target_map = cv2.undistort(
        src=src_map, cameraMatrix=intrins, distCoeffs=dist, newCameraMatrix=intrins)
    target_map = cv2.resize(target_map, [tgt_w, tgt_h])
    target_map = target_map[top_crop_bgn:top_crop_len+top_crop_bgn, :]
    return target_map

def bgr_to_nv12_split(img_bgr):
    img_bgr = img_bgr.squeeze(axis=0).astype(np.uint8)
    # 转换为YUV420 (I420)
    yuv_i420 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = img_bgr.shape[:2]

    # 提取Y平面
    y_plane = yuv_i420[:h, :]

    uv_start = h
    uv_height = h // 4
    u_plane = yuv_i420[uv_start:uv_start+uv_height, :]
    v_plane = yuv_i420[uv_start+uv_height:uv_start+2*uv_height, :]

    uv_interleaved = np.zeros((h//2 * w//2 * 2), dtype=np.uint8)
    uv_interleaved[0::2] = u_plane.flatten()
    uv_interleaved[1::2] = v_plane.flatten()

    y_plane = y_plane.reshape(1, h, w, 1)  # .transpose(0, 3, 1, 2)
    uv_interleaved = uv_interleaved.reshape(
        1, h//2, w//2, 2)  # .transpose(0, 3, 1, 2)
    return y_plane, uv_interleaved

def rgb_to_nv12_split(img_rgb):
    img_rgb = img_rgb.squeeze(axis=0).astype(np.uint8)
    # 转换为YUV420 (I420)
    yuv_i420 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV_I420)
    h, w = img_rgb.shape[:2]
    
    # 提取Y平面
    y_plane = yuv_i420[:h, :]
    
    # 提取U和V平面（注意I420的UV排列为UUUUVVVV）
    uv_start = h
    uv_height = h // 4  # I420的U/V平面高度为h/4
    u_plane = yuv_i420[uv_start:uv_start+uv_height, :]
    v_plane = yuv_i420[uv_start+uv_height:uv_start+2*uv_height, :]
    
    # 将U和V平面合并成交替的UV平面（NV12）
    # 需要将U和V从 (h//4, w//2) 上采样到 (h//2, w)
    uv_interleaved = np.zeros((h//2 * w//2 * 2), dtype=np.uint8)
    uv_interleaved[0::2] = u_plane.flatten()
    uv_interleaved[1::2] = v_plane.flatten()
    
    # 组合NV12数据
    # nv12 = np.vstack([y_plane, uv_interleaved])
    return  y_plane.reshape(1, h, w, 1), uv_interleaved.reshape(1, h//2, w//2, 2)
