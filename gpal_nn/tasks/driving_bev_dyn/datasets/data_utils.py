
import cv2
import random

import numpy as np


def aug_image(image):
    randomn = np.random.randint(0,5)
    if randomn == 0:
        augment_hsv(image,
                    np.random.randint(0,100)/100,
                    np.random.randint(0,10)/10,
                    np.random.randint(0,10)/10)
    elif randomn ==1:
        image = clahe_Img(image,np.random.randint(3,10))
    elif randomn ==2:
        image = spiced_salt_noise(image,np.random.randint(1,10)/10)
    else:
        pass
    return image

def aug_radar_data_operation(points, points_former, rand_seed):
    if rand_seed == 1:  # 0.1 随机丢弃0.05的数据
        mask  = np.random.random(len(points)) < 0.1
        points = points[~mask]
        mask  = np.random.random(len(points_former)) < 0.1
        points_former = points_former[~mask]
    elif rand_seed == 2:
        pass
    else:
        pass
    
    return points, points_former

def augment_hsv(im, hgain=0.015, sgain=0.1, vgain=0.1):
    """
    参数：
    img: 待处理图片  BGR
    hgain: h通道色域参数 用于生成新的h通道，默认为0.5
    sgain: h通道色域参数 用于生成新的s通道，默认为0.5
    vgain: h通道色域参数 用于生成新的v通道，默认为0.5
    """
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数  用于生成新的hsv通道
        r = np.array([1,np.random.random()+1,np.random.random()/2+0.5])

        # 图像的通道拆分 h s v
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 生成新的h通道
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 生成新的s通道
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 生成新的v通道

        # 图像通道合并 img_hsv=h+s+v  随机调整hsv之后重新组合hsv通道
        # cv2.LUT(hue, lut_hue)   通道色域变换 输入变换前通道hue 和变换后通道lut_hue
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))

        # no return needed  dst:输出图像
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)

def clahe_Img(image,ksize):
    """
    :param path: 图像路径
    :param ksize: 用于直方图均衡化的网格大小，默认为8
    :return: clahe之后的图像
    """
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ksize,ksize))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b, g, r])
    return image

def blur(img,scale):
    img = cv2.blur(img,(scale,scale)) # scale越大越模糊
    return img

# 添加椒盐噪声
def spiced_salt_noise(img,prob):
    output = np.zeros(img.shape,np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0 # 椒盐噪声由纯黑和纯白的像素点随机组成
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output
