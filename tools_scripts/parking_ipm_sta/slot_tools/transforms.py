import numpy as np
import random
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
class Compose_img(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        #size = [random.randint(self.min_size, self.max_size)]
        # size = [896,896]
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        # image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        # target = F.resize(target, size)
        # target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # image = pad_if_smaller(image, self.size)
        # target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ToTensor_img(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image = F.normalize(image, mean=self.mean,std=self.std)
        return image

class RGBToBGR(object):
    def __call__(self, image, target):
        image = np.array(image)
        image = image[:,:,::-1]
        image = np.ascontiguousarray(image)
        return image, target
    
class Resize(object):
    def __init__(self, resize_w, resize_h):
        self.resize_w = resize_w
        self.resize_h = resize_h

    def __call__(self, image, target):
        #size = [random.randint(self.min_size, self.max_size)]
        size = (self.resize_w,self.resize_h)
        image = cv2.resize(image, size, interpolation = cv2.INTER_NEAREST)
        target = cv2.resize(target, size, interpolation = cv2.INTER_NEAREST)

        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        # image = F.resize(image, size, interpolation=T.InterpolationMode.NEAREST)
        # image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        # target = F.resize(target, size)
        # target = F.resize(target, size)
        return image, target