import numpy as np
import cv2
import random
import torch
from typing import Any, Dict, Optional, Sequence, Tuple, Union

Number = Union[int, float]


class CutImageUpper(object):
    """Remove upper image, because there are no target feature be needed
    Args:
        start_h (int): 0~start_h will be removed

    Required Keys:

        - images (np.uint8)/(np.float32)
    """

    def __init__(self, start_h=112, just_update_meta=False):
        self.start_h = start_h
        # onnx使用原图输入时just_update_meta=True
        self.just_update_meta = just_update_meta

    def __call__(self, data) -> dict:
        assert 'image' in data, '`image` is not found in results'
        if not self.just_update_meta:
            data['image'] = {k: data['image'][k][self.start_h:, :, :]
                            for k in data['image']}
        data['calib']['ists'][:, 1, 2] -= self.start_h
        data['meta']['cut_h'] = True
        data['meta']['cut_h_value'] = self.start_h
        # if 'seg' in data['annot']:
        #     data['annot']['seg'] = data['annot']['seg'][self.start_h:, :]

        return data


class Normalize(object):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value
    """

    def __init__(
            self,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375)
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, data: dict) -> dict:
        # {k: print("s",data['image'][k].dtype) for k in data['image']}

        assert 'image' in data, '`image` is not found in results'
        # data['image'] = (data['image'] - self.mean) / self.std
        data['image'] = {k: (data['image'][k] - self.mean) /
                         self.std for k in data['image']}

        return data


class ToTensor(object):
    def __init__(self,
                 toTorchKeys=['images_src', 'images', 'exts', 'ists', 'ists_norm', 'dists']):
        self.toTorchKeys = toTorchKeys

    def __call__(self, data: dict) -> dict:
        for key in data:
            if key in self.toTorchKeys:
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key].astype(np.float32))
                elif isinstance(data[key], list):
                    data[key] = torch.from_numpy(
                        np.array(data[key]).astype(np.float32))
                else:
                    raise TypeError

                if key in ['images', 'images_src'] and isinstance(data[key], torch.Tensor):
                    if len(data[key].shape) == 3:
                        data[key] = data[key].permute((2, 0, 1)).contiguous()
                    elif len(data[key].shape) == 4:
                        data[key] = data[key].permute(
                            (0, 3, 1, 2)).contiguous()
        return data


class MultiViewPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from RGB to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to RGB
    7. random contrast (mode 1)

    Required Keys:

    - images (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.8, 1.2),
                 saturation_range: Sequence[Number] = (0.8, 1.2),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def _random_flags(self) -> Sequence[Number]:
        mode = random.randint(0, 2)
        brightness_flag = random.randint(0, 2)
        contrast_flag = random.randint(0, 2)
        saturation_flag = random.randint(0, 2)
        hue_flag = random.randint(0, 2)
        delta_value = random.uniform(-self.brightness_delta,
                                     self.brightness_delta)
        alpha_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        hue_value = random.uniform(-self.hue_delta, self.hue_delta)

        return (mode, brightness_flag, contrast_flag, saturation_flag,
                hue_flag, delta_value, alpha_value,
                saturation_value, hue_value)

    def __call__(self, data: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert 'image' in data, '`images` is not found in results'
        imgs = data['image']
        # assert len(imgs.shape) == 4

        (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
         delta_value, alpha_value, saturation_value, hue_value) = self._random_flags()

        for cam_name in imgs:

            img = imgs[cam_name]
            img = img.astype(np.float32)
            # random brightness
            if brightness_flag:
                img += delta_value

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            if mode == 1:
                if contrast_flag:
                    img *= alpha_value
            # img = np.clip(img, 0, 255)

            # convert color from RGB to HSV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # random saturation
            if saturation_flag:
                img[..., 1] *= saturation_value
                # For image(type=float32), after convert rgb to hsv by opencv,
                # valid saturation value range is [0, 1]
                if saturation_value > 1:
                    img[..., 1] = img[..., 1].clip(0, 1)

            # random hue
            if hue_flag:
                img[..., 0] += hue_value
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to RGB
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

            # random contrast
            if mode == 0:
                if contrast_flag:
                    img *= alpha_value
            img = np.clip(img, 0, 255)

            # 应对mifa和吉祥车相机对比度低
            if random.random() < 0.3:
                mean_value = np.mean(img)
                scale = random.uniform(0.3, 0.5)
                ideal_offset = mean_value - mean_value * scale
                random_offset = random.uniform(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=scale, beta=ideal_offset + random_offset)

            data['image'][cam_name] = img

        return data


class MultiViewRandomCutOut(object):
    """CutOut operation for segmentation task.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        prob: Cutout probability.
        n_holes: Number of regions to be dropped. If it is given as a list,
        number of holes will be randomly selected from the closed interval
            [`n_holes[0]`, `n_holes[1]`].
        cutout_shape: The candidate shape of dropped regions. It can be
            `tuple[int, int]` to use a fixed cutout shape, or
            `list[tuple[int, int]]` to randomly choose shape from the list.
        cutout_ratio: The candidate ratio of dropped regions. It can be
            `tuple[float, float]` to use a fixed ratio or
            `list[tuple[float, float]]` to randomly choose ratio from the list.
            Please note that `cutout_shape` and `cutout_ratio` cannot be both
            given at the same time.
        fill_in: The value of pixel to fill in the dropped regions. Default is
            (0, 0, 0).
        seg_fill_in: The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default is None.
    """

    def __init__(
            self,
            prob: float,
            n_holes: Union[int, Tuple[int, int]],
            cutout_shape: Optional[
                Union[Tuple[int, int], Tuple[Tuple[int, int], ...]]
            ] = None,
            cutout_ratio: Optional[
                Union[Tuple[int, int], Tuple[Tuple[int, int], ...]]
            ] = None,
            fill_in: Tuple[float, float, float] = (0, 0, 0),
            seg_fill_in: Optional[int] = None,
    ):
        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (
            cutout_ratio is None
        ), "Either cutout_shape or cutout_ratio should be specified."
        assert isinstance(cutout_shape, (list, tuple)) or isinstance(
            cutout_ratio, (list, tuple)
        )
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (
                isinstance(seg_fill_in, int)
                and 0 <= seg_fill_in
                and seg_fill_in <= 255
            )
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, data):
        """Call function to drop some regions of image."""
        assert 'image' in data, '`image` is not found in results'
        # cutout = True if np.random.rand() < self.prob else False
        # if cutout is False:
        #     return data

        imgs = data["image"]

        num_cam = 1
        # if len(imgs.shape) == 4:
        #     num_cam, h, w, c = imgs.shape
        # elif len(imgs.shape) == 3:
        #     h, w, c = imgs.shape

        # for cam_idx in range(num_cam):
        for cam_name in imgs:
            h, w, c = imgs[cam_name].shape
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)

            # print(cam_name, n_holes)
            # exit(1)
            for _ in range(n_holes):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)
                index = np.random.randint(0, len(self.candidates))

                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                data["image"][cam_name][y1:y2, x1:x2] = self.fill_in
                # if self.seg_fill_in is not None:
                #     if "seg" in data['annot']:
                #         data['annot']['seg'][y1:y2, x1:x2] = self.seg_fill_in
        # exit(1)
        return data
