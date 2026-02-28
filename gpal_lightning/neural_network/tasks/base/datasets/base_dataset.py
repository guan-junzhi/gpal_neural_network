import json
import logging
import random
from abc import ABC
from typing import Union
import copy
import pickle
import os

import torch
from torch import distributed
from torch.utils.data import Dataset

from gpal_lightning import const
from gpal_lightning.data.exceptions import DataloaderException, DataSkipException
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.utils.json_helpers.json_serializer import JsonSerializer
from gpal_lightning.utils.datatype_convert import _data2tensor
from gpal_lightning.utils.data_buffer import FastLoaderBuffer
import cv2
import numpy as np

class BaseDataset(Dataset, ABC):
    """BaseDataset is a iterable-style dataset. (https://pytorch.org/docs/stable/data.html)

    #TODO for fast implementation, current dataset not follow those data feteching pipeline

    Workflow:
    1. fetch a datablob given idx.
    2. pass datablob to _data_preprocess, this method will call task.preprocess with below
    """

    def __init__(
        self,
        global_config: GlobalConfig,
        task_config,
        preprocess,
        dataset_name: str,
        phase: str,
        root_dir: str,
        camera_name: str = None,
        shuffle: bool = True,
        shuffle_seed: int = 0,
        sql_filter: str = "",
        pseudo_labels_path: Union[str, list] = None,
        worker: int = 0,
        ratio: float = 0.0,
        disable_tensor_convert: bool = False,
        build_dataset: bool = True,
        fast_buffer_path: str = ""
    ):
        """
        Args:
            global_config: object from CONSTANT class
            task_config: object from task ConfigParser class
            preprocess: preprocess object of current task
            dataset_name: a str
            # TODO: support this feature
            camera_name: used to filter data from data source
            root_dir: data prefix
            phase: it can and only can be from "training" and "validation"
            worker: number of workers for the dataloader of this dataset, this value is calculated automatically.
            ratio: accumulated sampling ratio of this dataset, this can be calculated automatically based on the
            length of dataset, if you want to assign the ratio manully, please make sure the ratio of each dataset
            is in ascending order, and the ratio of last dataset is 1.0
        """
        self.global_config = global_config
        self.task_config = task_config
        self.task_name = task_config.name
        self.preprocess = preprocess
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.camera_name = camera_name
        self.disable_tensor_convert = disable_tensor_convert
        self.pseudo_labels_path = pseudo_labels_path

        if phase not in [const.PHASE_TRAINING, const.PHASE_VALIDATION, const.PHASE_INFERENCE]:
            raise KeyError(
                f"phase should be in {const.PHASE_TRAINING, const.PHASE_VALIDATION, const.PHASE_INFERENCE}"
                f" but got {phase}"
            )
        self.phase = phase

        self.sql_filter = sql_filter
        self._worker = worker
        self._ratio = ratio
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.data_upsample_uuids: list = []
        self.skipped_uuids: set = set()
        self.pseudo_labels: dict = {}

        try:
            self.global_rank = distributed.get_rank()
        except (RuntimeError, AssertionError):
            self.global_rank = -1

        if build_dataset:
            self._build_world_data_list()
            self._build_dataset()
            self._dataset_logging()

        if fast_buffer_path != '':
            os.makedirs(fast_buffer_path, exist_ok=True)
            self.buffer = FastLoaderBuffer(fast_buffer_path)
        else:
            self.buffer = None
        self.buffer_slice_write_cache = {}
        self.buffer_slice_read_cache = {}

    @property
    def worker(self):
        return self._worker

    @worker.setter
    def worker(self, worker):
        self._worker = worker

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self._ratio = ratio

    def _dataset_logging(self):
        if self.camera_name:
            logging.info("Logging for camera id : %s", self.camera_name)
        logging.info("%s: %s has %s %s labels", self.task_name,
                     self.dataset_name, len(self.dataset), self.phase)

    def _build_world_data_list(self):
        """build self.world_data_list"""
        raise NotImplementedError

    def _distribute_data(self):
        try:
            rank_curr = distributed.get_rank()
            world_size = distributed.get_world_size()
        except (RuntimeError, AssertionError):
            rank_curr = 0
            world_size = 1
        if self.phase in [const.PHASE_TRAINING]:
            if self.shuffle:
                random.Random(self.shuffle_seed).shuffle(self.world_data_list)

        size_per_rank = len(self.world_data_list) // world_size
        size_remain = 0
        if self.global_config.debug:
            start_index = rank_curr * size_per_rank
            end_index = start_index + 100 if size_per_rank > 100 else size_per_rank
        elif size_remain == 0:
            start_index = rank_curr * size_per_rank
            end_index = start_index + size_per_rank
        elif rank_curr < size_remain:
            start_index = rank_curr * (size_per_rank + 1)
            end_index = start_index + (size_per_rank + 1)
        else:
            start_index = size_remain * \
                (size_per_rank + 1) + (rank_curr - size_remain) * size_per_rank
            end_index = start_index + size_per_rank
        dataset = pickle.loads(pickle.dumps(
            self.world_data_list[start_index: end_index]))
        logging.info("Distirbuted dataset completed: rank %d, world_size %d, data %d.",
                     rank_curr, world_size, len(dataset))
        # reset world_data_list here to release memory
        self.world_data_list = None
        return dataset

    def _build_dataset(self):
        logging.info("Building distirbuted data list.")
        self.dataset = self._distribute_data()
        logging.info(f"Built distirbuted data list {len(self.dataset)}")
        # TODO: support upsample, skip and other features in furture

    def __len__(self):
        return len(self.dataset)

    def _fetch_data(self, idx):
        data_blob = self.dataset[idx]
        return data_blob

    def _data_preprocess(self, data_blob):
        raise NotImplementedError

    def __getitem__(self, idx):
        """A for loop to queue/preprocess label and image.
        Return us the image, label and label mask. Task owner should not
        override this method.
        """
        resample = False
        for _ in range(self.global_config.max_data_fetch_iteration):
            try:
                data_blob = self._fetch_data(idx)
                data_blob = self._data_preprocess(data_blob)

                metadata = data_blob[const.METADATA]
                metadata["task_name"] = self.task_name
                augmentation = data_blob.pop(const.AUGMENTATIONS, None)
                if augmentation:
                    augmentation.pop("affine", None)
                    metadata[const.AUGMENTATIONS] = augmentation
                processed_data_blob = {}
                for key, val in data_blob.items():
                    if key == const.METADATA:
                        processed_data_blob[key] = json.dumps(
                            metadata, cls=JsonSerializer)
                    else:
                        processed_data_blob[key] = _data2tensor(val)
                return processed_data_blob
            except DataloaderException as error:
                logging.error(
                    "DataloaderException: %s for %s task", error, self.task_name, exc_info=self.global_config.debug
                )
                break
            except DataSkipException:
                if self.phase != const.PHASE_VALIDATION:
                    resample = True
            except Exception as error:
                if self.phase != const.PHASE_VALIDATION or not self.global_config.dump_json_during_validation:
                    resample = True
                if not self.global_config.disable_resample_warning:
                    logging.warning(
                        "Resample warning: %s for %s task", error, self.task_name, exc_info=self.global_config.debug
                    )
            if resample:
                idx = random.randint(0, len(self) - 1)
            else:
                return None

        raise DataloaderException

    def _image_cache(self, key, img, pre_resize, quality):
        if (self.buffer is None):
            return False
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        h = int(img.shape[0]).to_bytes(4, byteorder='little', signed=False)
        w = int(img.shape[1]).to_bytes(4, byteorder='little', signed=False)
        img = cv2.resize(img, pre_resize)
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if result:
            return self.buffer.Cache(key, h + w + encimg.tobytes())
        else:
            return False

    def _image_buffer_access(self, key):
        if (self.buffer is None):
            return None, None
        try:
            x = self.buffer[key]
            h = int().from_bytes(x[:4], byteorder='little', signed=False)
            w = int().from_bytes(x[4:8], byteorder='little', signed=False)
            return cv2.imdecode(np.frombuffer(x[8:], dtype=np.uint8), cv2.IMREAD_COLOR), [h, w]
        except:
            return None, None

    def _slice_image_cache(self, slice_key, view_key, img, pre_resize, quality):
        if (self.buffer is None):
            return False

        if slice_key not in self.buffer_slice_write_cache:
            for k, v in self.buffer_slice_write_cache.items():
                ret = self.buffer.Cache(k, pickle.dumps(v))
                if not ret:
                    print(f"self.buffer.Cache {k} faild")

            self.buffer_slice_write_cache = {slice_key: {}}
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        h = int(img.shape[0]).to_bytes(4, byteorder='little', signed=False)
        w = int(img.shape[1]).to_bytes(4, byteorder='little', signed=False)
        img = cv2.resize(img, pre_resize)
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        self.buffer_slice_write_cache[slice_key][view_key] = h + w + encimg.tobytes()
        return True

    def _slice_image_buffer_access(self, slice_key, view_key):
        if (self.buffer is None):
            return None, None
        try:
        # if True:
            if slice_key not in self.buffer_slice_read_cache:
                self.buffer_slice_read_cache = {slice_key: pickle.loads(self.buffer[slice_key])}
           
            x = self.buffer_slice_read_cache[slice_key][view_key]
            h = int().from_bytes(x[:4], byteorder='little', signed=False)
            w = int().from_bytes(x[4:8], byteorder='little', signed=False)
            return cv2.imdecode(np.frombuffer(x[8:], dtype=np.uint8), cv2.IMREAD_COLOR), [h, w]
        except Exception as e:
            # print(f"_slice_image_buffer_access failed {e}")
            return None, None

if __name__ == "__main__":
    buffer = FastLoaderBuffer("DRIVING_BEV_DYN_data_buf_exp2")
    img = cv2.imread(
        "/data/dp_group/process-prod-bucket/data_collect/EKART_ID4001_2025-05-15-09-25-16/2025-05-15-09-39-31.17/img_front_120/1747274141.024808.jpg")
    compress_rate = 100
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_rate]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    print(result, type(encimg), encimg.shape, type(encimg.tobytes()))
    import numpy as np

    img_dec = cv2.imdecode(np.frombuffer(
        encimg.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("1747274141.024808.jpg",
                np.concatenate([img, img_dec], axis=1))
    print(img_dec.shape)

    # if result:
    #     return self.buffer.Cache(key, encimg.tobytes())
