import json
import logging
import os
from abc import abstractmethod
from collections import OrderedDict
from functools import partial

import numpy as np

from gpal_lightning import const
from gpal_lightning.neural_network.global_config import GlobalConfig
from gpal_lightning.neural_network.tasks import builder
# from gpal_lightning.neural_network.tasks.base.attributes.builder import build_attribute
from gpal_lightning.utils.datatype_convert import tensor2numpy
from gpal_lightning.utils.image_to_file import image_to_file
from gpal_lightning.utils.json_helpers.dict_to_json import dict_to_json


class BaseTask:
    """Base task should be inherit for all tasks class, it defines logics of task
    module initialization and common API for model training/validation
    """

    __version__ = "gpal_lightning"

    def __init__(self, global_config: GlobalConfig, task_config, name: str, attribute_registry=None):
        self.global_config = global_config
        self.task_config = task_config
        self.attribute_registry = attribute_registry

        self.name = name
        self.weight = self.task_config["weight"]

        self.preprocess = None
        self.postprocess = None
        self.kpi_formatter = None
        self.evaluator = None

        self.head = None
        self.train_datasets: list = []
        self.val_datasets: list = []
        self.logger = None

        self._setup()

    def _setup(self):
        self.build_task_config()
        self.build_attributes()
        self.build_callbacks()
        self.build_preprocess()
        self.build_postprocess()

        try:
            self.reset_kpi()
        except TypeError as error:
            logging.warning(
                f"Got error: {error} while building kpi_formatter and evaluator. KPI functionality is off.")

    def build_task_config(self):
        self.task_config = builder.build(
            self.global_config, self.task_config, builder.CONFIGPARSERS, self.name)
        self.task_config.name = self.name

    def build_attributes(self):
        attributes = OrderedDict()
        if hasattr(self.task_config, "attributes"):
            for name, attribute_config in self.task_config.attributes.items():
                # to remove once tasks change their attribute registry
                if self.attribute_registry:
                    attributes[name] = build_attribute(
                        self.global_config, self.task_config, attribute_config, self.attribute_registry, name
                    )

                else:
                    attributes[name] = builder.build(
                        self.global_config, self.task_config, builder.ATTRIBUTES, "", attribute_config, sub_name=name
                    )

        self.task_config.attributes = attributes

    def build_callbacks(self):
        callbacks = OrderedDict()
        if hasattr(self.task_config, "callbacks"):
            for name in self.task_config.callbacks:
                callbacks[name] = builder.build(
                    self.global_config, self.task_config, builder.CALLBACKS, self.name, sub_name=name
                )
        self.task_config.callbacks = callbacks

    def build_preprocess(self):
        self.preprocess = builder.build(
            self.global_config, self.task_config, builder.PREPROCESSES, self.name)

    def build_postprocess(self):
        self.postprocess = builder.build(
            self.global_config, self.task_config, builder.POSTPROCESSES, self.name)

    def _build_kpi_formatter(self):
        return builder.build(self.global_config, self.task_config, builder.KPI_FORMATTERS, self.name)

    def build_kpi_formatter(self):
        self.kpi_formatter = self._build_kpi_formatter()
        self.kpi_formatters = {}

    def _build_evaluator(self):
        return builder.build(
            self.global_config, self.task_config, builder.EVALUATORS, self.name, **self.task_config.Evaluator
        )

    def build_evaluator(self):
        self.evaluator = self._build_evaluator()
        self.evaluators = {}

    def build_head(self):
        self.head = builder.build(
            self.global_config, self.task_config, builder.HEADS, self.name)

    def build_datasets(self, phase):
        """This method will be called during model training/validation/inference."""
        # reset the datasets state to clean out the dataset instance
        if phase == const.PHASE_TRAINING:
            self.train_datasets = []
        elif phase == const.PHASE_VALIDATION:
            self.val_datasets = []
        elif phase == const.PHASE_INFERENCE:
            self.infer_datasets = []
        datasets_config = self.task_config.datasets[phase]
        for dataset_config in datasets_config:
            dataset = builder.build(
                self.global_config,
                self.task_config,
                builder.DATASETS,
                self.name,
                phase=phase,
                preprocess=self.preprocess,
                **dataset_config,
            )
            if phase == const.PHASE_TRAINING:
                self.train_datasets.append(dataset)
            elif phase == const.PHASE_VALIDATION:
                self.val_datasets.append(dataset)
            elif phase == const.PHASE_INFERENCE:
                self.infer_datasets.append(dataset)

    def build_logger(self):
        """
        use logger to replace Gpalnet log in the furture
        """
        self.logger = builder.build(
            self.global_config, self.task_config, builder.LOGGERS, self.name)

    def light_log(self, iteration, phase, log_writer, loss_info, **kwargs):
        """This API will only log scalar values to tensorboard"""
        self.logger.scalar_log(
            iteration, phase, log_writer, loss_info, **kwargs)

    def heavy_log(self, iteration, phase, log_writer, data, preds, masks, trues, metadata, loss_info=None):
        """This API will log both scalar values and images to tensorboard"""
        self.light_log(iteration, phase, log_writer, loss_info)
        # for idx, (image, pred, _, true, _) in enumerate(tensors):
        #     vis_pred = self.visualize(image, pred, metadata[idx], is_gt=False)
        #     vis_true = self.visualize(image, true, metadata[idx], is_gt=True)
        #     self.logger.image_log(iteration, phase, log_writer, idx, vis_pred, vis_true)

    @tensor2numpy
    def visualize(self, image: np.ndarray, vectors: np.ndarray, metadata: dict, is_gt: bool = False, scale: int = 2):
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255.0).astype(np.uint8)
        image = self._visualize(image, vectors, metadata, is_gt, scale)
        return np.transpose(image, (2, 0, 1))

    @abstractmethod
    def _visualize(self, image, vectors, metadata, is_gt, scale, **kwargs):
        raise NotImplementedError

    def visualize_badcase(self, dataset_identifier):
        pass

    @tensor2numpy
    def vector_to_json(self, vector, metadata, is_gt):
        json_dict = self.postprocess.process(
            vector,
            metadata,
            is_gt=is_gt,
        )

        return json_dict

    def vectors_to_json(self, metadata, data, dataloader_idx, vectors, is_gt):
        """This function is used to produce json files in validation_step and predict_step of Gpalnet.py.

        Args:
            metadata (dict)
            data (np.ndarray)
            dataloader_idx (int)
            vectors (np.ndarray): trues during validation_step and preds during prediction_step. Same shape as data.
            is_gt (bool): is ground truth or prediction

        Raises:
            ValueError: raises error if file extension is not recognized.
        """
        trues_or_preds = const.TRUES if is_gt else const.PREDS
        json_list, metadata_list = [], []
        for i in range(len(metadata)):
            vector = vectors[i]
            meta = metadata[i]
            if not isinstance(meta, dict):
                meta = json.loads(meta)
            metadata_list.append(meta)
            uuid = meta["uuid"]
            if "clip_id" in meta:
                clip_id = meta["clip_id"]
            else:
                clip_id = ""
            json_dict = self.vector_to_json(vector, meta, is_gt)
            json_list.append(json_dict)
            inference_root = os.path.join(
                const.JOB_EVALUATION_PATH, self.name, str(dataloader_idx))
            if self.global_config.dump_visualization:
                visualization = self._task_visualization(
                    i, data, vectors, meta, is_gt)

                image_root = os.path.join(
                    inference_root, const.IMAGES, clip_id)
                if not os.path.exists(image_root):
                    os.makedirs(image_root, exist_ok=True)
                image_to_file(
                    os.path.join(
                        image_root, f"{trues_or_preds.lower()[:-1]}_{uuid}{const.IMAGE_EXTENSION}"),
                    visualization,
                )
            for file_type, file_object in zip((trues_or_preds, const.METADATA), (json_dict, meta)):
                file_root = os.path.join(inference_root, file_type, clip_id)
                if not os.path.exists(file_root):
                    os.makedirs(file_root, exist_ok=True)
                if const.EVALUATION_FILES_EXTENSION.lower() == ".json":
                    dict_to_json(os.path.join(file_root, uuid +
                                 const.EVALUATION_FILES_EXTENSION), file_object)
                else:
                    raise ValueError(
                        f"Unrecognized EVALUATION_FILES_EXTENSION: {const.EVALUATION_FILES_EXTENSION}")
        return json_list, metadata_list

    def _task_visualization(self, curr_idx, images, vectors, metadata, is_gt):
        image = images[curr_idx]
        vector = vectors[curr_idx]
        visualization = self.visualize(image, vector, metadata, is_gt)
        return visualization

    def reset_kpi(self):
        self.build_evaluator()
        self.build_kpi_formatter()

    def dump_metric2tensorboard(self, writer, kpi, dataset_name, global_step):
        pass
