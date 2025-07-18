"""
Provides the implementation of base config parser
"""
from abc import ABC
from typing import List

from gpal_lightning.neural_network.global_config import GlobalConfig


class BaseConfigParser(ABC):
    """Base config parser that stores the values in yaml/dict into config instance."""

    def __init__(
        self,
        global_config: GlobalConfig,
        task_config: dict,
        special_keys: tuple = ("PreProcessing", "PostProcessing", "Dataset"),
    ):
        self.global_config = global_config
        self.task_config = task_config
        self._load_constants()
        for key, val in task_config.items():
            if key in special_keys:
                continue
            setattr(self, key, val)

        self._process_global()

        for key in special_keys:
            if key in task_config:
                parsed_config = getattr(self, "parse_" + key.lower())(task_config[key])
                for _key, _val in parsed_config.items():
                    setattr(self, _key, _val)

        self.post_parsing_check()

    def _load_constants(self) -> None:
        self.image_time = 0
        if hasattr(self, "global_config") and hasattr(self.global_config, "image_time"):
            self.image_time = self.global_config.image_time
        self.weight: float = 0.0
        self.backbone_feat_id: List[int] = [-1]
        self.group_feat_id: List[int] = []
        self.input_feat_id: List[int] = []
        self.frequency: float = 0.0
        self.group: str = ""
        self.neck: str = ""
        self.pre_task_neck: str = ""
        self.backbone: str = "backbone0"
        self.grad_bounds: float = 0.0
        self.Augmentations: dict = {}
        self.Evaluator: dict = {}
        self.sequential_data: bool = False
        self.amp_scaler: dict = {}
        self.head_reuse = False

    def _process_global(self) -> None:
        pass

    def post_parsing_check(self) -> None:
        """This method is used to implement the logic for config values validatoin,
        Please implement the config assertions here so jobs with invalid config will be
        killed at the begining.
        """
        assert self.weight != 0, "Task weight cannot be 0"
        if isinstance(self.backbone_feat_id, int):
            self.backbone_feat_id = [self.backbone_feat_id]
        assert isinstance(self.backbone_feat_id, list), "backbone_feat_id should be list type"

        if isinstance(self.group_feat_id, int):
            self.group_feat_id = [self.group_feat_id]
        assert isinstance(self.group_feat_id, list), "group_feat_id should be list type"

        self.input_feat_id = self.backbone_feat_id
        if self.group_feat_id:
            self.input_feat_id = self.group_feat_id

        assert self.grad_bounds != 0, "grad bounds cannot be 0"
        if not self.amp_scaler:
            self.amp_scaler = {"type": "AMPScaler"}

    @staticmethod
    def parse_dataset(dataset_config) -> dict:
        """a hook to provide config postprocess for dataset section"""
        return dataset_config

    @staticmethod
    def parse_preprocessing(preprocessing_config) -> dict:
        """a hook to provide config postprocess for preprocess section"""
        return preprocessing_config

    @staticmethod
    def parse_head(head_config) -> dict:
        """a hook to provide config postprocess for head section"""
        return head_config

    @staticmethod
    def parse_postprocessing(postprocessing_config) -> dict:
        """a hook to provide config postprocess for postprocess section"""
        return postprocessing_config

    def __str__(self):
        return "\n".join(self.__dict__.keys())

    def __repr__(self):
        return self.__str__()
