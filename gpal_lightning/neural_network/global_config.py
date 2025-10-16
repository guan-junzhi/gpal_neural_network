"""
Implementation of global config class
"""

import logging
from copy import deepcopy
from typing import List, Union

# from zpilot_lightning import const


class GlobalConfig:
    """Global config contains the parameters related to model training or some variables shared
    for all tasks"""

    def __init__(
        self,
        config: dict,
        special_keys: tuple = ("Image", "Global", "Camera", "Train"),
        remap_keys: tuple = (
            ("Train/optimizer/grad_bounds", "Tasks*/grad_bounds", "REPLACE"),
            ("Train/lr_scheduler/iterations",
             "Train/lr_scheduler/T_max", "REPLACE"),
            ("Train/lr_scheduler/warmup_epoch",
             "Train/lr_scheduler/warmup_iterations", "REPLACE"),
            ("Train/validate_every", "save_every_n_iterations", "COPY"),
        ),
    ):
        """

        Args:
            config: dictionary contains the global configs
            special_keys: in the above config, values under special_keys will be stored,
            those keys are required in the config.yaml.
            directly in the global config instance. Other keys will be stored as a dictionary
            in the instance. This is for backward compatibility
        Examples:
            config = {
                "Image": {"key1": "val1"},
                "Optimizer": {"key1": "val1"}
            }
            global_config = GlobalConfig(config)
            global_config.key1 == "val1"
            global_config.Optimizer == {"key1": "val1"}

        """
        # print(config)
        self.config = config
        self.special_keys = special_keys
        # self.remap_keys = remap_keys
        self._setup()

    def _setup(self):
        # self._remap_keys(self.config, self.remap_keys)
        self._default_values()
        self._load_config()
        self._post_loading_hook()
        # self._parameter_check()

    def _default_values(self):
        """This method is used to store the default value for global parameters

        Attributes:
        self.Backbones: dictionary contains the backbone parameters, it must contains the key
                       type: backbone name(str) to indicate which backbone to use in the pipeline
        self.Groups: dictionary contains the config for the backbones of each group
        self.Necks: dictionary contains the config for necks of each group
        self.image_height: The actual image height used for network input.
        self.image_width: The actual image width used for network input
        self.image_channel: The actual image channel number used for network input.
        self.max_data_fetch_iteration: The maximum allowance for data re-fetch for one data,
                                       training process will be killed if this limit is reached.
        self.process_timeout_seconds: Maximum waiting time(seconds) for backward cross cards/nodes,
                                      training process will be killed if this limit is reached.
        self.spatial2channel: boolean variable that control the global augmentation spatial2channel,
                              please note this flag will not influence other options such as image width/height or
                              network input channels
        self.save_checkpoint_before_validation: It is only effective in epoch-based training,
                                                i.e. when max_iterations is None.
                                                It modifies the behavior of check_val_every_n_epochs
                                                so that the checkpoint is saved before validation in
                                                case validation crashes.
        self.save_every_n_epochs: Only used in epoch-based training, i.e. if ``max_iteration`` is None.
                                  It allows user to save checkpoint during epoch-based regardless of
                                  validation frequency. By default, we save checkpoint every epoch.
        self.dist_val_split_n_chunks: During distributed validation, sometime the all_gather operation of
                                      a batch takes too much GPU memory. This argument allows user to
                                      break down a batch into n chunks to reduce GPU memory
                                      consumption. Default 1, i.e. no breaking down.
        """
        self.Backbones: dict = {}
        # self.customized_amp_scalar: dict = {}
        # self.PreTaskNecks: dict = {}
        self.load_from: str = ""
        self.resume_from: str = ""
        self.Groups: dict = {}
        self.Necks: dict = {}
        self.Transformer: dict = {}
        self.Tasks: dict = {}
        self.tasks: list = []
        self.debug: bool = False
        self.precision: int = 32
        self.gpus: int = 1
        self.num_nodes: int = 1
        self.save: str = ""
        self.image_time: int = 0
        self.image_per_gpu: int = 0
        self.log_every: int = 500
        self.visualize_every: int = 2000
        self.optimizer: dict = {}
        self.lr_scheduler: dict = {}
        self.dump_path: str = ""
        self.max_epochs: int = 0
        self.max_steps: int = 100000000
        self.max_iterations: Union[int, None] = None
        self.save_every_n_iterations: Union[int, None] = None
        self.save_every_n_epochs: int = 1
        self.logging_level: str = "INFO"
        self.dump_json_during_validation: bool = True
        self.dump_calibset = False
        self.onnx_path = None

    def _load_config(self):
        for key, val in self.config.items():
            if key in self.special_keys:
                for sub_key, sub_val in val.items():
                    setattr(self, sub_key, sub_val)
            else:
                setattr(self, key, val)
            setattr(self, key, val)

    def _post_loading_hook(self):

        if not self.max_epochs:
            self.iteration_based_lr_schedulers = True

        if self.skip_sanity_check:
            self.num_sanity_val_steps = 0

        if not self.max_epochs and self.max_iterations:
            # TODO Hack, training will be closed by max iteration
            self.max_epochs = int(1e6)

        # self.image_shape = (self.image_height, self.image_width, self.image_channel)

    def _parameter_check(self):
        """This method is called at the end of the _setup method, all
        parameter sanity check should be put here"""
        assert isinstance(self.image_per_gpu, int)
        assert self.image_per_gpu > 0

        if not valid_lr(self.optimizer):
            raise ValueError("Got invalid lr")
        if not valid_warmup_epochs(self.lr_scheduler):
            raise ValueError("Got invalid warm-up epochs")

        if self.ddp_init_method not in ["tcp"]:
            raise ValueError("only tcp is supported")

        assert self.Backbones, "Backbones sections are not found in config.yaml"

        if hasattr(self, "load_checkpoint_strict"):
            assert (
                self.load_checkpoint_strict
            ), "[Config] Please use config to control loading, do not set strict to False"

    @staticmethod
    def _remap_keys(config, remap_keys):
        """TODO, HACKY
        This is to remap the old config key to match with the new config key.
        """
        for org, dst, op in remap_keys:
            sub_org_dict, ignore = config, False
            org_keys, dst_keys = org.split("/"), dst.split("/")
            last_org_key, last_dst_key = org_keys.pop(), dst_keys.pop()

            for key in org_keys:
                if key not in sub_org_dict:
                    ignore = True
                    break
                sub_org_dict = sub_org_dict[key]

            if not ignore and last_org_key in sub_org_dict:
                if op == "REPLACE":
                    val = sub_org_dict.pop(last_org_key)
                elif op == "COPY":
                    val = deepcopy(sub_org_dict[last_org_key])
                else:
                    raise ValueError("op only support ['REPLACE', 'COPY']")
                sub_dsts = [(config, 0, val)]
                while len(sub_dsts) != 0:
                    sub_dst_dict, idx, val = sub_dsts.pop()
                    if idx == len(dst_keys):
                        if last_dst_key in sub_dst_dict and sub_dst_dict[last_dst_key]:
                            continue
                        sub_dst_dict[last_dst_key] = val
                    else:
                        key = dst_keys[idx]
                        idx += 1
                        if key[-1] == "*" and key[:-1] not in sub_dst_dict:
                            sub_dst_dict[key[:-1]] = {}
                        elif key not in sub_dst_dict:
                            sub_dst_dict[key] = {}

                        if key[-1] == "*":
                            key = key[:-1]
                            for subkey in sub_dst_dict[key].keys():
                                if subkey.upper() in val:
                                    sub_dsts.append(
                                        (sub_dst_dict[key][subkey], idx, val[subkey.upper()]))
                                elif subkey.lower() in val:
                                    sub_dsts.append(
                                        (sub_dst_dict[key][subkey], idx, val[subkey.lower()]))
                        else:
                            sub_dsts.append((sub_dst_dict[key], idx, val))

    def __str__(self):
        return "\n".join(self.__dict__.keys())

    def __repr__(self):
        return self.__str__()


def valid_warmup_epochs(lr_scheduler_config: dict) -> bool:
    """warmup_epochs can be empty or a number larget than 0"""
    if "warmup_epochs" not in lr_scheduler_config:
        return True

    if not isinstance(lr_scheduler_config["warmup_epochs"], (int, float)):
        return False

    if lr_scheduler_config["warmup_epochs"] <= 0:
        return False

    return True


def valid_lr(optimizer_config: dict) -> bool:
    """This function is used to check the optimizer's lr, this
    key is not required since some torch optimizer have default lr"""
    if "lr" not in optimizer_config:
        return True

    if optimizer_config["lr"] < 0:
        return False

    return True
