import torch

from gpal_lightning import const


class BaseLogger:
    """Save scalar and image log to tensorboard, not used now"""

    def __init__(self, global_config, task_config):
        self.global_config = global_config
        self.task_config = task_config

    def scalar_log(self, iteration, phase, log_writer, loss_info, **kwargs):
        """

        Args:
            iteration: int, current iteration of calling the log
            phase: str, used to form the name of the logging
            log_writer: tensorboard object
            loss_info: dictionary, loss_name: loss_value pairs
            **kwargs: dictionary contains
            other scalar values wants to be stored, shouldn't be nested dict

        Returns:

        """
        name = f"{self.task_config.name}/{phase}/"
        with torch.no_grad():
            for key, val in kwargs.items():
                log_writer.add_scalar(name + key, val, iteration)
            for key, val in loss_info.items():
                log_writer.add_scalar(name + key, val, iteration)

    def image_log(self, iteration, phase, log_writer, idx, vis_pred):
        """This method takes in a pair of pred/true images."""
        name = "/".join([self.task_config.name, phase])
        log_writer.add_image(f"{name}/vis_{idx}", vis_pred, iteration)
