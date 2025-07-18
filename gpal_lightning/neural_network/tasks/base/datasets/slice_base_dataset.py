import logging
from gpal_lightning.neural_network.tasks.base.datasets.image_base_dataset import ImageBaseDataset


class SliceBaseDataset(ImageBaseDataset):
    """
    This class is used to process the slice data, different from image base dataset, slice mean each data is a
    dict of multiple cameras, and those camera should have same image augmentations
    """
    def _dataset_logging(self):
        logging.info("%s: %s has %s %s slice labels", self.task_name, self.dataset_name, len(self.dataset), self.phase)

