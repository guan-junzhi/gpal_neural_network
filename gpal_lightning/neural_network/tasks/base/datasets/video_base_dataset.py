import logging
from gpal_lightning.neural_network.tasks.base.datasets.image_base_dataset import ImageBaseDataset


class VideoBaseDataset(ImageBaseDataset):
    """
    This class is used to process the video data, different from image base dataset, video mean each data is a
    list of sequential images, and those images should have same image augmentations
    """

    def _dataset_logging(self):
        logging.info("%s: %s has %s %s video labels", self.task_name,
                     self.dataset_name, len(self.dataset), self.phase)
