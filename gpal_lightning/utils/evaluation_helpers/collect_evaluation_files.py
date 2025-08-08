import os

from gpal_lightning import const
from gpal_lightning.utils.evaluation_helpers.collect_dumped_files import collect_dumped_files


def collect_evaluation_files(curr_task, dataset_idx: int, files_root: str) -> tuple:
    """Collect files for evaluation, please change the
    return type to dict if adding more return variables."""
    collected_preds = collect_dumped_files(os.path.join(
        curr_task.name, str(dataset_idx)), const.PREDS, files_root)
    collected_trues = collect_dumped_files(os.path.join(
        curr_task.name, str(dataset_idx)), const.TRUES, files_root)
    collected_metadata = collect_dumped_files(
        os.path.join(curr_task.name, str(dataset_idx)
                     ), const.METADATA, files_root
    )

    return collected_preds, collected_trues, collected_metadata
