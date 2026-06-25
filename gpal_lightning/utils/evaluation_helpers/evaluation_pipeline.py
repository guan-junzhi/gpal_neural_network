from typing import Tuple

from gpal_lightning.data.dataset_identifier import DatasetIdentifier
from gpal_lightning.utils.evaluation_helpers.collect_evaluation_files import collect_evaluation_files


class EvaluationPipeline:
    """This class is used to apply the evaluation pipeline
    1. Collect dumped data from pred step.
    2. Send collected files to task evaluator.
    3. Send results to task kpi formatter.
    """

    @classmethod
    def run(
        cls, curr_task, files_root: str, dataset_idx: int, dataset_identifier: DatasetIdentifier
    ) -> Tuple[dict, str]:
        """

        Args:
            curr_task: Task object
            files_root: path contains "pred", "trues", and "metadata" folders
            dataset_idx: int shows the idx of current dataset, it will be used to accumulate jsons for kpi
            dataset_identifier: current dataset's information, used by kpi formatter
        Returns: kpi and formatted kpi

        """

        collected_preds, collected_trues, collected_metadata = collect_evaluation_files(
            curr_task, dataset_idx, files_root
        )

        # print(collected_preds, collected_trues, collected_metadata)

        for uuid, pred in collected_preds.items():
            if pred is None or uuid not in collected_trues or uuid not in collected_metadata:
                continue
            if collected_trues[uuid] is None or collected_metadata[uuid] is None:
                continue
            true = collected_trues[uuid]
            metadata = collected_metadata[uuid]

            curr_task.evaluator.process(pred, true, metadata)

        curr_task.evaluator.dataset_identifier = dataset_identifier
        kpi = curr_task.evaluator.generate_kpi()
        # don't need to reset the camera_name since reset_kpi is called at the end
        formatted_kpi = curr_task.kpi_formatter.format_kpi(
            kpi, dataset_identifier)
        curr_task.reset_kpi()

        return kpi, formatted_kpi
