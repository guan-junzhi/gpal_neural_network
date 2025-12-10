#!/usr/bin/env python3
"""Evaluate stored prediction dumps without running inference.
python run_eval_from_dump.py --config xxx.yaml --dump-root dump/ --tasks driving_bev_sta --save ./ 
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gpal_lightning import const
from gpal_lightning.data.dataset_identifier import DatasetIdentifier
from gpal_lightning.neural_network.tasks.build_task import build_tasks, build_tasks_datasets
from gpal_lightning.utils.evaluation_helpers.evaluation_pipeline import EvaluationPipeline
from gpal_lightning.utils.load_global_config import load_global_config
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KPI evaluation from previously dumped predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the global config yaml.")
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--load_from", type=str,help="Passed through untouched.")
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--tasks", nargs="+", type=str,)
    parser.add_argument("--dump-root", type=str, required=True)
    parser.add_argument("--dataset-indices", type=int, nargs="+",
                        help="Dataset indices to run. Default: evaluate every dataset defined in the config.")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip datasets that do not have dumps instead of raising.")
    parser.add_argument("--print-json", action="store_true",
                        help="Pretty-print KPI dictionaries to stdout for each dataset.")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--image_per_gpu", type=int)
    parser.add_argument("--workers_per_gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--calib_data_save_path", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bar output.")

    args = parser.parse_args()
    args.num_nodes = const.NUM_NODES
    return args


def _extend_args_for_uppercase_tasks(args: argparse.Namespace) -> None:
    if args.tasks:
        args.tasks = [task.upper() for task in args.tasks]


def _build_dataset_identifiers(tasks: Sequence) -> Dict[str, List[DatasetIdentifier]]:
    mapping: Dict[str, List[DatasetIdentifier]] = defaultdict(list)
    for task in tasks:
        datasets = getattr(task, "val_datasets", [])
        for idx, dataset in enumerate(datasets):
            if len(dataset) == 0:
                continue
            identifier = DatasetIdentifier(
                getattr(dataset, "camera_name", ""),
                getattr(dataset, "root_dir", ""),
                getattr(dataset, "dataset_name", ""),
                getattr(dataset, "sql_filter", ""),
                idx,
                len(dataset),
            )
            mapping[task.name].append(identifier)
    return mapping


def _dump_exists(dump_root: str, task_name: str, dataset_idx: int) -> bool:
    preds_pattern = os.path.join(
        dump_root,
        task_name,
        str(dataset_idx),
        const.PREDS,
        "**",
        f"*{const.EVALUATION_FILES_EXTENSION}",
    )
    matching = glob(preds_pattern, recursive=True)
    return len(matching) > 0


def _select_indices(total: int, requested: Optional[Iterable[int]]) -> List[int]:
    if requested is None:
        return list(range(total))
    unique = sorted(set(requested))
    selected = []
    for idx in unique:
        if idx >= total:
            logging.warning("Requested dataset idx %d exceeds available dataset count %d; skipping", idx, total)
            continue
        selected.append(idx)
    return selected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()
    _extend_args_for_uppercase_tasks(args)

    logging.info("Loading global config from %s", args.config)
    global_config = load_global_config(args, override=False, dump_config=False)
    global_config.validation = True

    logging.info("Building task objects and validation datasets")
    tasks = build_tasks(global_config, const.PHASE_VALIDATION, tasks_root="gpal_nn.tasks")
    build_tasks_datasets(global_config, tasks, const.PHASE_VALIDATION)
    dataset_identifiers = _build_dataset_identifiers(tasks)

    total_eval_pairs = 0
    for task in tasks:
        identifiers = dataset_identifiers.get(task.name, [])
        if not identifiers:
            continue
        indices = _select_indices(len(identifiers), args.dataset_indices)
        total_eval_pairs += len(indices)

    results: List[Tuple[str, int, Dict, str]] = []
    progress_bar = None
    if total_eval_pairs > 0 and not args.no_progress:
        progress_bar = tqdm(total=total_eval_pairs, desc="Evaluating datasets", unit="dataset")

    for task in tasks:
        identifiers = dataset_identifiers.get(task.name, [])
        if not identifiers:
            logging.warning("Task %s has no validation datasets configured; skipping", task.name)
            continue

        indices = _select_indices(len(identifiers), args.dataset_indices)
        for dataset_idx in indices:
            identifier = identifiers[dataset_idx]
            dump_present = _dump_exists(args.dump_root, task.name, dataset_idx)
            if not dump_present:
                message = (
                    f"No dumps found under {args.dump_root}/{task.name}/{dataset_idx}"
                )
                if args.skip_missing:
                    logging.warning("%s -- skipping", message)
                    continue
                raise FileNotFoundError(message)

            logging.info("Evaluating task=%s dataset_idx=%d (%s)",
                         task.name, dataset_idx, identifier.dataset_name)
            kpi, formatted_kpi = EvaluationPipeline.run(
                task,
                args.dump_root,
                dataset_idx,
                identifier,
            )
            results.append((task.name, dataset_idx, kpi, formatted_kpi))

            if args.print_json:
                pretty = json.dumps(kpi, indent=2, ensure_ascii=False)
                logging.info("KPI JSON for %s idx=%d:\n%s", task.name, dataset_idx, pretty)
            else:
                logging.info("Formatted KPI: %s", formatted_kpi)

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"{task.name}:{dataset_idx}")

    if not results:
        logging.warning("No datasets were evaluated. Check --dataset-indices and dump availability.")
        if progress_bar is not None:
            progress_bar.close()
        return

    logging.info("Evaluation finished. Datasets processed: %d", len(results))
    if progress_bar is not None:
        progress_bar.close()


if __name__ == "__main__":
    main()
