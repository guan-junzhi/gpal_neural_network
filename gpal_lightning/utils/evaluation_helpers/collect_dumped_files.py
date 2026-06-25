import os
import logging
from glob import glob
from pathlib import Path

from gpal_lightning import const
from gpal_lightning.utils.json_helpers.json_to_dict import json_to_dict
from gpal_lightning.utils.pickle_helpers.pt_to_dict import pt_to_dict


def collect_dumped_files(task_name: str,
                         sub_path: str,
                         file_root: str = None) -> dict:
    """This function is used to collect lightning runner evaluation dumped files and collect them into memory for
    evaluation purpose."""
    if file_root is None:
        file_root = const.JOB_EVALUATION_PATH
    # loaded_data = {}
    files_path = os.path.join(file_root, task_name, sub_path)

    print(os.path.join(files_path, "**", "*" + const.EVALUATION_FILES_EXTENSION))
    files = glob(os.path.join(files_path,
                              "**",
                              "*" + const.EVALUATION_FILES_EXTENSION),
                 recursive=True)
    # print(files)
    logging.warning("Load dumped files ...")
    logging.warning("Start with {}".format(files[0]))
    # Serial loading to avoid fork/thread issues with HBRuntime/ONNX Runtime loaded
    loaded_data = []
    for file in files:
        uuid, data = _collect_dumped_files(file)
        loaded_data.append((uuid, data))
    logging.warning(
        "Dumped file loading done, {} files loaded.".format(len(loaded_data)))
    return dict(loaded_data)


def _collect_dumped_files(file):
    uuid = os.path.basename(Path(file).stem)
    ext = os.path.splitext(file)[1]
    if ext.lower() == '.json':
        try:
            data = json_to_dict(file)
        except:
            logging.warning("fail to load: %s", uuid)
            data = None
    elif ext.lower() == '.pth':
        data = pt_to_dict(file)
    else:
        raise ValueError("Unrecognized dumped file type: {}".format(file))
    return uuid, data
