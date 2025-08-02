
"""
This is the file to save shared common constants.
"""
import logging
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np

JOBNAME = os.environ.get("JOBNAME", -1)
RANK = int(os.environ.get("RANK", 0))
os.environ['NODE_RANK'] = str(RANK)

EPS = 1e-6
EPS_FP16 = 1e-3
# CPU_COUNT = multiprocessing.cpu_count()
# EVALUATION_LOG_FREQUENCY = 100
# IMAGE_EXTENSION = ".png"
PYTORCH_UPGRADE_VERSION = "2.0.0.dev"
# TAGGER = "tagger"
# IMAGES = "images"
# FEATURES = "features"
FILE_EXTENSION = ".pth"
ONNX_EXTENSION = ".onnx"
CHECKPOINT_NAME_LAST = "checkpoint"
CHECKPOINT_PATH = "checkpoint"
LOG_PATH = "log"
CONFIG_NAME = "config"
CONFIG_EXTENSION = ".yaml"
TEMP_PATH = "/tmp"

PREDS = "preds"
TRUES = "trues"
METADATA = "metadata"
EVALUATION = "evaluation"
EVALUATION_FILES_EXTENSION = ".json"
JOB_DUMP_PATH = os.path.join(TEMP_PATH, f"mf_jobs_{JOBNAME}")


PHASE_TRAINING = "training"
PHASE_VALIDATION = "validation"
PHASE_INFERENCE = "inference"

CURRENT_TIME = datetime.now().strftime("%Y%m%d%H%M%S")
# JOB_EVALUATION_PATH = os.path.join(JOB_DUMP_PATH, EVALUATION, CURRENT_TIME)

# ---------------------- HELPER FUNCTIONS AND CLASSES --------------------


def is_rank_zero():
    """Helper function to check gpu 0 node 0 process."""
    node_rank = int(os.getenv("NODE_RANK", RANK))
    if node_rank == 0:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return local_rank == 0
    return False


def sync(distributed):
    global CURRENT_TIME
    global JOB_EVALUATION_PATH
    time_list = ["" for _ in range(distributed.get_world_size())]
    distributed.all_gather_object(time_list, CURRENT_TIME)
    CURRENT_TIME = time_list[0]
    JOB_EVALUATION_PATH = os.path.join(JOB_DUMP_PATH, EVALUATION, CURRENT_TIME)
