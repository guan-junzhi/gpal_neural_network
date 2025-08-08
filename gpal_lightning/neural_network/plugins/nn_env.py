import os
import logging
from pytorch_lightning.plugins.environments import LightningEnvironment

# from zpilot_lightning import const


class NNTCPEnvironment(LightningEnvironment):
    def __init__(self):
        super().__init__()

        print("MASTER_PORT" in os.environ and "NODE_RANK" in os.environ and "MASTER_ADDR" in os.environ)
        exit(1)
        # the following environment variables should be set in const.py
        if "MASTER_PORT" in os.environ and "NODE_RANK" in os.environ and "MASTER_ADDR" in os.environ:
            return

        logging.warning("Missing variables in os.environ !!!")
