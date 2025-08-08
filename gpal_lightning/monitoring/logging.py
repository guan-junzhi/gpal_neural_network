import logging
import sys

logger = logging.getLogger('Gpal_Neural_Network')
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging_handler_out = logging.StreamHandler(sys.stdout)
logging_handler_out.setLevel(logging.DEBUG)
logging_handler_out.addFilter(lambda record: record.levelno <= logging.WARNING)
logging_handler_out.setFormatter(formatter)
logger.addHandler(logging_handler_out)

logging_handler_err = logging.StreamHandler(sys.stderr)
logging_handler_err.setLevel(logging.ERROR)
logging_handler_err.setFormatter(formatter)
logger.addHandler(logging_handler_err)


def is_logging(rank: int, step: int, log_interval: int):
    if rank != 0:
        return False
    if step <= 0:
        return False
    if step % log_interval != 0:
        return False
    return True
