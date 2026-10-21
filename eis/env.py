"""Paths and logging shared by the modules of the library."""

import logging
import os

import coloredlogs
import numpy as np

# folder of this package (eis/)
DIR_LIB = os.path.dirname(os.path.abspath(__file__))
# repository root, one level above eis/
DIR_BASE = os.path.abspath(os.path.join(DIR_LIB, os.pardir))

# the dataset workbooks (see eis.datasets)
DIR_DATASETS = os.path.join(DIR_BASE, "datasets")


LOG_FORMAT = "[%(asctime)s][%(levelname).3s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# The library logs warnings only. Progress messages (level INFO) are not shown.
logger = logging.getLogger("eis")
logger.setLevel(logging.WARNING)
# propagate = False keeps the messages away from the handlers of the root logger
logger.propagate = False

log_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# colored console output
coloredlogs.install(
    logger=logger,
    level="WARNING",
    fmt=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    isatty=True,
)

# numpy print options
np.set_printoptions(precision=6, suppress=True)
