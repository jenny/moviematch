import json
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "search.log")

os.makedirs(LOG_DIR, exist_ok=True)

_handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5)
_handler.setFormatter(logging.Formatter("%(message)s"))

_logger = logging.getLogger("moviematch.search")
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
_logger.propagate = False


def log_request(record: dict) -> None:
    _logger.info(json.dumps(record, default=str))
