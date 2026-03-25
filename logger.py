import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

_ON_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT"))

_logger = logging.getLogger("moviematch.search")
_logger.setLevel(logging.INFO)
_logger.propagate = False

os.makedirs(LOG_DIR, exist_ok=True)
_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "search.log"), maxBytes=10 * 1024 * 1024, backupCount=5
)
_handler.setFormatter(logging.Formatter("%(message)s"))

_logger.addHandler(_handler)

# Configure root logger so module loggers (app.py, db.py, etc.) surface to the same destination
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout if _ON_RAILWAY else None,  # None → defaults to stderr (standard local behavior)
)


def log_request(record: dict) -> None:
    _logger.info(json.dumps(record, default=str))
