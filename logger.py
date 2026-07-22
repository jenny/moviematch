import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import config
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

# DEBUG locally, INFO in production — keeps Railway logs clean while making local dev verbose
_log_level = logging.INFO if _ON_RAILWAY else logging.DEBUG

# Configure root logger so module loggers (app.py, db.py, etc.) surface to the same destination
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout if _ON_RAILWAY else None,  # None → defaults to stderr (standard local behavior)
)


# Config attributes holding a credential that must never reach a log line.
# The motivating case: APIs that authenticate by *query parameter* (Watchmode, OMDb)
# leak their key through any bare exception message, because requests' HTTPError
# embeds the full request URL — query string and all. TMDB and Anthropic authenticate
# by header and so don't leak this way, but they're listed for defense in depth.
_SECRET_CONFIG_ATTRS = (
    "WATCHMODE_API_KEY",
    "OMDB_API_KEY",
    "TMDB_KEY",
    "ANTHROPIC_API_KEY",
    "PINECONE_API_KEY",
    "ADMIN_PASSWORD",
    "ADMIN_SECRET_KEY",
)

# Below this length a "secret" is too short to blind-replace safely — a 3-character
# value would riddle unrelated log text with ***. Real keys are far longer.
_MIN_SECRET_LEN = 8


def redact(value: object) -> str:
    """Stringify `value` with every configured credential replaced by ***.

    Wrap any log argument that may carry a request URL or raw exception — chiefly
    `except Exception as e` handlers around `requests` calls. Reads config at call
    time (not import time) so tests can patch individual keys.
    """
    text = str(value)
    for attr in _SECRET_CONFIG_ATTRS:
        secret = getattr(config, attr, None)
        # Guard the empty string explicitly: "abc".replace("", "*") interleaves the
        # replacement between every character, which would mangle the whole message.
        if isinstance(secret, str) and len(secret) >= _MIN_SECRET_LEN:
            text = text.replace(secret, "***")
    return text


def log_request(record: dict) -> None:
    _logger.info(json.dumps(record, default=str))
