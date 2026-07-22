"""
OMDb API integration for critic/audience review scores.

OMDb (omdbapi.com) is a community movie-data API that redistributes review scores
from Rotten Tomatoes (critic Tomatometer), IMDb, and Metacritic in a single call.
It's the source for the "Ratings & Scores" overlay section, alongside the TMDB user
score (fetched separately in the /ratings route).

OPTIONAL: OMDb requires a free API key (1,000 requests/day). When OMDB_API_KEY is
unset, fetch_ratings() returns {} and logs at DEBUG — the app degrades to showing the
TMDB score alone (or nothing) rather than erroring. This mirrors watchmode.py's
fail-soft posture when WATCHMODE_API_KEY is absent.

Every non-cached OMDb HTTP call is logged with the tag [omdb_api_call] for grep-based
monitoring of daily budget consumption.
"""

import logging
import threading
import time
from typing import Any

import requests

from config import OMDB_API_KEY, OMDB_API_URL, OMDB_TIMEOUT_S

logger = logging.getLogger(__name__)

# TTL response cache: {key: (value, timestamp)}. Key: "ratings:{title}:{year}".
# 24-hour TTL — review scores drift slowly and a stale value for a day is harmless.
# Thread-safe via _cache_lock (batch lookups run concurrently in a thread pool).
_cache: dict[str, tuple[Any, float]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 86400  # seconds


def _cache_get(key: str) -> tuple[bool, Any]:
    """Return (True, value) on cache hit, (False, None) on miss or expiry."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return False, None
        value, ts = entry
        if time.time() - ts > _CACHE_TTL:
            del _cache[key]
            return False, None
        return True, value


def _cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = (value, time.time())


def _parse_percent(value: str) -> int | None:
    """Parse a Rotten Tomatoes-style '87%' into the int 87. Returns None on 'N/A'/garbage."""
    value = (value or "").strip().rstrip("%")
    try:
        return int(value)
    except ValueError:
        return None


def _parse_ratio(value: str) -> float | None:
    """Parse an IMDb-style '8.8/10' or bare '8.8' into 8.8. Returns None on 'N/A'/garbage."""
    value = (value or "").strip()
    if "/" in value:
        value = value.split("/", 1)[0].strip()
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    """Parse a Metacritic-style '74' or '74/100' into 74. Returns None on 'N/A'/garbage."""
    value = (value or "").strip()
    if "/" in value:
        value = value.split("/", 1)[0].strip()
    try:
        return int(value)
    except ValueError:
        return None


def fetch_ratings(title: str, year: str = "") -> dict:
    """Fetch RT/IMDb/Metacritic scores for a movie from OMDb.

    Returns a dict with any of these keys present only when OMDb supplied a real value:
        rt        -> int   (Rotten Tomatoes critic Tomatometer, e.g. 87)
        imdb      -> float (IMDb rating out of 10, e.g. 8.8)
        metacritic-> int   (Metascore out of 100, e.g. 74)
        imdb_id   -> str   (e.g. "tt1375666") — used to deep-link the IMDb page

    Returns {} when OMDB_API_KEY is unset (fail-soft), the title isn't found, or the
    request fails. Never raises. Results are cached for 24h to conserve the free-tier
    budget; failures are NOT cached so a transient error can be retried.
    """
    if not OMDB_API_KEY:
        logger.debug("OMDb: OMDB_API_KEY unset — skipping ratings lookup for '%s'", title)
        return {}

    cache_key = f"ratings:{title}:{year}"
    hit, cached = _cache_get(cache_key)
    if hit:
        return cached  # may be {} (title genuinely not found on OMDb)

    params = {"apikey": OMDB_API_KEY, "t": title, "type": "movie"}
    if year:
        params["y"] = year

    logger.info(f"[omdb_api_call] ratings title='{title}' year='{year}'")
    try:
        response = requests.get(OMDB_API_URL, params=params, timeout=OMDB_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()

        # OMDb signals "not found" with {"Response": "False", "Error": "..."} and HTTP 200.
        if data.get("Response") != "True":
            logger.debug("OMDb: no match for '%s' (%s): %s", title, year, data.get("Error"))
            _cache_set(cache_key, {})  # cache the miss — the title genuinely isn't on OMDb
            return {}

        result: dict = {}

        # RT critic Tomatometer lives only in the Ratings[] array, keyed by source name.
        for r in data.get("Ratings", []):
            if r.get("Source") == "Rotten Tomatoes":
                rt = _parse_percent(r.get("Value", ""))
                if rt is not None:
                    result["rt"] = rt
                break

        # IMDb rating: prefer the top-level imdbRating field (already a bare "8.8").
        imdb = _parse_ratio(data.get("imdbRating", ""))
        if imdb is not None:
            result["imdb"] = imdb

        # Metacritic: top-level Metascore field (bare "74"), "N/A" when absent.
        mc = _parse_int(data.get("Metascore", ""))
        if mc is not None:
            result["metacritic"] = mc

        # imdbID enables an exact deep link to the IMDb page (e.g. tt1375666).
        imdb_id = (data.get("imdbID") or "").strip()
        if imdb_id:
            result["imdb_id"] = imdb_id

        _cache_set(cache_key, result)
        logger.debug("OMDb: '%s' (%s) → %s", title, year, result)
        return result
    except Exception as e:
        logger.warning("OMDb: ratings lookup failed for '%s': %s", title, e)
        return {}  # don't cache failures — allow retry on the next request
