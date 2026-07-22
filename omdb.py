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
monitoring of daily budget consumption. Two failure modes get their own tags so they
can be told apart from ordinary network noise:

    [omdb_quota_exhausted]  the 1,000/day free-tier cap is spent
    [omdb_auth_failed]      the key is invalid, revoked, or never activated

Both are logged on every occurrence — deliberately not deduplicated, so the log line
count doubles as a measure of how much demand the exhausted quota is turning away.
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


def _redact(value: object) -> str:
    """Stringify `value` with the API key scrubbed out, for safe logging.

    requests' HTTPError message embeds the full request URL — query string included —
    so logging a raw exception would print OMDB_API_KEY on every failure. On Railway
    those lines go to stdout, where the key would be retained indefinitely.
    """
    text = str(value)
    if OMDB_API_KEY:
        text = text.replace(OMDB_API_KEY, "***")
    return text


def _error_message(response: requests.Response) -> str:
    """Best-effort read of OMDb's {"Error": "..."} body. Returns '' if it isn't JSON."""
    try:
        return str(response.json().get("Error", "")).strip()
    except Exception:
        return ""


def fetch_ratings(title: str, year: str = "") -> dict:
    """Fetch RT/IMDb/Metacritic scores for a movie from OMDb.

    Returns a dict with any of these keys present only when OMDb supplied a real value:
        rt        -> int   (Rotten Tomatoes critic Tomatometer, e.g. 87)
        imdb      -> float (IMDb rating out of 10, e.g. 8.8)
        metacritic-> int   (Metascore out of 100, e.g. 74)
        imdb_id   -> str   (e.g. "tt1375666") — used to deep-link the IMDb page

    Returns {} when OMDB_API_KEY is unset (fail-soft), the title isn't found, the daily
    quota is spent, the key is rejected, or the request fails. Never raises. Results are
    cached for 24h to conserve the free-tier budget; failures are NOT cached so a
    transient error can be retried.
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

        # OMDb answers 401 for BOTH a spent daily quota and a bad key, and only the JSON
        # body tells them apart. So inspect the body before raise_for_status(), which
        # would collapse both into one opaque "401 Client Error" and drop the reason.
        if response.status_code == 401:
            omdb_error = _error_message(response)
            if "limit" in omdb_error.lower():
                # Free tier is 1,000 req/day, reset by OMDb on their own schedule.
                # Scores silently degrade to the TMDB value alone until then.
                logger.error(
                    "[omdb_quota_exhausted] daily free-tier cap reached — review scores "
                    "degrade to TMDB-only until OMDb resets. title='%s' year='%s' omdb_error='%s'",
                    title, year, omdb_error,
                )
            else:
                # Almost always a newly-issued key whose activation link was never
                # clicked; also covers a revoked or mistyped key.
                logger.error(
                    "[omdb_auth_failed] OMDb rejected OMDB_API_KEY — check the key is set "
                    "correctly and that the activation link emailed by omdbapi.com was "
                    "clicked. title='%s' year='%s' omdb_error='%s'",
                    title, year, omdb_error,
                )
            # Not cached: both conditions are recoverable (quota resets, keys get
            # activated), so a later lookup for the same title should retry.
            return {}

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
        logger.warning("OMDb: ratings lookup failed for '%s': %s", title, _redact(e))
        return {}  # don't cache failures — allow retry on the next request
