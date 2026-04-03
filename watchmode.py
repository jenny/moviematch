"""
Watchmode API integration for streaming availability data.

Watchmode (watchmode.com) provides comprehensive, up-to-date streaming provider data
for the US and other regions. It's used as the primary source for the /streaming endpoint,
with TMDB as a fallback when WATCHMODE_API_KEY is not configured.

Free tier: 1,000 requests/month. Each /streaming call uses 2 requests (search + sources),
plus one shared request for the source logo catalog (cached in memory for the process lifetime).
Every non-cached Watchmode API call is logged with the tag [watchmode_api_call] for grep-based
monitoring of monthly budget consumption.
"""

import logging
import threading
import time
from typing import Any

import requests

from config import WATCHMODE_API_KEY

logger = logging.getLogger(__name__)

WATCHMODE_BASE_URL = "https://api.watchmode.com/v1"

# In-memory cache: {source_id: logo_url} — populated once from the Watchmode sources catalog
_source_logos: dict[int, str] = {}
_source_logos_lock = threading.Lock()
_source_logos_loaded = False

# TTL response cache: {key: (value, timestamp)}
# Keys: "search:{title}:{year}" for title IDs, "providers:{title_id}" for provider lists.
# 24-hour TTL — streaming availability doesn't change hourly. Thread-safe via _cache_lock.
_cache: dict[str, tuple[Any, float]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 86400  # seconds

# Session-lifetime API usage counters (reset on process restart / Railway deploy).
# Used by the admin panel to show budget consumption at a glance.
_api_calls = 0    # real Watchmode HTTP requests made (cache misses + logo catalog)
_cache_hits = 0   # lookups served from cache (API calls saved)


def _cache_get(key: str) -> tuple[bool, Any]:
    """Return (True, value) on cache hit, (False, None) on miss or expiry."""
    global _cache_hits
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return False, None
        value, ts = entry
        if time.time() - ts > _CACHE_TTL:
            del _cache[key]
            return False, None
        _cache_hits += 1
        return True, value


def _cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = (value, time.time())


def get_stats() -> dict:
    """Return current Watchmode API usage stats for the admin panel."""
    with _cache_lock:
        cache_size = len(_cache)
    return {
        "api_calls_session": _api_calls,
        "cache_hits_session": _cache_hits,
        "cache_size": cache_size,
        "monthly_limit": 1000,
    }


def _load_source_logos() -> None:
    """Fetch and cache Watchmode's source catalog to get provider logo URLs.

    Called lazily before the first title sources lookup. Uses one API request
    and caches the result for the lifetime of the process (~200 entries).
    """
    global _source_logos_loaded
    with _source_logos_lock:
        if _source_logos_loaded:
            return
        try:
            global _api_calls
            _api_calls += 1
            logger.info("[watchmode_api_call] sources catalog")
            response = requests.get(
                WATCHMODE_BASE_URL + "/sources/",
                params={"apiKey": WATCHMODE_API_KEY},
                timeout=10,
            )
            response.raise_for_status()
            for source in response.json():
                logo = source.get("logo_100px")
                if logo:
                    _source_logos[source["id"]] = logo
            _source_logos_loaded = True
            logger.info(f"Watchmode: cached {len(_source_logos)} source logos")
        except Exception as e:
            logger.warning(f"Watchmode: failed to load source logos: {e}")


def search_title(title: str, year: str = "") -> int | None:
    """Search Watchmode for a movie by title, return Watchmode title ID or None.

    When year is provided, prefers exact year match among search results.
    Falls back to the first result if no year match is found.
    Results are cached for 24 hours to conserve the free-tier API budget.
    """
    cache_key = f"search:{title}:{year}"
    hit, cached = _cache_get(cache_key)
    if hit:
        return cached  # may be None (title not found on Watchmode)

    # Increment under _cache_lock so all counter mutations are serialized by one lock.
    with _cache_lock:
        global _api_calls
        _api_calls += 1
    logger.info(f"[watchmode_api_call] search title='{title}' year='{year}'")
    try:
        response = requests.get(
            WATCHMODE_BASE_URL + "/search/",
            params={
                "apiKey": WATCHMODE_API_KEY,
                "search_field": "name",
                "search_value": title,
                "types": "movie",
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("title_results", [])
        if not results:
            logger.debug(f"Watchmode: no results for '{title}'")
            _cache_set(cache_key, None)
            return None
        logger.debug(f"Watchmode: search results for '{title}': {[(r.get('id'), r.get('year')) for r in results[:5]]}")
        # Prefer an exact year match when the caller provides one
        if year:
            for r in results:
                if str(r.get("year", "")) == str(year):
                    logger.debug(f"Watchmode: '{title}' ({year}) → id {r['id']}")
                    _cache_set(cache_key, r["id"])
                    return r["id"]
        title_id = results[0]["id"]
        logger.debug(f"Watchmode: '{title}' → id {title_id} (first result, year={results[0].get('year')})")
        _cache_set(cache_key, title_id)
        return title_id
    except Exception as e:
        logger.warning(f"Watchmode: search failed for '{title}': {e}")
        return None  # don't cache failures — allow retry


def fetch_providers(title_id: int, country: str = "US") -> list[dict]:
    """Fetch streaming providers for a Watchmode title.

    Returns a deduplicated list of providers with name, logo URL, and availability type.
    Includes subscription, free (ad-supported), rent, and buy sources.
    Excludes TV Everywhere (tve) sources which require a cable subscription to activate.
    When a provider appears under multiple types, the best type wins: sub > free > rent > buy.
    Results are cached for 24 hours to conserve the free-tier API budget.
    """
    cache_key = f"providers:{title_id}"
    hit, cached = _cache_get(cache_key)
    if hit:
        return cached

    _load_source_logos()
    # Increment under _cache_lock so all counter mutations are serialized by one lock.
    with _cache_lock:
        global _api_calls
        _api_calls += 1
    logger.info(f"[watchmode_api_call] providers title_id={title_id}")
    try:
        response = requests.get(
            WATCHMODE_BASE_URL + f"/title/{title_id}/sources/",
            params={"apiKey": WATCHMODE_API_KEY, "regions": country},
            timeout=10,
        )
        response.raise_for_status()
        sources = response.json()
        logger.debug(f"Watchmode: raw sources for title {title_id}: {[(s.get('name'), s.get('type'), s.get('region')) for s in sources]}")

        type_priority = {"sub": 0, "free": 1, "rent": 2, "buy": 3}
        # best entry per provider name: keeps highest-priority type
        best: dict[str, dict] = {}
        for s in sources:
            stype = s.get("type", "")
            if stype == "tve":
                continue
            name = s["name"]
            if name not in best or type_priority.get(stype, 99) < type_priority.get(best[name]["type"], 99):
                best[name] = {
                    "name": name,
                    "type": stype,
                    "logo": _source_logos.get(s["source_id"]),
                }
        providers = list(best.values())
        _cache_set(cache_key, providers)
        return providers
    except Exception as e:
        logger.warning(f"Watchmode: sources failed for title {title_id}: {e}")
        return []  # don't cache failures — allow retry
