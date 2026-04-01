"""
Watchmode API integration for streaming availability data.

Watchmode (watchmode.com) provides comprehensive, up-to-date streaming provider data
for the US and other regions. It's used as the primary source for the /streaming endpoint,
with TMDB as a fallback when WATCHMODE_API_KEY is not configured.

Free tier: 1,000 requests/month. Each /streaming call uses 2 requests (search + sources),
plus one shared request for the source logo catalog (cached in memory for the process lifetime).
"""

import logging
import threading

import requests

from config import WATCHMODE_API_KEY

logger = logging.getLogger(__name__)

WATCHMODE_BASE_URL = "https://api.watchmode.com/v1"

# In-memory cache: {source_id: logo_url} — populated once from the Watchmode sources catalog
_source_logos: dict[int, str] = {}
_source_logos_lock = threading.Lock()
_source_logos_loaded = False


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
    """
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
            return None
        logger.debug(f"Watchmode: search results for '{title}': {[(r.get('id'), r.get('year')) for r in results[:5]]}")
        # Prefer an exact year match when the caller provides one
        if year:
            for r in results:
                if str(r.get("year", "")) == str(year):
                    logger.debug(f"Watchmode: '{title}' ({year}) → id {r['id']}")
                    return r["id"]
        logger.debug(f"Watchmode: '{title}' → id {results[0]['id']} (first result, year={results[0].get('year')})")
        return results[0]["id"]
    except Exception as e:
        logger.warning(f"Watchmode: search failed for '{title}': {e}")
        return None


def fetch_providers(title_id: int, country: str = "US") -> list[dict]:
    """Fetch streaming providers for a Watchmode title.

    Returns a deduplicated list of providers with name, logo URL, and availability type.
    Includes subscription, free (ad-supported), rent, and buy sources.
    Excludes TV Everywhere (tve) sources which require a cable subscription to activate.
    When a provider appears under multiple types, the best type wins: sub > free > rent > buy.
    """
    _load_source_logos()
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
        return list(best.values())
    except Exception as e:
        logger.warning(f"Watchmode: sources failed for title {title_id}: {e}")
        return []
