"""
Watchmode API integration for streaming availability data.

Watchmode (watchmode.com) provides comprehensive, up-to-date streaming provider data
for the US and other regions. It's used as the primary source for the /streaming endpoint,
with TMDB as a fallback when WATCHMODE_API_KEY is not configured.

Free tier: 1,000 requests/month. Each /streaming call uses 2 requests (search + sources),
plus one shared request for the source logo catalog (cached in memory for the process lifetime).
Every non-cached Watchmode API call is logged with the tag [watchmode_api_call] for grep-based
monitoring of monthly budget consumption.

Watchmode authenticates by query parameter (apiKey=...), so a raw requests exception would
print the key — requests embeds the full request URL in HTTPError messages. Every failure
log therefore passes the exception through logger.redact().

API call counts are persisted to LOG_DIR/watchmode_calls.json so they survive process
restarts and Railway deploys. Counts are stored keyed by calendar month ("YYYY-MM"), so a
month rollover requires no detection and no reset — the new month is simply a new key.
This is deliberate: an earlier design kept a single count plus a "which month is this?"
label and only re-checked that label at process start, so a long-lived container (Railway
runs for weeks) carried the previous month's total forward and then re-stamped it with the
new month on the next write, corrupting the file permanently.
"""

import datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any

import requests

from config import LOG_DIR, WATCHMODE_API_KEY
from logger import redact

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
_api_calls = 0    # real Watchmode HTTP requests made this session
_cache_hits = 0   # lookups served from cache (API calls saved)

# Persistent per-month counters — survive process restarts and Railway deploys.
# Backed by LOG_DIR/watchmode_calls.json (same volume as search.log in production).
_COUNTER_FILE = os.path.join(LOG_DIR, "watchmode_calls.json")
_counter_lock = threading.Lock()

# {"YYYY-MM": count} — the month is the key, never a mutable label on a shared count.
# A rollover is a non-event: the first call in a new month creates a new key and the
# previous month's total is left untouched (and kept, for month-over-month history).
_counts: dict[str, int] = {}

# How many months of history to retain on disk. Bounds the file at a trivial size while
# leaving enough for a year-over-year glance in the admin panel.
_COUNTER_HISTORY_MONTHS = 12

# The only key shape _counts may hold. Enforced on load — see _load_persistent_counter.
_MONTH_KEY_RE = re.compile(r"\d{4}-\d{2}")


def _get_current_month() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


def _prune_history(protect: str | None = None) -> None:
    """Drop all but the most recent _COUNTER_HISTORY_MONTHS entries.

    Month keys are "YYYY-MM", so lexicographic sort is chronological. Caller must
    hold _counter_lock (or be running single-threaded at import).

    `protect` is never evicted. The month being counted has to survive even when it
    sorts oldest of the set — a clock rollback, or a file seeded with future-dated
    keys — or the call just recorded would be discarded the instant it was written,
    and the panel would show zero while real quota was being spent.
    """
    excess = len(_counts) - _COUNTER_HISTORY_MONTHS
    if excess <= 0:
        return
    evictable = [month for month in sorted(_counts) if month != protect]
    for month in evictable[:excess]:
        del _counts[month]


def _load_persistent_counter() -> None:
    """Load per-month API call counts from disk.

    Accepts the legacy single-count format ({"month": ..., "count": ...}) and migrates
    it into the keyed form, so an existing deployment's file isn't silently discarded.
    Note the legacy file may itself hold a corrupted total (see module docstring) —
    migration preserves whatever is there rather than guessing; delete the file to zero it.
    """
    global _counts
    try:
        with open(_COUNTER_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        _counts = {}  # no file yet — start empty
        return
    except (ValueError, OSError) as e:
        logger.warning(f"Watchmode: could not read call counter file, starting empty: {e}")
        _counts = {}
        return

    if not isinstance(data, dict):
        logger.warning("Watchmode: call counter file has unexpected shape, starting empty")
        _counts = {}
        return

    if "month" in data and "count" in data:
        # Legacy format — one count plus a month label. Migrate to the keyed form.
        # The month is validated exactly as the keyed branch validates its keys: this
        # path writes straight into _counts, so without the check a malformed label
        # would seed the same junk key the keyed branch refuses (and, sorting last,
        # would be painted as the current month by the panel).
        legacy_month = str(data["month"])
        if not _MONTH_KEY_RE.fullmatch(legacy_month):
            logger.warning(
                f"Watchmode: legacy call counter has a non-month label "
                f"{data['month']!r}, starting empty"
            )
            _counts = {}
            return
        try:
            _counts = {legacy_month: int(data["count"])}
            logger.info(
                f"Watchmode: migrated legacy call counter "
                f"({legacy_month}={data['count']}) to per-month format"
            )
        except (TypeError, ValueError):
            logger.warning("Watchmode: legacy call counter unparseable, starting empty")
            _counts = {}
        return

    # Keyed format — keep only well-formed "YYYY-MM": int entries. The *key* is checked
    # as strictly as the value: letters sort after digits, so a stray non-date key would
    # sort last in get_stats()'s ordering, and the panel treats the last entry as the
    # current month — putting the severity colour and "so far" marker on a junk row while
    # the real current month rendered as history. _prune_history would also retain it
    # forever, evicting real months to keep it.
    loaded = {}
    for month, count in data.items():
        if not _MONTH_KEY_RE.fullmatch(str(month)):
            logger.warning(f"Watchmode: skipping counter entry with non-month key {month!r}")
            continue
        try:
            loaded[str(month)] = int(count)
        except (TypeError, ValueError):
            logger.warning(f"Watchmode: skipping malformed counter entry {month!r}")
    _counts = loaded
    _prune_history()


def _persist_counter() -> None:
    """Write per-month API call counts to disk, atomically.

    Called under _counter_lock after each real API request. Fails soft so a read-only
    filesystem (e.g. no volume mounted) doesn't break the app — the counter is a
    monitoring aid, not something worth failing a user request over.

    Written to a temp file and moved into place with os.replace(), which is atomic on
    POSIX. Writing in place would truncate first, so a SIGTERM or crash inside that
    window (Railway redeploys mid-traffic, and this runs on every real API call) would
    leave a half-written file that fails to parse on next load. That used to cost one
    number; now it would discard the whole retained history, which cannot be rebuilt.
    """
    tmp_path = _COUNTER_FILE + ".tmp"
    try:
        os.makedirs(os.path.dirname(_COUNTER_FILE) or ".", exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(_counts, f)
        os.replace(tmp_path, _COUNTER_FILE)
    except Exception as e:
        logger.warning(f"Watchmode: failed to persist call counters: {e}")
        # Don't leave a partial temp file behind to be retried or mistaken for data.
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _increment_api_calls() -> None:
    """Increment the session counter and the current month's counter, then persist.

    The current month is resolved at write time, so a process alive across a month
    boundary starts a fresh key on its very next call — no rollover check needed.
    """
    global _api_calls
    with _counter_lock:
        _api_calls += 1
        month = _get_current_month()
        if month not in _counts:
            # First call of a new month (or first ever). Log it: this is the moment the
            # budget window resets, which is worth being able to find in the logs.
            logger.info(f"Watchmode: starting new monthly counter window {month}")
            _counts[month] = 0
        _counts[month] += 1
        # Prune *after* the increment, never between creating the key and using it:
        # if `month` sorted oldest of the resulting set (a clock rollback, or a file
        # seeded with future-dated keys) an earlier prune would delete the key the
        # next line touches, raising KeyError inside the request path. Passing
        # `protect` covers the other half — otherwise prune would simply discard the
        # count we just recorded instead of crashing on it.
        _prune_history(protect=month)
        _persist_counter()


# Load persistent counter once at module import time.
_load_persistent_counter()


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
    with _counter_lock:
        # Resolved at read time, so the admin panel shows the new month as 0 immediately
        # after a rollover even if no API call has been made yet.
        calls_month = _counts.get(_get_current_month(), 0)
        calls_session = _api_calls
    return {
        "api_calls_month": calls_month,    # persists across deploys (primary budget metric)
        "api_calls_session": calls_session, # resets on process restart (useful for debugging)
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
            _increment_api_calls()
            logger.info("[watchmode_api_call] sources catalog")
            response = requests.get(
                WATCHMODE_BASE_URL + "/sources/",
                params={"apiKey": WATCHMODE_API_KEY},
                timeout=10,
            )
            response.raise_for_status()
            for source in response.json():
                logo = source.get("logo_100px")
                # Watchmode sometimes returns a URL whose filename is the
                # literal string "null" (e.g. ".../provider_logos/null") when
                # a logo is missing. Reject those alongside Python None.
                if logo and not logo.rstrip("/").endswith("null"):
                    _source_logos[source["id"]] = logo
            _source_logos_loaded = True
            logger.info(f"Watchmode: cached {len(_source_logos)} source logos")
        except Exception as e:
            logger.warning(f"Watchmode: failed to load source logos: {redact(e)}")


def _clean_web_url(value: Any) -> str | None:
    """Return value if it's a usable http(s) deep link, else None.

    Watchmode puts prose where a URL should be on the free tier — ios_url/android_url
    literally read "Deeplinks available for paid plans only." web_url is a real URL today,
    but this guard keeps any such placeholder (or a javascript:/data: value) from ever
    reaching an href in the overlay.
    """
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    return None


def _clean_price(value: Any) -> float | None:
    """Return value as a float when Watchmode supplied a real price, else None.

    Subscription and free sources carry price: null; rent/buy rows carry a number. A
    provider with no price still renders — the chip just omits the amount.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    _increment_api_calls()
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
        logger.warning(f"Watchmode: search failed for '{title}': {redact(e)}")
        return None  # don't cache failures — allow retry


def fetch_providers(title_id: int, country: str = "US") -> list[dict]:
    """Fetch streaming providers for a Watchmode title.

    Returns a deduplicated list of providers with name, logo URL, availability type,
    a deep link to the title on that provider (``url``), and the rent/buy ``price``.
    Includes subscription, free (ad-supported), rent, and buy sources.
    Excludes TV Everywhere (tve) sources which require a cable subscription to activate.
    When a provider appears under multiple types, the best type wins: sub > free > rent > buy;
    within the same type the cheapest format wins (see the dedupe loop below).
    Results are cached for 24 hours to conserve the free-tier API budget.

    NOTE: only ``web_url`` is usable — Watchmode's ``ios_url``/``android_url`` return the
    literal string "Deeplinks available for paid plans only." on the free tier. That costs us
    nothing in practice: the major services' web URLs are registered universal links, so on
    iOS/Android they open the installed app at the title anyway.
    """
    cache_key = f"providers:{title_id}:{country}"
    hit, cached = _cache_get(cache_key)
    if hit:
        return cached

    _load_source_logos()
    _increment_api_calls()
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
        # Best entry per provider name. Watchmode emits one row per *format* (SD/HD/4K),
        # each with its own price, so a provider legitimately appears several times at the
        # same type — hence the two-stage merge: better type replaces outright, equal type
        # keeps the cheaper row and backfills anything the incumbent was missing.
        best: dict[str, dict] = {}
        for s in sources:
            stype = s.get("type", "")
            if stype == "tve":
                continue
            name = s["name"]
            entry = {
                "name": name,
                "type": stype,
                "logo": _source_logos.get(s["source_id"]),
                "url": _clean_web_url(s.get("web_url")),
                "price": _clean_price(s.get("price")),
            }
            current = best.get(name)
            if current is None:
                best[name] = entry
                continue

            new_rank = type_priority.get(stype, 99)
            cur_rank = type_priority.get(current["type"], 99)
            if new_rank < cur_rank:
                best[name] = entry
            elif new_rank == cur_rank:
                # Same availability type, different format — show the cheapest.
                if entry["price"] is not None and (
                    current["price"] is None or entry["price"] < current["price"]
                ):
                    current["price"] = entry["price"]
                # A row missing a link/logo shouldn't erase one we already found.
                current["url"] = current["url"] or entry["url"]
                current["logo"] = current["logo"] or entry["logo"]
        providers = list(best.values())
        _cache_set(cache_key, providers)
        return providers
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            # Watchmode returns 400 instead of an empty list when a title has no
            # sources in the requested region, or when the region isn't enabled on
            # the account. Either way it's an expected, handled outcome.
            logger.debug(
                "Watchmode: no sources for title_id=%d in region %s (400 — region not available for this title/account)",
                title_id, country,
            )
        else:
            logger.warning("Watchmode: sources failed for title %d: %s", title_id, redact(e))
        return []  # don't cache failures — allow retry
    except Exception as e:
        logger.warning("Watchmode: sources failed for title %d: %s", title_id, redact(e))
        return []  # don't cache failures — allow retry
