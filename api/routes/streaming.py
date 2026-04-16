import asyncio
import logging
import threading
import urllib.parse

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

from config import WATCHMODE_API_KEY
from tmdb import fetch_certification, fetch_watch_providers, search_movie_by_title
import watchmode

logger = logging.getLogger(__name__)
router = APIRouter()

# Only allow logo URLs from known CDN hostnames to prevent SSRF via the proxy.
_ALLOWED_LOGO_HOSTS = frozenset({"cdn.watchmode.com", "image.tmdb.org"})

# In-memory logo image cache: CDN URL → (image bytes, content-type).
# Populated lazily on first proxy request; survives the process lifetime so
# later requests are served from memory even if the upstream CDN is unavailable.
_logo_image_cache: dict[str, tuple[bytes, str]] = {}
_logo_image_cache_lock = threading.Lock()


def _proxy_logo_url(logo: str | None) -> str | None:
    """Rewrite a CDN logo URL to our /logo-proxy endpoint.

    Returns None unchanged. The proxy adds a 30-day Cache-Control header and
    caches image bytes server-side, so the browser holds a long-lived copy and
    CDN downtime doesn't break already-loaded logos.
    """
    if not logo:
        return None
    return "/logo-proxy?url=" + urllib.parse.quote(logo, safe="")


# Module-level semaphore shared across all batch requests — caps concurrent Watchmode
# calls to 3 to avoid hitting per-second rate limits on the free tier.
_batch_semaphore = asyncio.Semaphore(3)


def get_providers_for_title(title: str, year: str) -> list[dict]:
    """Return US streaming providers for one movie.

    Uses Watchmode as the primary source when WATCHMODE_API_KEY is configured —
    it has significantly better coverage than TMDB's watch provider data.
    Falls back to TMDB when the key is absent.

    Logo URLs are rewritten to go through our /logo-proxy endpoint, which caches
    image bytes server-side and serves them with a 30-day Cache-Control header.
    """
    if WATCHMODE_API_KEY:
        title_id = watchmode.search_title(title, year)
        if title_id is None:
            logger.info(f"Watchmode: '{title}' ({year}) not found")
            return []
        logger.info(f"Watchmode: '{title}' ({year}) → title_id={title_id}")
        providers = watchmode.fetch_providers(title_id)
        logger.info(f"Watchmode: {len(providers)} provider(s) for '{title}': {[p['name'] for p in providers]}")
        return [{**p, "logo": _proxy_logo_url(p.get("logo"))} for p in providers]

    # Fallback: TMDB (incomplete coverage — set WATCHMODE_API_KEY for better results)
    logger.debug(f"WATCHMODE_API_KEY not set, falling back to TMDB for '{title}'")
    movie_id = search_movie_by_title(title, year)
    if movie_id is None:
        return []
    providers = fetch_watch_providers(movie_id)
    return [{**p, "logo": _proxy_logo_url(p.get("logo"))} for p in providers]


def get_certification_for_title(title: str, year: str) -> str:
    """Return MPAA certification (e.g. 'PG-13') for a movie, or '' if unavailable."""
    movie_id = search_movie_by_title(title, year)
    if movie_id is None:
        return ""
    return fetch_certification(movie_id)


@router.get("/logo-proxy")
def logo_proxy(url: str) -> Response:
    """Proxy a provider logo image with a 30-day browser cache.

    Fetches from the upstream CDN on first request and stores the image bytes
    in memory. Subsequent requests are served from the in-memory cache without
    hitting the CDN, so logos keep working even if the CDN has an outage.
    Only proxies URLs from known CDN hostnames to prevent SSRF attacks.
    """
    # SSRF guard: reject URLs from any host not in the allowlist.
    try:
        host = urllib.parse.urlparse(url).hostname
    except Exception:
        host = None
    if host not in _ALLOWED_LOGO_HOSTS:
        raise HTTPException(status_code=400, detail="Logo URL not allowed")

    with _logo_image_cache_lock:
        cached = _logo_image_cache.get(url)

    if cached:
        content, content_type = cached
    else:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            content = r.content
            content_type = r.headers.get("Content-Type", "image/png")
            with _logo_image_cache_lock:
                _logo_image_cache[url] = (content, content_type)
        except Exception as e:
            logger.warning(f"Logo proxy: failed to fetch {url!r}: {e}")
            raise HTTPException(status_code=502, detail="Logo unavailable")

    return Response(
        content=content,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=2592000, immutable"},
    )


@router.get("/streaming")
def streaming_providers(title: str, year: str = ""):
    """Return US streaming providers and content rating for a movie."""
    return {
        "providers": get_providers_for_title(title, year),
        "certification": get_certification_for_title(title, year),
    }


class TitleRequest(BaseModel):
    title: str
    year: str = ""


class BatchRequest(BaseModel):
    titles: list[TitleRequest]

    @field_validator("titles")
    @classmethod
    def max_ten_titles(cls, v: list) -> list:
        if len(v) > 10:
            raise ValueError("Maximum 10 titles per batch request")
        return v


@router.post("/streaming/batch")
async def streaming_providers_batch(request: BatchRequest):
    """Batch lookup of streaming providers for multiple movies.

    Runs lookups concurrently (capped at 3 simultaneous Watchmode calls) to
    avoid hammering the free-tier rate limit. Returns results in input order.
    """
    async def lookup(item: TitleRequest) -> dict:
        async with _batch_semaphore:
            providers = await asyncio.to_thread(
                get_providers_for_title, item.title, item.year
            )
            certification = await asyncio.to_thread(
                get_certification_for_title, item.title, item.year
            )
        return {
            "title": item.title,
            "year": item.year,
            "providers": providers,
            "certification": certification,
        }

    results = await asyncio.gather(*[lookup(item) for item in request.titles])
    return {"results": results}
