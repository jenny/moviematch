import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from config import WATCHMODE_API_KEY
from tmdb import fetch_certification, fetch_watch_providers, search_movie_by_title
import watchmode

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level semaphore shared across all batch requests — caps concurrent Watchmode
# calls to 3 to avoid hitting per-second rate limits on the free tier.
_batch_semaphore = asyncio.Semaphore(3)


def get_providers_for_title(title: str, year: str) -> list[dict]:
    """Return US streaming providers for one movie.

    Uses Watchmode as the primary source when WATCHMODE_API_KEY is configured —
    it has significantly better coverage than TMDB's watch provider data.
    Falls back to TMDB when the key is absent.
    """
    if WATCHMODE_API_KEY:
        title_id = watchmode.search_title(title, year)
        if title_id is None:
            logger.info(f"Watchmode: '{title}' ({year}) not found")
            return []
        logger.info(f"Watchmode: '{title}' ({year}) → title_id={title_id}")
        providers = watchmode.fetch_providers(title_id)
        logger.info(f"Watchmode: {len(providers)} provider(s) for '{title}': {[p['name'] for p in providers]}")
        return providers

    # Fallback: TMDB (incomplete coverage — set WATCHMODE_API_KEY for better results)
    logger.debug(f"WATCHMODE_API_KEY not set, falling back to TMDB for '{title}'")
    movie_id = search_movie_by_title(title, year)
    if movie_id is None:
        return []
    return fetch_watch_providers(movie_id)


def get_certification_for_title(title: str, year: str) -> str:
    """Return MPAA certification (e.g. 'PG-13') for a movie, or '' if unavailable."""
    movie_id = search_movie_by_title(title, year)
    if movie_id is None:
        return ""
    return fetch_certification(movie_id)


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
