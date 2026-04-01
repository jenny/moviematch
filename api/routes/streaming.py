import logging

from fastapi import APIRouter

from config import WATCHMODE_API_KEY
from tmdb import search_movie_by_title, fetch_watch_providers
import watchmode

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/streaming")
def streaming_providers(title: str, year: str = ""):
    """Return US streaming providers for a movie.

    Uses Watchmode as the primary source when WATCHMODE_API_KEY is configured —
    it has significantly better coverage than TMDB's watch provider data.
    Falls back to TMDB when the key is absent.
    """
    if WATCHMODE_API_KEY:
        title_id = watchmode.search_title(title, year)
        if title_id is None:
            logger.info(f"Watchmode: '{title}' ({year}) not found")
            return {"providers": []}
        logger.info(f"Watchmode: '{title}' ({year}) → title_id={title_id}")
        providers = watchmode.fetch_providers(title_id)
        logger.info(f"Watchmode: {len(providers)} provider(s) for '{title}': {[p['name'] for p in providers]}")
        return {"providers": providers}

    # Fallback: TMDB (incomplete coverage — set WATCHMODE_API_KEY for better results)
    logger.debug(f"WATCHMODE_API_KEY not set, falling back to TMDB for '{title}'")
    movie_id = search_movie_by_title(title, year)
    if movie_id is None:
        return {"providers": []}
    return {"providers": fetch_watch_providers(movie_id)}
