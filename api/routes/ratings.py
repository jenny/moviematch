"""
/ratings endpoints — critic/audience review scores for the "Ratings & Scores" overlay.

Combines two sources, mirroring how /streaming combines Watchmode + TMDB:
  - OMDb  → Rotten Tomatoes (critic), IMDb, Metacritic scores (+ imdbID for deep links)
  - TMDB  → the TMDB user score (vote_average) + its id, for a themoviedb.org deep link

Each present provider is returned as {provider, score, url} in a fixed display order
(RT → IMDb → Metacritic → TMDB). Providers with no score are simply omitted, so an
empty list means "no scores anywhere" (rendered as a muted note on the frontend).

Deep links: IMDb (via imdbID) and TMDB (via id) link straight to the film's page.
Rotten Tomatoes and Metacritic expose no stable per-title id in our data, so their
glyphs link to the provider's search results for the title — always valid, never a 404.
"""

import asyncio
import logging
from urllib.parse import quote, quote_plus

from fastapi import APIRouter, Request
from pydantic import BaseModel, field_validator

from api.limiter import limiter
from config import RATINGS_RATE_LIMIT
from tmdb import fetch_movie_rating
import omdb

logger = logging.getLogger(__name__)
router = APIRouter()

# Caps concurrent upstream calls across a batch (each title = up to 1 OMDb + 1 TMDB call),
# matching the streaming batch's protection against per-second rate limits.
_batch_semaphore = asyncio.Semaphore(3)


def get_scores_for_title(title: str, year: str = "") -> list[dict]:
    """Return an ordered list of {provider, score, url} for one movie.

    Order is fixed: Rotten Tomatoes, IMDb, Metacritic, TMDB. Absent scores are omitted.
    Never raises — OMDb and TMDB helpers both fail soft to {}/None.
    """
    omdb_data = omdb.fetch_ratings(title, year)   # {} when key unset / not found / error
    tmdb_data = fetch_movie_rating(title, year)   # None when not found / error

    scores: list[dict] = []

    # Rotten Tomatoes (critic Tomatometer) — no per-title id, link to provider search.
    if omdb_data.get("rt") is not None:
        scores.append({
            "provider": "rt",
            "score": f"{omdb_data['rt']}%",
            "url": f"https://www.rottentomatoes.com/search?search={quote_plus(title)}",
        })

    # IMDb — deep link via imdbID when present, otherwise fall back to IMDb search.
    if omdb_data.get("imdb") is not None:
        imdb_id = omdb_data.get("imdb_id")
        url = (
            f"https://www.imdb.com/title/{imdb_id}/"
            if imdb_id
            else f"https://www.imdb.com/find/?q={quote_plus(title)}"
        )
        scores.append({"provider": "imdb", "score": f"{omdb_data['imdb']}", "url": url})

    # Metacritic — no per-title id, link to provider search.
    if omdb_data.get("metacritic") is not None:
        scores.append({
            "provider": "metacritic",
            "score": f"{omdb_data['metacritic']}",
            "url": f"https://www.metacritic.com/search/{quote(title)}/",
        })

    # TMDB user score — deep link via the movie id. vote_average of 0 means "no votes",
    # which we treat as "no score" rather than showing a misleading 0.0.
    if tmdb_data and tmdb_data.get("vote_average"):
        scores.append({
            "provider": "tmdb",
            "score": f"{round(tmdb_data['vote_average'], 1)}",
            "url": f"https://www.themoviedb.org/movie/{tmdb_data['id']}",
        })

    logger.debug("[ratings] title=%r year=%r → %d score(s)", title, year, len(scores))
    return scores


@router.get("/ratings")
@limiter.limit(RATINGS_RATE_LIMIT)
def ratings(request: Request, title: str, year: str = ""):
    """Return critic/audience review scores for a single movie."""
    return {"scores": get_scores_for_title(title, year)}


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


@router.post("/ratings/batch")
@limiter.limit(RATINGS_RATE_LIMIT)
async def ratings_batch(request: Request, body: BatchRequest):
    """Batch score lookup for multiple movies.

    Runs lookups concurrently (capped at 3 simultaneous upstream calls) so the
    post-stream prefetch resolves quickly without tripping OMDb/TMDB rate limits.
    Returns results in input order. Scores are region-independent, so — unlike
    /streaming/batch — there's no country parameter.
    """
    async def lookup(item: TitleRequest) -> dict:
        async with _batch_semaphore:
            scores = await asyncio.to_thread(get_scores_for_title, item.title, item.year)
        return {"title": item.title, "year": item.year, "scores": scores}

    results = await asyncio.gather(*[lookup(item) for item in body.titles])
    return {"results": results}
