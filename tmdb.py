import os
import json
import time
import math
import threading

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import (
    TMDB_BASE_URL, TMDB_HEADERS, TMDB_KEY,
    TMDB_MIN_VOTE_COUNT, TMDB_RATE_LIMIT_SLEEP,
    SCORE_WEIGHT_RATING, SCORE_WEIGHT_POPULARITY,
    DATA_DIR, CAST_LIMIT, CREW_JOBS
)

TMDB_MAX_PAGE = 500
_index_lock = threading.Lock()


def _is_rate_limit_or_server_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code in (429, 500, 502, 503, 504)
    )


def _require_tmdb_key():
    if not TMDB_KEY:
        raise ValueError("TMDB_READ_ACCESS_TOKEN is not set. Check your .env file.")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception(_is_rate_limit_or_server_error),
)
def fetch_discover_page(page: int, sort_by: str = "vote_average.desc") -> list[dict]:
    response = requests.get(
        TMDB_BASE_URL + "/discover/movie",
        params={
            "sort_by": sort_by,
            "vote_count.gte": str(TMDB_MIN_VOTE_COUNT),
            "page": str(page)
        },
        headers=TMDB_HEADERS
    )
    response.raise_for_status()
    # Sleep after a successful response to stay within TMDB's rate limit
    time.sleep(TMDB_RATE_LIMIT_SLEEP)
    return response.json()["results"]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception(_is_rate_limit_or_server_error),
)
def fetch_movie_detail(movie_id: int) -> dict:
    response = requests.get(
        TMDB_BASE_URL + f"/movie/{movie_id}?append_to_response=keywords,credits",
        headers=TMDB_HEADERS
    )
    response.raise_for_status()
    # Sleep after a successful response to stay within TMDB's rate limit
    time.sleep(TMDB_RATE_LIMIT_SLEEP)
    return response.json()


def filter_cast(movie_json: dict) -> dict:
    movie_json["credits"]["cast"] = movie_json["credits"]["cast"][:CAST_LIMIT]
    return movie_json


def filter_crew(movie_json: dict) -> dict:
    crew = movie_json["credits"]["crew"]
    filtered = [m for m in crew if m["job"] in CREW_JOBS]
    movie_json["credits"]["crew"] = sorted(filtered, key=lambda c: c["job"])
    return movie_json


def collect_candidates(pages_per_sort: int) -> dict[int, dict]:
    """Fetch candidate movies from multiple sort criteria to ensure diversity."""
    sort_criteria = ["vote_average.desc", "popularity.desc", "revenue.desc"]
    candidates = {}
    for sort_by in sort_criteria:
        for page in range(1, pages_per_sort + 1):
            for movie in fetch_discover_page(page, sort_by):
                # First-seen wins: preserve the result from the highest-priority sort
                candidates.setdefault(movie["id"], movie)
            print(f"  [{sort_by}] page {page}/{pages_per_sort} — {len(candidates)} unique candidates so far")
    return candidates


def _composite_score(movie: dict, mean_rating: float, log_max_pop: float) -> float:
    """Score using Bayesian weighted rating (60%) + log-normalized popularity (40%).

    Popularity follows a power-law distribution, so log normalization prevents a
    handful of viral blockbusters from compressing all other scores toward zero.
    Weights are tunable via SCORE_WEIGHT_RATING / SCORE_WEIGHT_POPULARITY in config.
    """
    v = movie.get("vote_count", 0)
    R = movie.get("vote_average", 0)
    p = movie.get("popularity", 0)
    m = TMDB_MIN_VOTE_COUNT
    wr = (v / (v + m)) * R + (m / (v + m)) * mean_rating
    norm_pop = math.log1p(p) / log_max_pop if log_max_pop else 0
    return SCORE_WEIGHT_RATING * (wr / 10.0) + SCORE_WEIGHT_POPULARITY * norm_pop


def select_top_n(candidates: dict[int, dict], n: int) -> list[int]:
    """Rank candidates by composite score and return top n IDs."""
    movies = list(candidates.values())
    mean_rating = sum(m.get("vote_average", 0) for m in movies) / len(movies)
    log_max_pop = math.log1p(max((m.get("popularity", 0) for m in movies), default=0))
    ranked = sorted(movies, key=lambda m: _composite_score(m, mean_rating, log_max_pop), reverse=True)
    return [m["id"] for m in ranked[:n]]


def ingest_movie(movie_id: int) -> dict:
    _require_tmdb_key()
    file_path = os.path.join(DATA_DIR, f"{movie_id}.json")
    if os.path.exists(file_path):
        with open(file_path) as f:
            cached = json.load(f)
        print(f"Skipping {cached.get('title', movie_id)} (already ingested)")
        return cached
    movie_json = fetch_movie_detail(movie_id)
    movie_json = filter_cast(movie_json)
    movie_json = filter_crew(movie_json)
    with open(file_path, "w") as f:
        json.dump(movie_json, f, indent=2)
    print(f"Ingested {movie_json['title']} → {movie_id}.json")
    return movie_json


def ingest_index(n: int) -> list[int]:
    _require_tmdb_key()
    # TMDB discover caps at page 500 (10,000 results per sort criterion)
    pages_per_sort = min(math.ceil(n / 20), TMDB_MAX_PAGE)
    print(f"Collecting candidates ({pages_per_sort} pages × 3 sort criteria)...")
    candidates = collect_candidates(pages_per_sort)
    print(f"Collected {len(candidates)} unique candidates, ranking top {n}...")
    ids = select_top_n(candidates, n)
    if len(ids) < n:
        print(f"Warning: only {len(ids)} unique candidates found (requested {n}). "
              "Consider lowering TMDB_MIN_VOTE_COUNT or requesting fewer movies.")
    index = {"results": [{"id": m_id, "title": candidates[m_id]["title"]} for m_id in ids]}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Indexed {len(ids)} movies → data/index.json")
    return ids


def update_index(movie_id: int, title: str) -> None:
    """Thread-safely append a movie to index.json if not already present."""
    index_path = os.path.join(DATA_DIR, "index.json")
    with _index_lock:
        with open(index_path, "r") as f:
            index = json.load(f)
        if any(m["id"] == movie_id for m in index["results"]):
            return
        index["results"].append({"id": movie_id, "title": title})
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    print(f"Updated index.json → added {title} ({movie_id})")


if __name__ == "__main__":
    n = int(input("How many movies to index? "))
    ingest_index(n)
