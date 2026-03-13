import os
import json
import time
import math

import requests

from config import (
    TMDB_BASE_URL, TMDB_HEADERS, TMDB_KEY,
    TMDB_MIN_VOTE_COUNT, TMDB_RATE_LIMIT_SLEEP,
    DATA_DIR, CAST_LIMIT, CREW_JOBS
)

def _require_tmdb_key():
    if not TMDB_KEY:
        raise ValueError("TMDB_READ_ACCESS_TOKEN is not set. Check your .env file.")


def fetch_discover_page(page: int) -> list[dict]:
    _require_tmdb_key()
    response = requests.get(
        TMDB_BASE_URL + "/discover/movie",
        params={
            "sort_by": "vote_average.desc",
            "vote_count.gte": str(TMDB_MIN_VOTE_COUNT),
            "page": str(page)
        },
        headers=TMDB_HEADERS
    )
    response.raise_for_status()
    return response.json()["results"]


def fetch_movie_detail(movie_id: int) -> dict:
    _require_tmdb_key()
    response = requests.get(
        TMDB_BASE_URL + f"/movie/{movie_id}?append_to_response=keywords,credits",
        headers=TMDB_HEADERS
    )
    response.raise_for_status()
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


def ingest_movie(movie_id: int) -> dict:
    movie_json = fetch_movie_detail(movie_id)
    movie_json = filter_cast(movie_json)
    movie_json = filter_crew(movie_json)
    file_path = os.path.join(DATA_DIR, f"{movie_id}.json")
    with open(file_path, "w") as f:
        json.dump(movie_json, f, indent=2)
    print(f"Ingested {movie_json['title']} → {movie_id}.json")
    return movie_json


def ingest_index(n: int) -> list[int]:
    pages = math.ceil(n / 20)
    ids = []
    index = {"results": []}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "index.json"), "w") as f:
        for page in range(1, pages + 1):
            for movie in fetch_discover_page(page):
                if len(ids) < n:
                    ids.append(movie["id"])
                    index["results"].append({"id": movie["id"], "title": movie["title"]})
        json.dump(index, f, indent=2)
    print(f"Indexed {len(ids)} movies → data/index.json")
    return ids


if __name__ == "__main__":
    n = int(input("How many movies to index? "))
    ingest_index(n)
