import json
import logging
import os

from config import DATA_DIR, MIN_INGEST_VOTE_AVERAGE, MIN_INGEST_VOTE_COUNT

logger = logging.getLogger(__name__)
from tmdb import ingest_index, ingest_movie, update_index
from richtext import compile_all_richtexts, build_richtext
from embeddings import initialize_all_embeddings, upsert_movie


def ingest_single(movie_id: int, vote_average: float, vote_count: int) -> bool:
    """Quality-gate, ingest, embed, and index a single movie. Returns True if added.

    Intended for lazy ingestion of tool-discovered movies. Safe to call concurrently —
    index.json writes are serialized via a lock in update_index().
    """
    if vote_average < MIN_INGEST_VOTE_AVERAGE or vote_count < MIN_INGEST_VOTE_COUNT:
        logger.debug(f"Skipping movie {movie_id}: below quality threshold "
                     f"(rating={vote_average}, votes={vote_count})")
        return False

    file_path = os.path.join(DATA_DIR, f"{movie_id}.json")
    if os.path.exists(file_path):
        with open(file_path) as f:
            existing = json.load(f)
        if "richtext" in existing:
            logger.debug(f"Skipping {existing.get('title', movie_id)}: already in dataset")
            return False

    movie_json = ingest_movie(movie_id)
    movie_json["richtext"] = build_richtext(movie_json)
    with open(file_path, "w") as f:
        json.dump(movie_json, f, indent=2)

    upsert_movie(str(movie_id))
    update_index(movie_id, movie_json["title"])
    logger.info(f"Lazily ingested {movie_json['title']} ({movie_id})")
    return True


def initialize_all(n: int) -> dict:
    logger.info(f"Starting full pipeline initialization for {n} movies...")

    movie_ids = ingest_index(n)

    failed_ids = []
    for movie_id in movie_ids:
        try:
            ingest_movie(movie_id)
        except Exception as e:
            logger.warning(f"Failed to ingest movie {movie_id}: {e}")
            failed_ids.append(movie_id)

    if failed_ids:
        logger.warning(f"{len(failed_ids)} movies failed to ingest: {failed_ids}")

    compile_all_richtexts()

    embedded_count = initialize_all_embeddings()

    return {
        "movie_count": len(movie_ids) - len(failed_ids),
        "indexed_count": len(movie_ids),
        "embedded_count": embedded_count,
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids
    }


if __name__ == "__main__":
    n = int(input("How many movies to initialize? "))
    result = initialize_all(n)
    print(f"\nPipeline complete: {result}")
