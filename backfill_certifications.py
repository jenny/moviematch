"""One-off script: backfill MPAA certification metadata into the vector store.

For each vector that has an empty certification, fetches the rating from TMDB
and updates the metadata in-place — no re-embedding required.

Usage:
    source venv/bin/activate
    python backfill_certifications.py           # live run
    python backfill_certifications.py --dry-run # preview without writing

Notes:
    - Works with both ChromaDB (local) and Pinecone (production) backends.
    - Skips vectors that already have a non-empty certification (idempotent — safe to re-run)
    - Rate-limits at ~20 req/s, well under TMDB's 40 req/s ceiling
    - Pinecone's index.list() paginates automatically; ChromaDB's collection.get() returns everything in one call
"""

import logging
import sys
import time

from config import VECTOR_DB
from db import _get_chroma_collection, _get_pinecone_index
from tmdb import fetch_certification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Stay well under TMDB's 40 req/s rate limit
_TMDB_REQUEST_DELAY = 0.05  # seconds between requests (~20 req/s)


def _backfill_chroma(dry_run: bool = False) -> tuple[int, int]:
    """Returns (updated, skipped) counts."""
    collection = _get_chroma_collection()
    result = collection.get(include=["metadatas"])
    ids = result["ids"]
    metadatas = result["metadatas"]

    updated = skipped = 0
    for vector_id, meta in zip(ids, metadatas):
        if meta.get("certification"):
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[dry-run] Would fetch cert for {meta.get('title', vector_id)!r} (id={vector_id})")
            updated += 1
            continue

        cert = fetch_certification(int(vector_id))
        collection.update(ids=[vector_id], metadatas=[{**meta, "certification": cert}])
        logger.info(f"Updated {meta.get('title', vector_id)!r}: {cert or '(none)'}")
        updated += 1
        time.sleep(_TMDB_REQUEST_DELAY)

    return updated, skipped


def _backfill_pinecone(dry_run: bool = False) -> tuple[int, int]:
    """Returns (updated, skipped) counts."""
    index = _get_pinecone_index()
    updated = skipped = 0

    # list() paginates automatically and yields vector IDs
    for vector_id in index.list():
        fetch_result = index.fetch(ids=[vector_id])
        vector = fetch_result.vectors.get(vector_id)
        if vector is None:
            continue

        existing_cert = (vector.metadata or {}).get("certification", "")
        if existing_cert:
            skipped += 1
            continue

        title = (vector.metadata or {}).get("title", vector_id)
        if dry_run:
            logger.info(f"[dry-run] Would fetch cert for {title!r} (id={vector_id})")
            updated += 1
            continue

        cert = fetch_certification(int(vector_id))
        index.update(id=vector_id, set_metadata={"certification": cert})
        logger.info(f"Updated {title!r}: {cert or '(none)'}")
        updated += 1
        time.sleep(_TMDB_REQUEST_DELAY)

    return updated, skipped


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        logger.info("Dry run — no changes will be written.")

    logger.info(f"Backfilling certifications in {VECTOR_DB} vector store...")
    if VECTOR_DB == "pinecone":
        updated, skipped = _backfill_pinecone(dry_run=dry_run)
    else:
        updated, skipped = _backfill_chroma(dry_run=dry_run)

    action = "Would update" if dry_run else "Updated"
    logger.info(f"Done. {action}: {updated}, already had cert: {skipped}")
