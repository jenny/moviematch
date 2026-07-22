"""One-off audit: how many stored poster_path values 404 at the TMDB image origin?

Answers the "how often does a poster go missing from origin?" question that we
can't get from server logs (the 404 is a client-side image fetch that never hits
our backend). Enumerates every vector's stored `movie_poster`, HEAD-requests the
w185 CDN variant the card uses, and reports the stale/missing rate.

Read-only: touches only the vector store (via db.py) and image.tmdb.org. No writes.

Usage (from project root, in venv):
    python tools/audit_posters.py            # audit all vectors
    python tools/audit_posters.py --limit 200  # sample the first N
"""
import argparse
import concurrent.futures
import os
import sys

# Running `python tools/audit_posters.py` puts tools/ on sys.path, not the project
# root — add the root so `import db`/`config` resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

# TMDB card variant the frontend renders (app.html card-poster img).
IMAGE_BASE = "https://image.tmdb.org/t/p/w185"
WORKERS = 16
TIMEOUT_S = 10


def _all_poster_records() -> list[dict]:
    """Return [{title, movie_poster}, ...] for every vector in the store."""
    from db import VECTOR_DB, _get_chroma_collection, _get_pinecone_index

    if VECTOR_DB == "pinecone":
        # Pinecone has no cheap "get all"; list ids in pages, then fetch metadata.
        index = _get_pinecone_index()
        records = []
        for id_page in index.list():
            fetched = index.fetch(ids=id_page)
            for v in fetched.vectors.values():
                if v.metadata:
                    records.append({
                        "title": v.metadata.get("title", ""),
                        "movie_poster": v.metadata.get("movie_poster", ""),
                    })
        return records

    result = _get_chroma_collection().get(include=["metadatas"])
    return [
        {"title": (m or {}).get("title", ""), "movie_poster": (m or {}).get("movie_poster", "")}
        for m in result["metadatas"]
    ]


def _check(record: dict) -> tuple[str, str, int]:
    """HEAD the poster URL. Returns (title, poster_path, status) where status is the
    HTTP code, or 0 for a request-level error."""
    path = record["movie_poster"]
    try:
        resp = requests.head(IMAGE_BASE + path, timeout=TIMEOUT_S, allow_redirects=True)
        return record["title"], path, resp.status_code
    except requests.RequestException:
        return record["title"], path, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="audit only the first N vectors (0 = all)")
    args = ap.parse_args()

    records = _all_poster_records()
    total_vectors = len(records)

    # Split out vectors with no stored poster_path at all — those are a *separate*
    # class from "path present but 404s at origin" and shouldn't inflate the stale rate.
    missing_path = [r for r in records if not r["movie_poster"]]
    with_path = [r for r in records if r["movie_poster"]]
    if args.limit:
        with_path = with_path[:args.limit]

    print(f"Vectors total:            {total_vectors}")
    print(f"  no poster_path stored:  {len(missing_path)}")
    print(f"  poster_path to check:   {len(with_path)}")
    print(f"Checking {IMAGE_BASE} (workers={WORKERS})...\n")

    ok = 0
    stale = []  # (title, path, status)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (title, path, status) in enumerate(ex.map(_check, with_path), start=1):
            if status == 200:
                ok += 1
            else:
                stale.append((title, path, status))
            if i % 100 == 0:
                print(f"  ...{i}/{len(with_path)} checked", file=sys.stderr)

    checked = len(with_path)
    print("\n=== Results ===")
    print(f"Checked (path present):   {checked}")
    print(f"  200 OK:                 {ok}")
    print(f"  stale/404/error:        {len(stale)}")
    if checked:
        print(f"  origin-stale rate:      {len(stale) / checked:.2%}")
    if with_path:
        pass
    print(f"\nOverall no-poster surface (no path OR stale) out of {total_vectors} vectors: "
          f"{(len(missing_path) + len(stale)) / total_vectors:.2%}" if total_vectors else "")

    if stale:
        print("\nStale/missing at origin (up to 40 shown):")
        for title, path, status in stale[:40]:
            print(f"  [{status or 'ERR'}] {title}  {path}")


if __name__ == "__main__":
    main()
