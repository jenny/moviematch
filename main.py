import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import (
    SEARCH_CANDIDATES, SEARCH_DOC_TRUNCATE,
    RICHTEXT_PREFIX_CAST, RICHTEXT_PREFIX_DIRECTOR, RICHTEXT_PREFIX_GENRES, RICHTEXT_PREFIX_PLOT,
    PERSON_LOOKUP_TIMEOUT_S, PREPARSE_EXECUTOR_WORKERS,
)
from db import get_model, vector_count, vector_query
from claude import rerank, rerank_stream, _ingest_filmography_background
from query_parser import parse_query, apply_hard_filters, resolve_persons

# Module-level executor for concurrent person TMDB pre-fetches.
# Shared across requests; each request submits its own future.
_person_fetch_executor = ThreadPoolExecutor(max_workers=PREPARSE_EXECUTOR_WORKERS)

logger = logging.getLogger(__name__)


def _parse_document(doc: str) -> dict:
    """Extract structured fields from a richtext document string."""
    result = {"year": "", "overview": "", "genres": [], "director": "", "cast": []}

    if RICHTEXT_PREFIX_PLOT in doc:
        start = doc.index(RICHTEXT_PREFIX_PLOT) + len(RICHTEXT_PREFIX_PLOT)
        end = doc.find("\n\n", start)
        result["overview"] = doc[start:end].strip() if end != -1 else doc[start:].strip()

    if RICHTEXT_PREFIX_GENRES in doc:
        start = doc.index(RICHTEXT_PREFIX_GENRES) + len(RICHTEXT_PREFIX_GENRES)
        end = doc.find("\n", start)
        genres_str = doc[start:end].strip() if end != -1 else doc[start:].strip()
        result["genres"] = [g.strip() for g in genres_str.split(",") if g.strip()]

    if RICHTEXT_PREFIX_DIRECTOR in doc:
        start = doc.index(RICHTEXT_PREFIX_DIRECTOR) + len(RICHTEXT_PREFIX_DIRECTOR)
        end = doc.find("\n", start)
        result["director"] = doc[start:end].strip() if end != -1 else doc[start:].strip()

    if RICHTEXT_PREFIX_CAST in doc:
        start = doc.index(RICHTEXT_PREFIX_CAST) + len(RICHTEXT_PREFIX_CAST)
        end = doc.find("\n\n", start)
        cast_str = doc[start:end].strip() if end != -1 else doc[start:].strip()
        result["cast"] = [c.strip() for c in cast_str.split(",") if c.strip()]

    year_match = re.search(r'\((\d{4})\)', doc)
    if year_match:
        result["year"] = year_match.group(1)

    return result


def _fetch_candidates(query: str) -> tuple[list[dict], int, int]:
    """Embed query, query the vector store, and return (candidates, embedding_ms, chroma_ms)."""
    count = vector_count()
    if count == 0:
        raise RuntimeError("Vector store is empty. Please run embeddings.py first.")

    try:
        t0 = time.perf_counter()
        q_embedding = get_model().encode(query).tolist()
        embedding_ms = round((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        n_results = min(SEARCH_CANDIDATES, count)
        matches = vector_query(q_embedding, n_results)
        chroma_ms = round((time.perf_counter() - t0) * 1000)
    except Exception as e:
        raise RuntimeError(f"Error querying vector store: {e}")

    candidates = []
    seen_titles: set[str] = set()
    for match in matches:
        title = match.get("title")
        if not title:
            logger.warning("Missing title in vector result, skipping.")
            continue
        if title in seen_titles:
            logger.warning(f"Duplicate title in vector results, skipping: {title!r}")
            continue
        seen_titles.add(title)
        full_doc = match.get("document", "")
        parsed = _parse_document(full_doc)
        candidates.append({
            "title": title,
            "movie_poster": match.get("movie_poster") or "",
            "certification": match.get("certification", ""),
            "document": full_doc[:SEARCH_DOC_TRUNCATE],
            "year": parsed["year"],
            "overview": parsed["overview"],
            "genres": parsed["genres"],
            "director": parsed["director"],
            "cast": parsed["cast"],
        })

    return candidates, embedding_ms, chroma_ms


def search(query: str) -> tuple[list[dict], dict | None, dict]:
    # Pre-parse and concurrent person lookup (same as search_stream).
    parsed = parse_query(query)
    person_future = None
    if parsed.person_names:
        person_future = _person_fetch_executor.submit(resolve_persons, parsed)

    candidates, embedding_ms, chroma_ms = _fetch_candidates(query)
    timing = {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": 0}

    if person_future is not None:
        try:
            person_future.result(timeout=PERSON_LOOKUP_TIMEOUT_S)
            for pf in parsed.person_filmographies:
                if pf.get("movies"):
                    threading.Thread(
                        target=_ingest_filmography_background,
                        args=(pf["movies"],),
                        daemon=True,
                    ).start()
        except Exception:
            logger.warning("Person pre-fetch failed or timed out; falling back to Claude tools")

    candidates = apply_hard_filters(candidates, parsed)

    if not candidates:
        return [], None, timing

    poster_by_title = {c["title"]: c["movie_poster"] for c in candidates}
    certification_by_title = {c["title"]: c["certification"] for c in candidates}

    t0 = time.perf_counter()
    reranked, usage = rerank(query, candidates, parsed=parsed)
    timing["claude_ms"] = round((time.perf_counter() - t0) * 1000)

    for result in reranked:
        result["movie_poster"] = poster_by_title.get(result["title"], "")
        result["certification"] = certification_by_title.get(result["title"], "")
    return reranked, usage, timing


def search_stream(query: str):
    """Generator that yields result dicts (with movie_poster) as they stream from Claude,
    then yields {"__meta": {"embedding_ms": ..., "chroma_ms": ..., "claude_ms": ..., "usage": ...}}."""
    # Pre-parse the query for structured tokens (year, genre, cert, person names).
    # If person names are found, submit a TMDB lookup concurrently with embedding
    # so the round-trip latency is absorbed rather than added.
    parsed = parse_query(query)
    person_future = None
    if parsed.person_names:
        person_future = _person_fetch_executor.submit(resolve_persons, parsed)

    candidates, embedding_ms, chroma_ms = _fetch_candidates(query)

    if person_future is not None:
        try:
            person_future.result(timeout=PERSON_LOOKUP_TIMEOUT_S)
            # Trigger background ingestion for pre-fetched filmography films,
            # mirroring the ingestion that fires when Claude calls get_filmography directly.
            for pf in parsed.person_filmographies:
                if pf.get("movies"):
                    threading.Thread(
                        target=_ingest_filmography_background,
                        args=(pf["movies"],),
                        daemon=True,
                    ).start()
        except Exception:
            logger.warning("Person pre-fetch failed or timed out; falling back to Claude tools")

    candidates = apply_hard_filters(candidates, parsed)

    if not candidates:
        yield {"__meta": {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": 0, "usage": None}}
        return

    # Keyed by lowercased title to match _filter_results' case-insensitive behaviour.
    # Claude sees exact candidate titles in the prompt but may use slightly different
    # casing, and filmography-sourced titles may differ from vector DB titles.
    poster_by_title = {c["title"].lower(): c["movie_poster"] for c in candidates}
    certification_by_title = {c["title"].lower(): c["certification"] for c in candidates}
    year_by_title = {c["title"].lower(): c["year"] for c in candidates}
    overview_by_title = {c["title"].lower(): c["overview"] for c in candidates}
    genres_by_title = {c["title"].lower(): c["genres"] for c in candidates}
    director_by_title = {c["title"].lower(): c["director"] for c in candidates}
    cast_by_title = {c["title"].lower(): c["cast"] for c in candidates}

    # Seed with filmography metadata for titles not already covered by vector candidates.
    # Filmography movies from TMDB carry poster_path and release_date but not the richer
    # fields (overview, genres, cast) that only exist in the vector DB document.
    if parsed and parsed.person_filmographies:
        for pf in parsed.person_filmographies:
            for movie in pf.get("movies", []):
                title_lower = (movie.get("title") or "").lower()
                if not title_lower or title_lower in poster_by_title:
                    continue
                raw_date = movie.get("release_date", "")
                poster_by_title[title_lower] = movie.get("poster_path", "")
                year_by_title[title_lower] = raw_date[:4] if raw_date else ""
                certification_by_title[title_lower] = ""
                overview_by_title[title_lower] = ""
                genres_by_title[title_lower] = []
                director_by_title[title_lower] = ""
                cast_by_title[title_lower] = []

    t0 = time.perf_counter()
    usage = None
    error = None
    for item in rerank_stream(query, candidates, parsed=parsed):
        if "__usage" in item:
            usage = item["__usage"]
            error = usage.pop("error", None)
        else:
            t = (item.get("title") or "").lower()
            item["movie_poster"] = poster_by_title.get(t, "")
            item["certification"] = certification_by_title.get(t, "")
            item["year"] = year_by_title.get(t, "")
            item["overview"] = overview_by_title.get(t, "")
            item["genres"] = genres_by_title.get(t, [])
            item["director"] = director_by_title.get(t, "")
            item["cast"] = cast_by_title.get(t, [])
            yield item

    claude_ms = round((time.perf_counter() - t0) * 1000)
    yield {"__meta": {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": claude_ms, "usage": usage, "error": error}}


if __name__ == "__main__":
    query = input("Enter a search query: ")
    try:
        results, usage, timing = search(query)
    except RuntimeError as e:
        print(f"Error: {e}")
    else:
        if not results:
            print("No relevant matches found. Please try a different query.")
        else:
            print(f"\nFound {len(results)} relevant matches:\n")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"   {result['explanation']}\n")
            if usage:
                print(f"Usage: {usage}")
            print(f"Timing: {timing}")
