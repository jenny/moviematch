import logging
import time

from config import SEARCH_CANDIDATES, SEARCH_DOC_TRUNCATE
from db import get_model, vector_count, vector_query
from claude import rerank, rerank_stream

logger = logging.getLogger(__name__)


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
    for match in matches:
        title = match.get("title")
        if not title:
            logger.warning("Missing title in vector result, skipping.")
            continue
        candidates.append({
            "title": title,
            "movie_poster": match.get("movie_poster") or "",
            "document": match.get("document", "")[:SEARCH_DOC_TRUNCATE],
        })

    return candidates, embedding_ms, chroma_ms


def search(query: str) -> tuple[list[dict], dict | None, dict]:
    candidates, embedding_ms, chroma_ms = _fetch_candidates(query)
    timing = {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": 0}

    if not candidates:
        return [], None, timing

    poster_by_title = {c["title"]: c["movie_poster"] for c in candidates}

    t0 = time.perf_counter()
    reranked, usage = rerank(query, candidates)
    timing["claude_ms"] = round((time.perf_counter() - t0) * 1000)

    for result in reranked:
        result["movie_poster"] = poster_by_title.get(result["title"], "")
    return reranked, usage, timing


def search_stream(query: str):
    """Generator that yields result dicts (with movie_poster) as they stream from Claude,
    then yields {"__meta": {"embedding_ms": ..., "chroma_ms": ..., "claude_ms": ..., "usage": ...}}."""
    candidates, embedding_ms, chroma_ms = _fetch_candidates(query)

    if not candidates:
        yield {"__meta": {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": 0, "usage": None}}
        return

    poster_by_title = {c["title"]: c["movie_poster"] for c in candidates}

    t0 = time.perf_counter()
    usage = None
    error = None
    for item in rerank_stream(query, candidates):
        if "__usage" in item:
            usage = item["__usage"]
            error = usage.pop("error", None)
        else:
            item["movie_poster"] = poster_by_title.get(item.get("title", ""), "")
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
