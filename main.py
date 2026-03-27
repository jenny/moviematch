import logging
import re
import time

from config import SEARCH_CANDIDATES, SEARCH_DOC_TRUNCATE
from db import get_model, vector_count, vector_query
from claude import rerank, rerank_stream

logger = logging.getLogger(__name__)


def _parse_document(doc: str) -> dict:
    """Extract structured fields from a richtext document string."""
    result = {"year": "", "overview": "", "director": "", "cast": []}

    if "Plot: " in doc:
        start = doc.index("Plot: ") + 6
        end = doc.find("\n\n", start)
        result["overview"] = doc[start:end].strip() if end != -1 else doc[start:].strip()

    if "Director: " in doc:
        start = doc.index("Director: ") + 10
        end = doc.find("\n", start)
        result["director"] = doc[start:end].strip() if end != -1 else doc[start:].strip()

    if "Top Cast: " in doc:
        start = doc.index("Top Cast: ") + 10
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
    for match in matches:
        title = match.get("title")
        if not title:
            logger.warning("Missing title in vector result, skipping.")
            continue
        full_doc = match.get("document", "")
        parsed = _parse_document(full_doc)
        candidates.append({
            "title": title,
            "movie_poster": match.get("movie_poster") or "",
            "document": full_doc[:SEARCH_DOC_TRUNCATE],
            "year": parsed["year"],
            "overview": parsed["overview"],
            "director": parsed["director"],
            "cast": parsed["cast"],
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
    year_by_title = {c["title"]: c["year"] for c in candidates}
    overview_by_title = {c["title"]: c["overview"] for c in candidates}
    director_by_title = {c["title"]: c["director"] for c in candidates}
    cast_by_title = {c["title"]: c["cast"] for c in candidates}

    t0 = time.perf_counter()
    usage = None
    error = None
    for item in rerank_stream(query, candidates):
        if "__usage" in item:
            usage = item["__usage"]
            error = usage.pop("error", None)
        else:
            t = item.get("title", "")
            item["movie_poster"] = poster_by_title.get(t, "")
            item["year"] = year_by_title.get(t, "")
            item["overview"] = overview_by_title.get(t, "")
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
