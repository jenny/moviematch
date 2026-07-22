import itertools
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from config import (
    SEARCH_CANDIDATES, SEARCH_DOC_TRUNCATE,
    RICHTEXT_PREFIX_CAST, RICHTEXT_PREFIX_DIRECTOR, RICHTEXT_PREFIX_GENRES, RICHTEXT_PREFIX_PLOT,
    PERSON_LOOKUP_TIMEOUT_S, TITLE_LOOKUP_TIMEOUT_S, PREPARSE_EXECUTOR_WORKERS,
    ANCHOR_FETCH_DEPTH, ANCHOR_CANDIDATES_QUALIFIED,
    CERT_FETCH_WORKERS, CERT_FETCH_TIMEOUT_S,
)
from db import get_model, vector_count, vector_query, vector_fetch_by_ids
from claude import rerank, rerank_stream, _ingest_filmography_background, _ingest_reference_background
from query_parser import parse_query, apply_hard_filters, resolve_persons, resolve_reference_titles
from tmdb import fetch_certification

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


def _matches_to_candidates(matches: list[dict], drop_titles: set[str] = frozenset()) -> list[dict]:
    """Turn raw vector-store matches into candidate dicts, deduplicating by title.

    drop_titles (lowercased) are skipped — used by anchor retrieval to remove the
    reference movie itself from its own neighbor list (it sits at distance ~0 as its
    own top hit, and is surfaced separately via the always-append guarantee).
    """
    candidates = []
    seen_titles: set[str] = set()
    for match in matches:
        title = match.get("title")
        if not title:
            logger.warning("Missing title in vector result, skipping.")
            continue
        if title.lower() in drop_titles:
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
    return candidates


def _interleave(ranked_lists: list[list[dict]]) -> list[dict]:
    """Round-robin merge several ranked match lists, dropping cross-list duplicate
    titles and preserving each list's relative order.

    Used for multi-reference queries ("movies like X and Y") so each anchor's
    neighbors are represented evenly rather than one anchor dominating the pool.
    """
    merged = []
    seen: set[str] = set()
    for tier in itertools.zip_longest(*ranked_lists):
        for match in tier:
            if match is None:
                continue
            title = (match.get("title") or "").lower()
            if not title or title in seen:
                continue
            seen.add(title)
            merged.append(match)
    return merged


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

    return _matches_to_candidates(matches), embedding_ms, chroma_ms


def _fetch_candidates_anchored(query: str, parsed) -> tuple[list[dict], int, int]:
    """Anchor-aware retrieval. Returns (candidates, embedding_ms, chroma_ms).

    When a referenced title ("movies like X") resolved to a movie already in the
    vector DB, retrieve by that movie's stored DOCUMENT embedding (retrieval-by-
    example) rather than the query token embedding — a symmetric encoder can't turn
    a bare title into the film's content, but its document lands squarely among true
    neighbors. Each resolved reference that is found in the DB gets its document
    attached as ref["_doc"] so the caller can (a) tell in-DB refs from cold ones for
    background ingestion and (b) build the always-append guarantee card.

    Falls back to the plain query-embedding path (_fetch_candidates) for non-reference
    queries and for cold references not yet in the DB.
    """
    anchor_docs: list[dict] = []
    if parsed and parsed.reference_movie_ids:
        for ref in parsed.reference_movie_ids:
            try:
                docs = vector_fetch_by_ids([str(ref["id"])])
            except Exception as e:
                logger.warning("Anchor doc fetch failed for %r: %s", ref.get("title"), e)
                docs = []
            if docs:
                ref["_doc"] = docs[0]          # in-DB: attach for guarantee + ingest gating
                anchor_docs.append(docs[0])

    # No usable anchor (non-reference query, or every reference is cold) → token path.
    if not anchor_docs:
        return _fetch_candidates(query)

    count = vector_count()
    if count == 0:
        raise RuntimeError("Vector store is empty. Please run embeddings.py first.")

    try:
        depth = min(ANCHOR_FETCH_DEPTH, count)
        t0 = time.perf_counter()
        # Re-encode each anchor's stored richtext to reproduce its vector (encoding is
        # deterministic), then fetch its nearest neighbors. Deep fetch (ANCHOR_FETCH_DEPTH)
        # leaves headroom for post-retrieval hard filters; truncation happens in the caller.
        per_anchor = [get_model().encode(d["document"]).tolist() for d in anchor_docs]
        per_anchor_matches = [vector_query(vec, depth) for vec in per_anchor]
        chroma_ms = round((time.perf_counter() - t0) * 1000)
    except Exception as e:
        raise RuntimeError(f"Error querying vector store: {e}")

    drop_titles = {(d.get("title") or "").lower() for d in anchor_docs}
    candidates = _matches_to_candidates(_interleave(per_anchor_matches), drop_titles=drop_titles)
    # embedding_ms folded into chroma_ms here (anchor encode + neighbor query are one step).
    return candidates, 0, chroma_ms


def _reference_guarantee_cards(parsed) -> list[dict]:
    """Result-ready cards for referenced films, so the referenced film itself is always
    available to append if Claude's rerank didn't surface it.

    Gated to references that actually anchored — i.e. were found in the vector DB
    (`_doc` attached by _fetch_candidates_anchored). A *cold* reference is deliberately
    NOT guaranteed here: we only resolved a TMDB id for it, which for an ambiguous phrase
    ("movies like the ones from the 80s") may be a wrong/irrelevant match, and appending
    that would bypass Claude's filtering. Cold references are background-ingested instead,
    so a genuine one becomes anchored — and thus guaranteed — on the next search.
    """
    cards = []
    for ref in (parsed.reference_movie_ids if parsed else []):
        doc = ref.get("_doc")
        if not doc:
            continue  # cold reference — not guaranteed until ingested (see docstring)
        pdoc = _parse_document(doc.get("document", ""))
        cards.append({
            "title": doc.get("title") or ref["title"],
            "explanation": "The film you referenced.",
            "movie_poster": doc.get("movie_poster") or "",
            "certification": doc.get("certification", ""),
            "year": pdoc["year"],
            "overview": pdoc["overview"],
            "genres": pdoc["genres"],
            "director": pdoc["director"],
            "cast": pdoc["cast"],
        })
    return cards


def _build_metadata_lookups(candidates: list[dict], parsed=None) -> dict[str, dict]:
    """Build title→field lookup dicts (lowercased keys) for the reranked results.

    First indexes the vector candidates, then seeds metadata for filmography
    movies that fell outside the top-N vector candidates: a batch vector_fetch
    by TMDB ID yields full rich metadata for films already ingested, and the
    rest fall back to the sparse poster+year from the TMDB filmography payload.

    Shared by search() and search_stream() so both apply identical casing and
    filmography-seeding behaviour. Returns a dict of per-field lookup dicts.
    """
    lookups = {
        "movie_poster": {c["title"].lower(): c["movie_poster"] for c in candidates},
        "certification": {c["title"].lower(): c["certification"] for c in candidates},
        "year": {c["title"].lower(): c["year"] for c in candidates},
        "overview": {c["title"].lower(): c["overview"] for c in candidates},
        "genres": {c["title"].lower(): c["genres"] for c in candidates},
        "director": {c["title"].lower(): c["director"] for c in candidates},
        "cast": {c["title"].lower(): c["cast"] for c in candidates},
    }

    # Seed metadata for pre-fetched filmography films that fell outside the vector
    # candidate pool. Mid-stream get_filmography discoveries are seeded separately by
    # search_stream via the __filmography sentinel (same helper).
    if parsed and parsed.person_filmographies:
        movies = [movie for pf in parsed.person_filmographies for movie in pf.get("movies", [])]
        _seed_filmography_metadata(lookups, movies)

    return lookups


def _batch_fetch_certifications(movie_ids: list[int]) -> dict[int, str]:
    """Runtime certification backfill for films that carry no cert in their sparse
    filmography payload and aren't yet ingested in the vector DB. Fanned out across a
    bounded thread pool so N films cost ~one TMDB round-trip; hard-capped by
    CERT_FETCH_TIMEOUT_S so a slow/hung TMDB can't stall the stream — unresolved films
    stay unrated (surfaced as "Rating not available" client-side) and get corrected once
    background ingestion stores their real cert.

    Returns {movie_id: certification}; only films that resolved to a non-empty cert.
    """
    if not movie_ids:
        return {}
    certs: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=CERT_FETCH_WORKERS) as ex:
        future_to_id = {ex.submit(fetch_certification, mid): mid for mid in movie_ids}
        try:
            for future in as_completed(future_to_id, timeout=CERT_FETCH_TIMEOUT_S):
                mid = future_to_id[future]
                try:
                    cert = future.result()
                    if cert:
                        certs[mid] = cert
                except Exception as e:
                    logger.warning("Runtime certification fetch failed for movie %s: %s", mid, e)
        except TimeoutError:
            # Partial results are fine — whatever resolved in time is used; the rest
            # stay blank. Cancel stragglers so the pool tears down promptly.
            unresolved = [mid for f, mid in future_to_id.items() if not f.done()]
            logger.warning("Runtime certification fetch timed out for %d film(s): %s", len(unresolved), unresolved)
            for f in future_to_id:
                f.cancel()
    return certs


def _seed_filmography_metadata(lookups: dict, movies: list[dict]) -> None:
    """Seed poster/certification/year/etc into `lookups` for filmography films not
    already present. Mutates `lookups` in place.

    Films already ingested in the vector DB get full rich metadata (incl. certification)
    via a batch vector_fetch_by_ids. Films not yet ingested fall back to the sparse
    poster+year from the TMDB filmography payload; their certification — absent from that
    payload — is backfilled at runtime via a bounded concurrent TMDB fetch so the rating
    still renders on first view.

    Shared by _build_metadata_lookups (pre-fetched filmographies) and search_stream
    (mid-stream get_filmography discoveries) so both apply identical seeding.
    """
    # Only seed titles we don't already have — candidates and prior seeds win.
    existing_lower = set(lookups["movie_poster"].keys())
    outside = [
        movie for movie in movies
        if (movie.get("title") or "").lower() and (movie.get("title") or "").lower() not in existing_lower
    ]
    if not outside:
        return

    # Batch-fetch by TMDB ID — direct lookup, no embedding needed.
    id_to_movie = {str(movie["id"]): movie for movie in outside if movie.get("id")}
    fetched = vector_fetch_by_ids(list(id_to_movie.keys()))

    # Seed full rich metadata for films found in the vector DB.
    fetched_titles_lower: set[str] = set()
    for fetched_movie in fetched:
        t = (fetched_movie.get("title") or "").lower()
        if not t:
            continue
        parsed_doc = _parse_document(fetched_movie["document"])
        lookups["movie_poster"][t] = fetched_movie["movie_poster"]
        lookups["certification"][t] = fetched_movie["certification"]
        lookups["year"][t] = parsed_doc["year"]
        lookups["overview"][t] = parsed_doc["overview"]
        lookups["genres"][t] = parsed_doc["genres"]
        lookups["director"][t] = parsed_doc["director"]
        lookups["cast"][t] = parsed_doc["cast"]
        fetched_titles_lower.add(t)

    # Films not yet ingested: sparse poster+year from the filmography payload. Their
    # cert isn't in that payload, so backfill it concurrently from TMDB (#2).
    not_ingested = [
        movie for movie in outside
        if (movie.get("title") or "").lower() not in fetched_titles_lower
    ]
    certs_by_id = _batch_fetch_certifications([movie["id"] for movie in not_ingested if movie.get("id")])
    for movie in not_ingested:
        title_lower = movie["title"].lower()
        raw_date = movie.get("release_date", "")
        lookups["movie_poster"][title_lower] = movie.get("poster_path", "")
        lookups["year"][title_lower] = raw_date[:4] if raw_date else ""
        lookups["certification"][title_lower] = certs_by_id.get(movie.get("id"), "")
        lookups["overview"][title_lower] = ""
        lookups["genres"][title_lower] = []
        lookups["director"][title_lower] = ""
        lookups["cast"][title_lower] = []


def _trigger_reference_ingestion(parsed) -> None:
    """Background-ingest cold references (resolved but not yet in the DB) with the
    quality gate bypassed, so they become anchorable on the next query. In-DB
    references (those with an attached _doc from _fetch_candidates_anchored) are
    skipped — already ingested."""
    if not (parsed and parsed.reference_movie_ids):
        return
    cold = [ref for ref in parsed.reference_movie_ids if "_doc" not in ref]
    if cold:
        threading.Thread(target=_ingest_reference_background, args=(cold,), daemon=True).start()


def _candidate_limit(parsed) -> int:
    """Candidates handed to Claude: a wider slice when a reference query carries a soft
    qualifier, so the rerank has room to reorder within the anchor neighborhood."""
    return ANCHOR_CANDIDATES_QUALIFIED if (parsed and parsed.has_soft_qualifier) else SEARCH_CANDIDATES


def search(query: str) -> tuple[list[dict], dict | None, dict]:
    # Pre-parse and concurrent person/reference lookups (same as search_stream).
    parsed = parse_query(query)
    person_future = None
    if parsed.person_names:
        person_future = _person_fetch_executor.submit(resolve_persons, parsed)
    # Reference-title resolution is awaited before retrieval since it decides whether
    # we anchor by-document or fall back to the query token embedding.
    title_future = None
    if parsed.reference_titles:
        title_future = _person_fetch_executor.submit(resolve_reference_titles, parsed)
    if title_future is not None:
        try:
            title_future.result(timeout=TITLE_LOOKUP_TIMEOUT_S)
        except Exception:
            logger.warning("Reference-title resolution failed or timed out; using token retrieval")

    candidates, embedding_ms, chroma_ms = _fetch_candidates_anchored(query, parsed)
    timing = {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": 0}

    _trigger_reference_ingestion(parsed)

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
    candidates = candidates[:_candidate_limit(parsed)]

    # Bail out only when there is genuinely nothing to recommend from — no candidates,
    # no pre-fetched filmography, and no referenced film to guarantee.
    has_filmography = bool(parsed and parsed.person_filmographies)
    has_references = bool(parsed and parsed.reference_movie_ids)
    reranked: list[dict] = []
    usage = None
    if candidates or has_filmography:
        # Shared lowercased lookups (incl. filmography-outside seeding) — identical to search_stream.
        lookups = _build_metadata_lookups(candidates, parsed)
        t0 = time.perf_counter()
        reranked, usage = rerank(query, candidates, parsed=parsed)
        timing["claude_ms"] = round((time.perf_counter() - t0) * 1000)
        for result in reranked:
            t = (result.get("title") or "").lower()
            result["movie_poster"] = lookups["movie_poster"].get(t, "")
            result["certification"] = lookups["certification"].get(t, "")
    elif not has_references:
        return [], None, timing

    # Always-append guarantee: the referenced film itself must be in the results,
    # appended (even past the candidate limit) if the rerank didn't surface it.
    yielded = {(r.get("title") or "").lower() for r in reranked}
    for card in _reference_guarantee_cards(parsed):
        if card["title"].lower() not in yielded:
            reranked.append(card)
            yielded.add(card["title"].lower())
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
    # Reference-title resolution runs concurrently, but is awaited before retrieval
    # since it decides anchor (by-document) vs query token retrieval.
    title_future = None
    if parsed.reference_titles:
        title_future = _person_fetch_executor.submit(resolve_reference_titles, parsed)
    if title_future is not None:
        try:
            title_future.result(timeout=TITLE_LOOKUP_TIMEOUT_S)
        except Exception:
            logger.warning("Reference-title resolution failed or timed out; using token retrieval")

    candidates, embedding_ms, chroma_ms = _fetch_candidates_anchored(query, parsed)

    # Background-ingest cold references (not yet in the DB) so they're anchorable next time.
    _trigger_reference_ingestion(parsed)

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
    candidates = candidates[:_candidate_limit(parsed)]

    # Only bail out when there is genuinely nothing to recommend from. When a person
    # filmography was pre-fetched, or a title was referenced, we can still produce
    # output even if hard filters removed every semantic candidate — filmography titles
    # are seeded into the prompt, and referenced films are appended via the guarantee.
    has_filmography = bool(parsed and parsed.person_filmographies)
    has_references = bool(parsed and parsed.reference_movie_ids)
    yielded_titles: set[str] = set()
    usage = None
    error = None
    claude_ms = 0

    if candidates or has_filmography:
        # Shared lowercased lookups keyed by title (incl. filmography-outside seeding).
        # Lowercased to match _filter_results' case-insensitive behaviour: Claude sees
        # exact candidate titles in the prompt but may use slightly different casing,
        # and filmography-sourced titles may differ from vector DB titles.
        lookups = _build_metadata_lookups(candidates, parsed)

        t0 = time.perf_counter()
        for item in rerank_stream(query, candidates, parsed=parsed):
            if "__usage" in item:
                usage = item["__usage"]
                error = usage.pop("error", None)
            elif "__filmography" in item:
                # Claude discovered these films via a mid-stream get_filmography call —
                # they're absent from the candidate pool, so seed their poster/cert now,
                # before the result items that reference them stream through. Without this
                # both fields render blank for tool-discovered titles.
                _seed_filmography_metadata(lookups, item["__filmography"])
            else:
                t = (item.get("title") or "").lower()
                item["movie_poster"] = lookups["movie_poster"].get(t, "")
                item["certification"] = lookups["certification"].get(t, "")
                item["year"] = lookups["year"].get(t, "")
                item["overview"] = lookups["overview"].get(t, "")
                item["genres"] = lookups["genres"].get(t, [])
                item["director"] = lookups["director"].get(t, "")
                item["cast"] = lookups["cast"].get(t, [])
                yielded_titles.add(t)
                yield item
        claude_ms = round((time.perf_counter() - t0) * 1000)
    elif not has_references:
        yield {"__meta": {"embedding_ms": embedding_ms, "chroma_ms": chroma_ms, "claude_ms": 0, "usage": None}}
        return

    # Always-append guarantee: emit any referenced film the rerank didn't surface,
    # after the model stream and before the meta sentinel (even past the candidate limit).
    for card in _reference_guarantee_cards(parsed):
        if card["title"].lower() not in yielded_titles:
            yielded_titles.add(card["title"].lower())
            yield card

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
