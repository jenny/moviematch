# Project: MovieMatch

## Quick Facts
- Stack: FastAPI + SentenceTransformers + ChromaDB (local) / Pinecone (prod) + Anthropic API.
- Purpose: Semantic movie recommendation engine. Users submit a natural language query; the app embeds it, retrieves vector-similar candidates, then uses Claude to rerank/filter them.

## Architecture at a Glance

| File | Role |
|------|------|
| `config.py` | All constants: models, thresholds, pricing, env vars |
| `query_parser.py` | Rule-based pre-parser: `parse_query()`, `apply_hard_filters()`, `resolve_persons()` |
| `claude.py` | Anthropic integration: `rerank()`, `rerank_stream()`, tool definitions |
| `main.py` | Orchestrates pre-parse → embedding → vector query → hard filter → Claude rerank |
| `db.py` | Singletons for embedding model + vector DB (`get_model`, `vector_query`, `vector_upsert_batch`, `vector_count`) |
| `embeddings.py` | Text embedding pipeline, batch upsert |
| `tmdb.py` | TMDB API: fetch, score, ingest movies; `search_person`, `get_filmography`, `search_movie_by_title`, `fetch_watch_providers`, `extract_certification`, `fetch_certification`, `get_certification_for_title` |
| `watchmode.py` | Watchmode API: streaming provider lookup (`search_title`, `fetch_providers`); primary source for `/streaming` endpoint |
| `richtext.py` | Builds `document` string for each movie (what gets embedded) |
| `pipeline.py` | Full init pipeline + single-movie ingestion |
| `api/app.py` | FastAPI app factory, CORS, lifespan; login/logout routes |
| `api/auth.py` | HMAC-SHA256 session cookie signing; `require_admin` FastAPI dependency |
| `login.html` | Admin login page (dark theme, JSON POST via fetch) |
| `api/routes/search.py` | `POST /recommend` — SSE streaming endpoint |
| `api/routes/admin.py` | `/initialize`, `/status`, `/logs` — all require admin auth |
| `api/routes/streaming.py` | `GET /streaming` — streaming provider lookup via Watchmode (primary) or TMDB (fallback); also returns MPAA certification |
| `logger.py` | JSON request logging; DEBUG level locally, INFO on Railway; writes to rotating file locally, stdout on Railway |
| `migrate_to_pinecone.py` | One-off script: copies all vectors from Chroma to Pinecone |
| `backfill_certifications.py` | One-off script: patches MPAA certification into existing Pinecone vector metadata without re-embedding (supports `--dry-run`; idempotent) |

## Query Pre-Parsing

Every query runs through `parse_query()` in `query_parser.py` before embedding. This is a pure regex pass (<1ms, no I/O) that extracts structured tokens into a `ParsedQuery` dataclass:

- **Year/decade**: "90s" → `year_min=1990, year_max=1999`; "early 2000s", "after 1985", "last 10 years", etc.
- **Genres**: "no documentaries" → `excluded_genres=["Documentary"]`; "sci-fi" → `required_genres=["Science Fiction"]`
- **Certifications**: "family friendly" → `allowed_certifications=["G","PG"]`; "no R-rated" → `excluded_certifications=["R"]`
- **Person names**: "directed by Bong Joon-ho" → `person_names=["Bong Joon-ho"], person_department="directing"`; also handles film slang ("Spike Lee joint")
- **Title references**: "something like Inception" → `reference_titles=["Inception"]`; "in the style of Wes Anderson" → also adds to `person_names`
- **Relative date hints**: "older", "classic", "more recent" → `relative_date_hints` (soft guidance injected into Claude's prompt)

**Person lookup — concurrent with embedding**: when `person_names` is non-empty, `resolve_persons()` is submitted to a `ThreadPoolExecutor` at the same time `_fetch_candidates()` starts. The TMDB round-trip (~500ms) overlaps with embedding+vector search (~220ms), then the result is awaited with a 1.5s timeout before calling Claude.

**Hard filtering**: `apply_hard_filters()` drops vector candidates that violate year, genre, or certification constraints before they reach Claude. Candidates with missing metadata are kept conservatively.

**Prompt injection**: resolved filmographies and extracted constraints are injected into Claude's prompt. All user-derived strings are sanitized with `_sanitize()` (strips `<`, `>`, `&`) before injection. When all requested persons are pre-resolved, person-lookup tools are removed from the tools list entirely (hard guarantee, not a prompt-only instruction).

**Background ingestion**: after a successful person pre-fetch, `_ingest_filmography_background` is triggered in a daemon thread — same as when Claude calls `get_filmography` directly.

Key config values:
- `PERSON_LOOKUP_TIMEOUT_S = 1.5` — max wait for concurrent TMDB pre-fetch
- `PREPARSE_EXECUTOR_WORKERS = 2` — thread pool size

## Anthropic Integration
- **Round 1**: Always Haiku (`CLAUDE_FAST_MODEL`) — cheap, handles tool-free queries
- **Round 2+**: Switches to Opus (`CLAUDE_MODEL`) only if non-terminal tools are invoked
- **Tools**: `search_person`, `get_filmography`, `return_results` (terminal); person-lookup tools removed from schema when filmography is pre-resolved
- **Anti-hallucination**: `_filter_results()` validates returned titles against candidate + filmography sets using case-insensitive matching
- **Streaming**: `rerank_stream()` uses `_extract_result_objects()` to yield results as JSON chunks arrive

Key config values:
- `SEARCH_CANDIDATES = 15` — vector results passed to Claude
- `SEARCH_DOC_TRUNCATE = 200` — chars of each movie doc sent in prompt
- `AGENT_MAX_TOOL_ROUNDS = 4`

## Tests
Run in venv with: `pytest` from project root.

| Test file | What it covers |
|-----------|---------------|
| `tests/test_query_parser.py` | `parse_query`: year/decade normalization, genre/cert/person extraction, slang triggers, "in the style of" dual-purpose, relative date hints; `apply_hard_filters`: year, genre, cert filtering, conservative empty-metadata handling; `resolve_persons`: TMDB mocked, success/failure/timeout paths |
| `tests/test_claude.py` | Pure functions: `_filter_results` (case-insensitive), `_extract_result_objects`, `_sanitize`, `_build_rerank_prompt` (constraint injection, filmography cap, suppression instruction, XSS sanitization) |
| `tests/test_api.py` | FastAPI endpoints via TestClient; mocks `search_stream`, `vector_count`, `get_model`, `search_movie_by_title`, `fetch_watch_providers`, `watchmode` |
| `tests/test_main.py` | `_parse_document()` richtext extraction; `search_stream` pre-parse integration: year filter reduces candidates, person timeout degrades gracefully, successful pre-fetch triggers background ingestion |
| `tests/test_tmdb.py` | Scoring: `_composite_score`, `select_top_n`, `filter_cast`, `filter_crew`; TMDB lookup: `search_movie_by_title`, `fetch_watch_providers`; certification: `extract_certification`, `fetch_certification`, `get_certification_for_title` |
| `tests/test_richtext.py` | `build_richtext()` edge cases |
| `tests/test_watchmode.py` | Watchmode lookup: `search_title` (year matching, not-found), `fetch_providers` (type filtering, deduplication, logo cache); `_load_source_logos` (null URL rejection) |
| `tests/test_auth.py` | Cookie signing: valid tokens, expiry, tampered signatures, missing secret |

**No test touches the real Anthropic, TMDB, or Watchmode APIs.** All external calls are mocked at the module level.

## Environment
Required env vars: `ANTHROPIC_API_KEY`, `TMDB_READ_ACCESS_TOKEN`
Optional: `WATCHMODE_API_KEY` (free tier at watchmode.com; enables reliable streaming availability data — falls back to TMDB without it), `VECTOR_DB` (auto-selects `pinecone` when `RAILWAY_ENVIRONMENT` is set, else `chroma`), `PINECONE_*` keys, `CORS_ORIGINS`, `RATE_LIMIT`, `LOG_DIR` (default `logs`; set to a Railway Volume path for log persistence)

Admin auth (all three required to enable the admin panel; fails closed if any is unset): `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_SECRET_KEY` (random 32-byte hex string — generate with `openssl rand -hex 32`)

## Key Decisions
- Haiku-first strategy: most queries resolve in round 1 without Opus
- `tool_choice={"type": "any"}` forces Claude to always call a tool (prevents prose responses)
- **Query pre-parsing chosen over Haiku extraction**: Haiku structured extraction was prototyped but rejected — P90 latency of 1,736ms is ~9× the 200ms embedding window. Regex is <1ms and $0 cost, covering the vast majority of query patterns.
- **Person pre-fetch runs concurrently with embedding**: the TMDB round-trip (~500ms) overlaps with embedding+vector search (~220ms) via `ThreadPoolExecutor`. Net cost: ~0ms added latency for person queries that resolve within the embedding window.
- **Conditional tool schema**: when all persons in a query are pre-resolved, `search_person` and `get_filmography` are removed from the tools list passed to Claude — a hard guarantee, not a prompt-only suggestion.
- Background thread ingests filmography discoveries into vector DB for future queries (triggered both when Claude calls `get_filmography` directly and when pre-fetch succeeds)
- `rerank` and `rerank_stream` share the same prompt/logic; `rerank` is kept for non-streaming use
- Certification is stored in vector DB metadata at ingest time and displayed immediately from search results; the vector DB is the sole source of truth for ratings — the `/streaming` endpoint handles providers only

## Model Use During Development
- Use Sonnet for planning and orchestration, but launch parallel sub-agents with Haiku for execution and research.

## Coding Hygiene
- Discuss and get approval for the technical approach first. Do not proceed with code changes until the technical approach is agreed upon.
- Code changes should be implemented in a short-lived feature branch that is deleted only after all code changes are committed and merged to main.
- Code changes should include clear inline comments for future collaborators.

## Forbidden Directories
Do not read or modify files in these directories:
- `venv/` — Python environment
- `data/` — raw TMDB JSON dumps
- `embeddings/` — ChromaDB binary data files
- `local/archive/` — older scratch notes and design proposals (already digested into this file)
- `.git/` — use `git` CLI commands instead
