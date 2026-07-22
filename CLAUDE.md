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
| `db.py` | Singletons for embedding model + vector DB (`get_model`, `vector_query`, `vector_fetch_by_ids`, `vector_upsert_batch`, `vector_count`) |
| `embeddings.py` | Text embedding pipeline, batch upsert |
| `tmdb.py` | TMDB API: fetch, score, ingest movies; `search_person`, `get_filmography`, `search_movie_by_title`, `fetch_movie_rating` (id + vote_average for `/ratings`), `fetch_watch_providers`, `extract_certification`, `fetch_certification`, `get_certification_for_title` |
| `watchmode.py` | Watchmode API: streaming provider lookup (`search_title`, `fetch_providers`); primary source for `/streaming` endpoint; cache keyed by `title_id:country` |
| `omdb.py` | OMDb API: critic/audience review scores (`fetch_ratings` → RT critic Tomatometer, IMDb, Metacritic + imdbID); source for the `/ratings` endpoint; 24h TTL cache keyed `ratings:{title}:{year}`; **fail-soft** — returns `{}` when `OMDB_API_KEY` is unset |
| `richtext.py` | Builds `document` string for each movie (what gets embedded) |
| `pipeline.py` | Full init pipeline + single-movie ingestion |
| `api/app.py` | FastAPI app factory, CORS (GET+POST, Content-Type only), security-header middleware, lifespan (warms embedding model via a throwaway `encode`, Anthropic client, and TMDB connection to keep the first user query off the cold path); login/logout routes |
| `api/auth.py` | HMAC-SHA256 session cookie signing; `require_admin` FastAPI dependency |
| `api/geo.py` | `get_client_ip()`, `resolve_country()` — IP → ISO country code via ipinfo.io (free tier, 50k/mo); process-lifetime cache; private/loopback fast-path |
| `login.html` | Admin login page (dark theme, JSON POST via fetch) |
| `api/routes/search.py` | `POST /recommend` — SSE streaming endpoint |
| `api/routes/admin.py` | `/initialize`, `/status`, `/logs` — all require admin auth |
| `api/routes/streaming.py` | `GET /region` — IP-based country detection (called once on page load); `GET /streaming` and `POST /streaming/batch` — region-aware provider lookup via Watchmode (primary) or TMDB (fallback); `country` param defaults to `US` |
| `api/routes/ratings.py` | `GET /ratings` and `POST /ratings/batch` — critic/audience review scores for the overlay "Reviews" section. Combines OMDb (RT/IMDb/Metacritic) + TMDB (`vote_average`); returns ordered `{provider, score, url}` (RT→IMDb→Metacritic→TMDB), absent scores omitted. Region-independent (no `country` param) |
| `logger.py` | JSON request logging; DEBUG level locally, INFO on Railway; writes to rotating file locally, stdout on Railway |
| `migrate_to_pinecone.py` | One-off script: copies all vectors from Chroma to Pinecone |
| `backfill_certifications.py` | One-off script: patches MPAA certification into existing Pinecone vector metadata without re-embedding (supports `--dry-run`; idempotent) |
| `tools/audit_posters.py` | Read-only audit: HEAD-checks every stored `poster_path` against the TMDB CDN to measure the origin-stale / no-path rate (supports `--limit`) |

## Query Pre-Parsing

Every query runs through `parse_query()` in `query_parser.py` before embedding. This is a pure regex pass (<1ms, no I/O) that extracts structured tokens into a `ParsedQuery` dataclass:

- **Year/decade**: "90s" → `year_min=1990, year_max=1999`; "early 2000s", "after 1985", "last 10 years", etc.
- **Genres**: "no documentaries" → `excluded_genres=["Documentary"]`; "sci-fi" → `required_genres=["Science Fiction"]`; "animated" / "animation" → `required_genres=["Animation"]`; "live action" / "live-action" → `excluded_genres=["Animation"]` (special-cased via `_LIVE_ACTION_PATTERN` since "live action" is not a TMDB genre)
- **Certifications**: "family friendly" / "kids movie" / "children's film" / "for children" etc. → `excluded_certifications=["NC-17"]` + `certification_caveats` (soft Claude guidance to prefer G/PG/PG-13, surface R only as last resort with explicit note); "no R-rated" → `excluded_certifications=["R"]`
- **Person names**: "directed by Bong Joon-ho" → `person_names=["Bong Joon-ho"], person_department="directing"`; also handles film slang ("Spike Lee joint")
- **Title references**: `like <title>` is treated as a reference **by default** — "something like Inception", "more movies like Parasite (2019)", "movies like Krull", "sci-fi like Die Hard", "like inception but funnier" — and titles need **not** be capitalized (`movies like krull` works). The *verb* sense of "like" (subject-pronoun + like: "I like", "we like", "you'd like") is the carved-out exception via negative lookbehind. Genuinely ambiguous leftovers ("movies like the ones from the 80s") are intentionally allowed through to TMDB resolution rather than special-cased — they fail to resolve or get filtered by retrieval + Claude. Also "similar to The Godfather" and quoted `like "The Dark Knight"`. All → `reference_titles=[...]`. "in the style of Wes Anderson" → also adds to `person_names`; reference titles are injected into the Claude prompt as a `<reference_films>` block
- **Relative date hints**: "older", "classic", "more recent" → `relative_date_hints` (soft guidance injected into Claude's prompt)

**Person lookup — concurrent with embedding**: when `person_names` is non-empty, `resolve_persons()` is submitted to a `ThreadPoolExecutor` at the same time `_fetch_candidates()` starts. The TMDB round-trip (~500ms) overlaps with embedding+vector search (~220ms), then the result is awaited with a 1.5s timeout before calling Claude.

**Hard filtering**: `apply_hard_filters()` drops vector candidates that violate year, genre, or certification constraints before they reach Claude. Candidates with missing metadata are kept conservatively.

**Prompt injection**: resolved filmographies, extracted constraints, and reference titles are injected into Claude's prompt as structured XML blocks (`<constraints>`, `<reference_films>`, `<filmography>`). All user-derived strings are sanitized with `_sanitize()` (strips `<`, `>`, `&`) before injection. When all requested persons are pre-resolved, person-lookup tools are removed from the tools list entirely (hard guarantee, not a prompt-only instruction).

**Background ingestion**: after a successful person pre-fetch, `_ingest_filmography_background` is triggered in a daemon thread — same as when Claude calls `get_filmography` directly.

Key config values:
- `PERSON_LOOKUP_TIMEOUT_S = 1.5` — max wait for concurrent TMDB pre-fetch
- `PREPARSE_EXECUTOR_WORKERS = 2` — thread pool size

## Anthropic Integration
- **Round 1**: Always Haiku (`CLAUDE_FAST_MODEL`) — cheap, handles tool-free queries
- **Round 2+**: Stays on Haiku by default; escalates to Opus (`CLAUDE_MODEL`) only if `FORCE_FAST_MODEL=false` is set and non-terminal tools are invoked
- **Tools**: `search_person`, `get_filmography`, `return_results` (terminal); person-lookup tools removed from schema when filmography is pre-resolved
- **Final-round return-only**: on the last permitted round (`round_num == AGENT_MAX_TOOL_ROUNDS`) the tools list is forced to `[RETURN_RESULTS_TOOL]` in both `rerank` and `rerank_stream`, so Claude must emit results from the candidates it already has instead of burning the round on another lookup and falling through the empty exhaustion path. Fixes the "Claude loops on `search_person` → 0 results" failure where a bare/reference title (e.g. "Perfect Date") is misread as a person name.
- **Anti-hallucination**: `_filter_results()` validates returned titles against candidate + filmography sets using case-insensitive matching
- **Streaming**: `rerank_stream()` uses `_extract_result_objects()` to yield results as JSON chunks arrive

Key config values:
- `SEARCH_CANDIDATES = 15` — vector results passed to Claude
- `SEARCH_DOC_TRUNCATE = 200` — chars of each movie doc sent in prompt
- `AGENT_MAX_TOOL_ROUNDS = 4` — the final round is forced to return-only tools (see above)
- `FORCE_FAST_MODEL = true` — keeps Haiku for all rounds; set `FORCE_FAST_MODEL=false` in `.env` to re-enable Opus escalation

## Tests
Run in venv with: `pytest` from project root.

| Test file | What it covers |
|-----------|---------------|
| `tests/test_query_parser.py` | `parse_query`: year/decade normalization, genre/cert/person extraction, slang triggers, "in the style of" dual-purpose, "more like X" title refs, relative date hints; `apply_hard_filters`: year, genre, cert filtering, conservative empty-metadata handling; `resolve_persons`: TMDB mocked, success/failure/timeout paths |
| `tests/test_claude.py` | Pure functions: `_filter_results` (case-insensitive), `_extract_result_objects`, `_sanitize`, `_build_rerank_prompt` (constraint injection, reference_films block, filmography cap, suppression instruction, XSS sanitization); `rerank_stream`: valid_lower staleness regression (filmography titles discovered via tool calls are streamed through without spurious rejection) and `__filmography` sentinel emission carrying the discovered movie payloads; final-round return-only tools (`rerank` + `rerank_stream` force `[RETURN_RESULTS_TOOL]` on the last round) |
| `tests/test_api.py` | FastAPI endpoints via TestClient; mocks `search_stream`, `vector_count`, `get_model`, `tmdb.warmup` (keeps startup lifespan off the network), `search_movie_by_title`, `fetch_watch_providers`, `watchmode`; security headers; user-agent log capture; `/region` endpoint (IP → country, no-IP fallback); `country` param threading through `/streaming` and `/streaming/batch`; `/ratings` + `/ratings/batch` (ordered scores, provider omission, imdb-id vs search-URL links, zero-vote TMDB omission, fail-soft) |
| `tests/test_main.py` | `_parse_document()` richtext extraction; `search_stream` pre-parse integration: year filter reduces candidates, person timeout degrades gracefully, successful pre-fetch triggers background ingestion, case-insensitive metadata lookup, filmography-only title seeding, runtime cert backfill for not-yet-ingested films, and `__filmography` sentinel seeding for mid-stream tool discoveries (autouse fixture prevents real `fetch_certification` network calls) |
| `tests/test_tmdb.py` | Scoring: `_composite_score`, `select_top_n`, `filter_cast`, `filter_crew`; TMDB lookup: `search_movie_by_title`, `fetch_watch_providers`; certification: `extract_certification`, `fetch_certification`, `get_certification_for_title`; filmography: `get_filmography` includes `poster_path`; `warmup` (primes `/configuration`, no-ops without a key, swallows errors) |
| `tests/test_richtext.py` | `build_richtext()` edge cases |
| `tests/test_watchmode.py` | Watchmode lookup: `search_title` (year matching, not-found), `fetch_providers` (type filtering, deduplication, logo cache, cache key isolation by country); `_load_source_logos` (null URL rejection) |
| `tests/test_omdb.py` | OMDb: score parsers (`_parse_percent`/`_parse_ratio`/`_parse_int`, incl. `N/A`); `fetch_ratings` — fail-soft with no key (no HTTP call), full/partial parse, not-found caches the miss, success cache hit, failures not cached, cache-key isolation by year, TTL expiry |
| `tests/test_frontend_assets.py` | Static parse of `app.html` (no browser): every `SCORE_PROVIDERS` glyph resolves to a real `<symbol>` (a bad id renders an *invisible* glyph), glyph `width` tracks the viewBox aspect at the uniform 20px height, and the Metacritic mark stays the official 3-path monogram rather than regressing to a `<text>` glyph |
| `tests/test_auth.py` | Cookie signing: valid tokens, expiry, tampered signatures, missing secret |
| `tests/test_history.py` | Browser History API: search URL updates, back navigation between searches, overlay open/close via back and ESC, pivot-from-overlay stack, carousel no-op, `?q=` deep links; clear-query button (hidden when empty, shown on input, empties+refocuses on click, re-hides on back to blank). Uses Playwright + a minimal static HTTP server; requires `pip install pytest-playwright && playwright install chromium` |

**No test touches the real Anthropic, TMDB, or Watchmode APIs.** All external calls are mocked at the module level.

**`tests/test_history.py` requires Playwright** (not installed by default). Install once with `pip install pytest-playwright && playwright install chromium`.

## Environment
Required env vars: `ANTHROPIC_API_KEY`, `TMDB_READ_ACCESS_TOKEN`
Optional: `WATCHMODE_API_KEY` (free tier at watchmode.com; enables reliable streaming availability data — falls back to TMDB without it), `OMDB_API_KEY` (free tier at omdbapi.com, 1,000 req/day; enables RT/IMDb/Metacritic scores in the overlay "Reviews" section — degrades to the TMDB score alone without it), `VECTOR_DB` (auto-selects `pinecone` when `RAILWAY_ENVIRONMENT` is set, else `chroma`), `PINECONE_*` keys, `CORS_ORIGINS`, `LOG_DIR` (default `logs`; set to a Railway Volume path for log persistence)

Rate limits are constants in `config.py`, not env vars: `RATE_LIMIT` (`10/minute` on `/recommend`), `STREAMING_RATE_LIMIT` (`30/minute` on `/streaming` endpoints), `RATINGS_RATE_LIMIT` (`30/minute` on `/ratings` endpoints), `REGION_RATE_LIMIT` (`10/minute` on `/region`), `LOGIN_RATE_LIMIT` (`5/minute` on `/admin/login`).

Admin auth (all three required to enable the admin panel; fails closed if any is unset): `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_SECRET_KEY` (random 32-byte hex string — generate with `openssl rand -hex 32`)

## Key Decisions
- Haiku-first strategy: most queries resolve in round 1 without Opus
- `tool_choice={"type": "any"}` forces Claude to always call a tool (prevents prose responses)
- **Query pre-parsing chosen over Haiku extraction**: Haiku structured extraction was prototyped but rejected — P90 latency of 1,736ms is ~9× the 200ms embedding window. Regex is <1ms and $0 cost, covering the vast majority of query patterns.
- **Person pre-fetch runs concurrently with embedding**: the TMDB round-trip (~500ms) overlaps with embedding+vector search (~220ms) via `ThreadPoolExecutor`. Net cost: ~0ms added latency for person queries that resolve within the embedding window.
- **Conditional tool schema**: when all persons in a query are pre-resolved, `search_person` and `get_filmography` are removed from the tools list passed to Claude — a hard guarantee, not a prompt-only suggestion.
- Background thread ingests filmography discoveries into vector DB for future queries (triggered both when Claude calls `get_filmography` directly and when pre-fetch succeeds)
- `rerank` and `rerank_stream` share the same prompt/logic; `rerank` is kept for non-streaming use
- Certification is stored in vector DB metadata at ingest time and displayed immediately from search results; the vector DB is the primary source of truth for ratings — the `/streaming` endpoint handles providers only. **Runtime cert backfill**: filmography films not yet ingested carry no cert in their sparse TMDB payload, so `_batch_fetch_certifications` (`main.py`) fans out `fetch_certification` calls across a bounded pool (`CERT_FETCH_WORKERS`, hard-capped by `CERT_FETCH_TIMEOUT_S`) during metadata seeding; unresolved films stay unrated and are corrected once background ingestion stores their real cert. An empty/unknown cert renders as an explicit muted "Rating not available" note (not a blank span, and never an "NR"-looking chip)
- **"More like this" pivot**: the detail overlay exposes a forked-arrow pivot icon (inline SVG `<symbol>`) next to the title, director, and each cast member. All three call `triggerSearch()` which pre-fills the input and auto-submits via `form.requestSubmit()`. Query formats: `"More movies like [Title] ([Year])"` (hits the `more like X` regex), `"Movies directed by [X]"` (hits person/directing regex), `"Movies starring [X]"` (hits person regex). No original query is appended — the pivot is intentionally standalone.
- **Reference-title anchoring** (retrieval-by-document for "movies like X"): a symmetric encoder (`all-mpnet-base-v2`) can't dereference a bare title token into the film's content, so instead of retrieving on the query text we anchor on the *referenced movie's document*. Flow: `parse_query` extracts `reference_titles` → `resolve_reference_titles()` runs concurrently with embedding and is awaited with a `TITLE_LOOKUP_TIMEOUT_S` (1.5s) cap, resolving each to a TMDB id via `search_movie_by_title` → (if resolution doesn't finish in time — e.g. a cold-start TMDB call — the query silently drops to token retrieval and loses the anchor+guarantee, so the FastAPI lifespan warms the embedding model *and* the TMDB connection at startup, see `tmdb.warmup()`, to keep the first user query off the cold path) → `_fetch_candidates_anchored()` fetches each in-DB reference's stored richtext (`vector_fetch_by_ids`), re-encodes it (deterministic), and queries its neighbors at `ANCHOR_FETCH_DEPTH` (deep, for post-retrieval hard-filter headroom), dropping the anchor itself. Multi-reference queries round-robin `_interleave` the anchor lists. **Retrieval-only, no fusion/RRF**: the qualifier in "like X but funnier" is handled in Claude's rerank, not retrieval — `parse_query` sets `has_soft_qualifier`/`residual_query` (query minus reference span minus structured tokens), which widens the candidate slice to `ANCHOR_CANDIDATES_QUALIFIED` and adds a `prefer:` line to the prompt so Claude reorders *within* the anchor neighborhood. **Cold references** (resolved but not yet ingested) fall back to the query token embedding and are background-ingested (`_ingest_reference_background`, quality gate bypassed via `ingest_single(force=True)`) so they're anchorable next time. **Always-append guarantee (gated to anchored refs)**: a referenced film that is in the vector DB is emitted in the results — appended after the stream (even past the candidate limit) if Claude's rerank didn't surface it (`_reference_guarantee_cards`, rich card from the anchor doc). **Cold references are NOT appended**: we only have a resolved TMDB id, which for an ambiguous phrase may be a wrong match, and appending it would bypass Claude's filtering — so cold refs are background-ingested instead and become guaranteed on the next search.
- **Filmography metadata seeding**: the metadata lookup dicts in `search_stream` (`main.py`) are keyed by lowercased title, built by `_build_metadata_lookups` and seeded by the shared `_seed_filmography_metadata` helper. For filmography movies not in the top-N vector candidates, seeding does a batch `vector_fetch_by_ids` call using the TMDB IDs carried in the filmography payload — this retrieves the full document (overview, genres, director, cast) for any film already in the vector DB regardless of its similarity rank. Films not yet ingested fall back to the sparse poster+year from the TMDB filmography response, with cert backfilled at runtime (see certification note above). **Mid-stream discoveries**: films Claude finds by calling `get_filmography` itself (not pre-resolved) are surfaced from `rerank_stream` as a `{"__filmography": [...movies...]}` sentinel *before* the result items that reference them; `search_stream` feeds that payload through the same `_seed_filmography_metadata` helper so those titles get poster/cert too — without it both fields render blank for tool-discovered titles.
- **Card/overlay poster resilience**: a missing `poster_path` OR a 404 at the TMDB origin both fall back to the film-reel glyph placeholder (`posterFallback()` swaps the broken `<img>` in place, keeping the poster frame) rather than removing/collapsing the poster. An offline audit (`tools/audit_posters.py`, read-only) HEAD-checks every stored poster against the CDN — origin-stale rate measured at ~0.08% (all transient), and ~0.6% of vectors have no `poster_path` stored at all; the placeholder covers both.
- **Clear-query button**: the search input is wrapped in a `.input-wrap` (`position: relative`) so a bare `&times;` button (`#clearBtn`) overlays its right edge. `toggleClearBtn()` shows it only when the input is non-empty and is called from every path that sets `input.value` programmatically (`triggerSearch`, `restoreResults`, the empty-state `popstate` reset, the `?q=` deep-link runner) plus the live `input` event. Clicking it clears the value, re-hides itself, and refocuses. The rule is scoped `.search-form .clear-btn` to out-specify the shared `.search-form button` styling (otherwise it inherits a white box + padding).
- **Mock search**: typing `__mock__` as the query bypasses the backend entirely and renders five hardcoded results through the same `appendResult()` path as live searches. Useful for frontend-only development without a running server.
- **Browser back button / History API**: `app.html` uses `history.pushState` / `popstate` so the back button navigates between searches and overlay detail views. Each search pushes `{type:"search", query, results, scrollY}` with URL `?q=<query>`; opening the detail overlay pushes `{type:"overlay", ..., overlayIndex}` with URL `?q=<query>&detail=N`. Back from overlay closes it (results remain); back from a search restores the previous search's results instantly from the cached state object without a new API call. `triggerSearch()` does NOT call `closeOverlay()` — the form submit handler calls the DOM-only `closeOverlayUI()` directly to avoid a race between `history.back()` and the subsequent `history.pushState()`. Carousel navigation (arrows, swipe) does not modify history. `?q=` deep links auto-run the search on page load.
- **Bot/scraper resilience**: rate limiting via `slowapi` on all public endpoints (`RATE_LIMIT` on `/recommend`, `STREAMING_RATE_LIMIT` on `/streaming` and `/streaming/batch`); security headers (`X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`) added to all responses via middleware; CORS restricted to `GET`/`POST` and `Content-Type` header only; `user_agent` field logged on every `/recommend` request.
- **Location-aware streaming**: `GET /region` resolves the client IP to an ISO 3166-1 alpha-2 country code via ipinfo.io (free tier, process-lifetime cache in `api/geo.py`). The frontend calls it once on page load — before the user types their first query — and stores the result in `userCountry`. All subsequent `/streaming` and `/streaming/batch` calls pass `country=XX` so Watchmode/TMDB return region-specific providers. URL param `?country=XX` overrides detection for testing and VPN workarounds. Watchmode cache key fixed to `providers:{title_id}:{country}` so US and non-US results don't share entries. Watchmode returns 400 (not empty array) when a title has no sources in a region; this is caught separately and logged at DEBUG, not WARNING.
- **Reviews (critic/audience scores)**: the detail overlay has a **"Reviews"** section (just above Streaming) showing review scores as clickable brand-glyph chips in fixed order **RT → IMDb → Metacritic → TMDB**. Scores are an async supplement fetched the same way streaming is — `renderScores()` on overlay open + `batchPrefetchScores()` on stream-complete, sharing a `scoresCache` Map keyed `title|year` (region-independent, no country). Backend `/ratings` combines **OMDb** (RT critic Tomatometer, IMDb, Metacritic — `omdb.py`, optional, fail-soft) + **TMDB** `vote_average` (`fetch_movie_rating`, straight off the search result — no detail call). Absent scores are omitted; a TMDB `vote_average` of 0 ("no votes") is treated as absent, not shown as `0.0`. **Naming**: "certification" = the MPAA G/PG/R chip; "score(s)" = these review scores — the two are deliberately kept lexically distinct (see the certification-naming refactor). **Glyphs**: inline `<svg><symbol>` brand marks — IMDb/TMDB wordmarks and the RT tomato are simplified approximations, while the Metacritic "m" monogram (gold ring + dark field + stepped white "m", square `0 0 40 40` viewBox) carries the **official vector paths**, since its letterform cannot be reproduced with a font glyph. Each is an `<a>` opening the provider page in a new tab (`rel="noopener noreferrer"`) with the provider name as a hover `title`. **Deep links vs search**: IMDb (via OMDb's `imdbID`) and TMDB (via id) link straight to the film's page; RT and Metacritic have no stable per-title id in our data, so their glyphs link to the provider's **search results** for the title (always valid, never a guessed-slug 404). Mock (`__mock__`) results carry an inline `scores` array so the section renders fully without a backend.

## Deferred Ideas
- **Genre-metadata backfill for filtered anchor retrieval**: Hard filters (genre/year/cert) run *post-retrieval* in Python (`apply_hard_filters`) because genres live only in each vector's `document` text, not as filterable vector metadata. For reference-anchor queries with a *structured* qualifier (e.g. "like Inception but animated"), this forces us to over-fetch (`ANCHOR_FETCH_DEPTH`) and filter down — which trades precision for recall as depth grows, and can never manufacture similarity that isn't there. **If such queries prove common in the logs**, the precise fix is to add `genres` to vector metadata and use a metadata-filtered `vector_query` (`where={genre: ...}`) to retrieve the top-N genre-matching neighbors *directly*, full precision, no depth gamble. Follow the `backfill_certifications.py` pattern — it patched certification into existing Pinecone vectors without re-embedding; a genre backfill would be idempotent and identical in shape. Not worth doing until demand is demonstrated; tonal qualifiers ("funnier", "darker") are handled by Claude's rerank and need no metadata.

## Model Use During Development
- Use Sonnet for planning and orchestration, but launch parallel sub-agents with Haiku for execution and research.

## Coding Hygiene
- Discuss and get approval for the technical approach first. Do not proceed with code changes until the technical approach is agreed upon.
- Make sure all changes are captured in a short lived branch before any coding begins
- Code changes should be implemented in a short-lived feature branch.
- Code changes should include clear inline comments for future collaborators and robust logging to help with debugging.
- Do not invoke /commit or create any git commits without explicit user sign-off that they have tested locally and are ready to commit.
- The feature branch should be deleted only after all code changes are committed and merged to main.


## Forbidden Directories
Do not read or modify files in these directories:
- `venv/` — Python environment
- `data/` — raw TMDB JSON dumps
- `embeddings/` — ChromaDB binary data files
- `local/archive/` — older scratch notes and design proposals (already digested into this file)
- `.git/` — use `git` CLI commands instead
