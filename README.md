# MovieMatch

A semantic movie recommendation engine. Describe what you're in the mood for and get ranked matches from a database of top-rated films.

## How it works

1. Movie metadata (plot, keywords, cast, crew) is fetched from the [TMDB API](https://www.themoviedb.org/documentation/api) via the discover endpoint across three sort criteria (rating, popularity, revenue), ranked by a composite Bayesian score, and stored as local JSON files
2. A richtext string is compiled for each movie and embedded using a local [Sentence Transformers](https://www.sbert.net/) model (`all-mpnet-base-v2`)
3. Embeddings are stored in a vector database — [ChromaDB](https://www.trychroma.com/) locally or [Pinecone](https://www.pinecone.io/) in production
4. At search time, the query first runs through a rule-based pre-parser that extracts structured constraints (year/decade, genre, certification, director/actor names, and "like X" title references); the query is then embedded and matched against the database using cosine similarity (for "like X" references, retrieval instead **anchors on the referenced film's own embedding** — see [Reference-title anchoring](#reference-title-anchoring)), candidates violating hard constraints are dropped, and [Claude](https://www.anthropic.com/claude) reranks the survivors via an agentic loop that can optionally look up director or actor filmographies from TMDB before returning ranked results
5. The frontend separately fetches region-aware streaming availability for each result via [Watchmode](https://api.watchmode.com/) (primary) or TMDB (fallback), with the viewer's country inferred from their IP

> **Streaming data is fetched at request time, never stored in the vector database.** Because streaming catalogs change frequently, provider lookups are served live and held only in a process-local in-memory cache (24-hour TTL, keyed by title and country) to conserve the Watchmode free-tier budget. The cache is volatile — it is lost on process restart. The vector database stores movie metadata and MPAA certifications only, not streaming availability.

## Reference-title anchoring

When a query references a specific film — "movies like Krull", "like Inception but funnier", "something like Parasite" — retrieval works differently. The embedding model is symmetric and can't dereference a bare title token into the film's content (embedding the word "Inception" lands nearer the common noun than Christopher Nolan's film), so instead of matching on the query text, retrieval **anchors on the referenced film's own stored document** and returns its nearest neighbors (retrieval-by-example).

- Any `like <title>` phrasing is treated as a reference by default (capitalization optional); the *verb* sense — "**I** like", "**we** like" — is the carved-out exception.
- The title is resolved to a TMDB id concurrently with query embedding; the film's stored richtext is then re-embedded and used as the retrieval vector. Multiple references ("like X and Y") are round-robin merged.
- A soft qualifier ("…but funnier") is **not** folded into retrieval — Claude's rerank reorders *within* the anchor's neighborhood toward the qualifier, and the candidate slice widens to give it room.
- If the referenced film isn't indexed yet, retrieval falls back to the query text and the film is ingested in the background (quality gate bypassed), so it anchors on the next search.
- A referenced film that is already indexed is **guaranteed** to appear in the results, appended after Claude's ranking if it wasn't already surfaced.

## The agentic rerank loop

Step 4 above is an agentic loop in `claude.py` that comes in two twins — `rerank_stream` (streaming) and `rerank` (non-streaming), described under [Streaming vs. non-streaming](#streaming-vs-non-streaming) below. Rather than a single prompt, Claude runs a bounded tool-use conversation over the vector-search candidates, optionally pulling in a director's or actor's filmography before committing to a final ranked list.

### Tools

Three tools are exposed. `tool_choice={"type": "any"}` forces Claude to always call a tool, which prevents free-text prose replies and keeps output machine-parseable.

| Tool | Terminal? | Purpose |
|------|-----------|---------|
| `search_person` | No | Look up a person (director/actor/writer) on TMDB by name → ID + known works |
| `get_filmography` | No | Fetch a person's movie credits by TMDB ID (`department` = `directing` or `cast`) |
| `return_results` | Yes | Emit the final reranked, filtered recommendations; ends the loop |

A `get_filmography` call has a side effect: any of its films not already indexed are ingested into the vector DB in a background daemon thread (see [Incremental updates at runtime](#incremental-updates-at-runtime)), and their titles are added to the set of valid titles Claude is allowed to return.

### The loop

Each round, Claude either calls a non-terminal tool (whose result is fed back into the conversation for the next round) or calls `return_results` to finish. The loop is capped at `AGENT_MAX_TOOL_ROUNDS = 4` rounds — a typical person query resolves in `search_person → get_filmography → return_results`.

```
Round 1 ─ Haiku ─┬─ return_results ──────────────► done (most queries end here)
                 └─ search_person / get_filmography
                        │  (feed tool result back in)
                        ▼
Round 2 ─ Haiku* ─┬─ return_results ──────────────► done
                  └─ another non-terminal tool → Round 3 …  (up to 4 rounds)
```

\* By default round 2+ also stays on Haiku — see model routing below.

If all persons named in the query were already resolved by the concurrent pre-fetch in `query_parser.py`, the two person-lookup tools are **removed from the schema entirely** (only `return_results` remains). This is a hard guarantee — not a prompt instruction — that Claude won't burn a round on a redundant TMDB call.

### When Haiku vs Opus is used

Two models are configured in `config.py`:

- `CLAUDE_FAST_MODEL = claude-haiku-4-5` — cheap, fast; used for **round 1 always** and, by default, every subsequent round.
- `CLAUDE_MODEL = claude-opus-4-6` — used only when escalation is enabled *and* a non-terminal tool is invoked.

Routing is governed by the `FORCE_FAST_MODEL` env var:

| `FORCE_FAST_MODEL` | Round 1 | Round 2+ (after a non-terminal tool call) |
|--------------------|---------|-------------------------------------------|
| `true` (**default**) | Haiku | Haiku — never escalates |
| `false` | Haiku | Opus |

In other words: **round 1 is always Haiku**, and the vast majority of queries never leave Haiku. Opus is only reached when you opt in with `FORCE_FAST_MODEL=false` *and* the query is complex enough that Claude reaches for a person/filmography tool before returning results. Token usage is tracked per-model (`haiku_*` / `opus_*` counters) and the model path is logged per request (e.g. `models=haiku→haiku` or `haiku→opus`).

### Anti-hallucination

Claude may only return titles that exist in the candidate set or in a filmography it looked up. Every returned title is validated (case-insensitively) against that allowed set in `_filter_results`; fabricated titles are dropped and logged. In the streaming path this check runs on each result as it arrives, with the allowed set rebuilt after every `get_filmography` round so newly discovered films aren't rejected.

### Streaming vs. non-streaming

The rerank loop exists as two twins that share the same prompt (`_build_rerank_prompt`), the same bounded loop, the same [Haiku/Opus routing](#when-haiku-vs-opus-is-used), the same [conditional tool schema](#the-loop), and the same `valid_titles` seeding. They differ only in how Claude's final answer is consumed:

| | **Streaming** — `rerank_stream` | **Non-streaming** — `rerank` |
|--|--|--|
| **Used by** | `main.py:search_stream` → `POST /recommend` (production path) | `main.py:search` → the `python main.py` CLI |
| **Claude call** | `messages.stream(...)` | blocking `messages.create` (via `_call_claude`) |
| **Output shape** | a generator that yields result dicts one at a time, then a `{"__usage": {...}}` sentinel | a single `(results, usage)` tuple returned once the loop finishes |
| **Result delivery** | `return_results`' tool JSON is parsed incrementally by `_extract_result_objects` and each complete result object is yielded the moment it's parseable, so the frontend renders cards progressively as they arrive | the entire `return_results` call is awaited, then filtered and returned in one shot |
| **Anti-hallucination** | inline per-result check against `valid_lower` as each object streams in (rebuilt after every `get_filmography` round), **plus** an end-of-round `_filter_results` safety net that yields any valid result the incremental parser missed | a single `_filter_results` pass over the complete result list |
| **Errors** | `RateLimitError` / `InternalServerError` / `APIConnectionError` are caught and surfaced as an `error` field on the usage sentinel (the SSE endpoint turns it into a `{"type": "error"}` event) rather than raising | propagate to the caller |

The streaming twin is what the web app uses: `search_stream` enriches each streamed result with poster/certification/metadata from the lowercased lookup dicts and re-yields it, and the SSE endpoint (`api/routes/search.py`) drives the generator in a thread executor, emitting each result as a `data:` event and logging usage/cost from the final `__meta` payload. The non-streaming twin is kept as a simpler, easier-to-reason-about equivalent for the CLI and ad-hoc testing — no SSE, no partial-JSON parsing, no thread executor.

## Setup

### Prerequisites

- Python 3.11+
- A [TMDB API read access token](https://developer.themoviedb.org/docs/getting-started)
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes the Pinecone client, so no separate install is needed for production. To run the test suite, also install the dev dependencies (Playwright is only needed for the browser History tests):

```bash
pip install -r requirements-dev.txt
playwright install chromium   # one-time, for tests/test_history.py
```

### Environment variables

Create a `.env` file in the project root:

```
TMDB_READ_ACCESS_TOKEN=your_tmdb_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

To use Pinecone instead of ChromaDB, add:

```
VECTOR_DB=pinecone
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=your_index_name
```

> `VECTOR_DB` auto-selects `pinecone` when `RAILWAY_ENVIRONMENT` is set (production) and `chroma` otherwise; set it explicitly to override.

### Optional environment variables

| Variable | Purpose |
|----------|---------|
| `WATCHMODE_API_KEY` | Enables [Watchmode](https://api.watchmode.com/) as the primary streaming-availability source (free tier). Falls back to TMDB when unset. |
| `CORS_ORIGINS` | Comma-separated allowed origins (default `http://localhost:8000,http://127.0.0.1:8000`). |
| `LOG_DIR` | Directory for request logs (default `logs`; point at a persistent volume in production). |
| `FORCE_FAST_MODEL` | When `true` (default) Claude uses Haiku for all rounds; set `false` to re-enable Opus escalation on tool use. |

> Rate limits are compile-time constants in `config.py`, not environment variables: `RATE_LIMIT` (`10/minute` on `/recommend`), `STREAMING_RATE_LIMIT` (`30/minute` on the `/streaming` endpoints), `REGION_RATE_LIMIT` (`10/minute` on `/region`), and `LOGIN_RATE_LIMIT` (`5/minute` on `/admin/login`). Edit `config.py` to change them.

### Admin panel auth

The admin panel and `/admin/*` API are gated by an HMAC-signed session cookie. All three variables must be set to enable it — if any is unset, admin access fails closed:

```
ADMIN_USERNAME=your_admin_user
ADMIN_PASSWORD=your_admin_password
ADMIN_SECRET_KEY=<random 32-byte hex string, e.g. `openssl rand -hex 32`>
```

## Initialization

Run the pipeline once to fetch metadata, build embeddings, and populate the database:

```bash
source venv/bin/activate
python pipeline.py
```

You will be prompted for the number of movies to index (there is no default — enter a value at the prompt). A dataset of **5,000 movies** is recommended and takes approximately 50 minutes (mostly TMDB API calls). If the pipeline is interrupted, re-running it will skip already-ingested movies and resume from where it left off.

### Incremental updates at runtime

Beyond this one-time bulk initialization, the embeddings database grows **lazily at search time**. When a query names a director or actor, their filmography is looked up from TMDB (either by Claude calling the `get_filmography` tool, or by the concurrent person pre-fetch in the query parser). Any films in that filmography that aren't already indexed are ingested in a **background daemon thread**, so the current search returns without waiting — the newly added movies simply become searchable for subsequent queries. A referenced film ("movies like X") that isn't yet indexed is ingested the same way, except its quality gate is bypassed — the user named it explicitly — so it becomes anchorable on the next search.

Each lazily discovered movie passes through the same steps as bulk ingestion (`ingest_single` in `pipeline.py`):

1. A **quality gate** drops films below the `MIN_INGEST_VOTE_AVERAGE` / `MIN_INGEST_VOTE_COUNT` thresholds in `config.py`.
2. Full metadata is fetched from TMDB, a richtext string is compiled, and it's embedded and upserted into the vector database.
3. `index.json` is updated (writes are serialized by a lock, so concurrent background ingestions are safe).

The process is **idempotent and de-duplicated**: movies already in the dataset are skipped, an in-flight set prevents the same movie from being ingested twice concurrently, and each filmography lookup is capped at `FILMOGRAPHY_INGEST_LIMIT` movies.

## Running the server

```bash
source venv/bin/activate
uvicorn api.app:app --reload
```

Then open <http://localhost:8000/> in a browser. The frontend (`app.html`) is served by the app's root route and uses relative API paths, so it must be loaded from the server — opening the `app.html` file directly (`file://`) will break all API calls.

## API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/recommend` | — | Get movie recommendations for a query (SSE stream) |
| `GET` | `/region` | — | Infer the client's ISO country code from their IP |
| `GET` | `/streaming` | — | Streaming providers for one title in a region |
| `POST` | `/streaming/batch` | — | Streaming providers for up to 10 titles at once |
| `GET` | `/health` | — | Health check |
| `GET`/`POST` | `/admin/login` | — | Admin login page / credential submission |
| `GET` | `/admin/logout` | — | Clear the session cookie |
| `POST` | `/admin/initialize` | ✔ | Trigger pipeline initialization |
| `GET` | `/admin/status` | ✔ | Database stats and initialization status |
| `GET` | `/admin/logs` | ✔ | Recent search logs and aggregate metrics |
| `GET` | `/admin/watchmode` | ✔ | Watchmode API usage stats |

> **Note:** Endpoints marked ✔ require a valid admin session cookie (see [Admin panel auth](#admin-panel-auth)). They fail closed if the admin credentials are not configured.

### POST /recommend

Returns a [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) stream. Each event is a JSON object on a `data:` line.

```
// Request
POST /recommend
{ "query": "a feel-good coming of age story" }

// Stream events
data: {"type": "result", "title": "Stand by Me", "explanation": "A nostalgic and heartfelt coming of age story...", "movie_poster": "/eze1b4v9GSnwNLpSaO4gqJ7FaBT.jpg"}

data: {"type": "done", "result_count": 5}

// On no matches
data: {"type": "done", "result_count": 0, "message": "No relevant matches found."}

// On error
data: {"type": "error", "message": "Could not connect to the AI service. Please try again."}
```

## Project structure

```
api/
  app.py               # FastAPI app factory, CORS, security headers, lifespan; login/logout routes
  auth.py              # HMAC-SHA256 session cookie signing; require_admin dependency
  geo.py               # Client IP → ISO country code via ipinfo.io
  limiter.py           # Rate limiter (slowapi)
  routes/
    search.py          # POST /recommend — SSE streaming endpoint
    admin.py           # /admin/initialize, /status, /logs, /watchmode (all admin-only)
    streaming.py       # GET /region, GET /streaming, POST /streaming/batch — region-aware providers
app.html               # Frontend UI
admin.html             # Admin panel UI
login.html             # Admin login page
pipeline.py            # Initialization pipeline (ingest → embed → store); lazy single-movie ingestion
main.py                # Core search logic (pre-parse → embed query → retrieve → hard filter → rerank)
query_parser.py        # Rule-based query pre-parser: constraint extraction, hard filters, person resolution
claude.py              # Claude agentic loop: tool use (person search, filmography) + result reranking
embeddings.py          # Embedding generation and batch upsert
tmdb.py                # TMDB API integration
watchmode.py           # Watchmode API integration — primary streaming-provider source
richtext.py            # Movie metadata → text for embedding
db.py                  # ChromaDB/Pinecone and Sentence Transformer singletons
config.py              # Configuration constants (dataset size, scoring weights, rate limits, quality thresholds, model selection)
logger.py              # JSON request logging; rotating file locally, stdout on Railway
migrate_to_pinecone.py # One-off script: copies all vectors from Chroma to Pinecone
backfill_certifications.py # One-off script: patches MPAA certifications into existing Pinecone metadata
```
