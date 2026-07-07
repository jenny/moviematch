# MovieMatch

A semantic movie recommendation engine. Describe what you're in the mood for and get ranked matches from a database of top-rated films.

## How it works

1. Movie metadata (plot, keywords, cast, crew) is fetched from the [TMDB API](https://www.themoviedb.org/documentation/api) via the discover endpoint across three sort criteria (rating, popularity, revenue), ranked by a composite Bayesian score, and stored as local JSON files
2. A richtext string is compiled for each movie and embedded using a local [Sentence Transformers](https://www.sbert.net/) model (`all-mpnet-base-v2`)
3. Embeddings are stored in a vector database — [ChromaDB](https://www.trychroma.com/) locally or [Pinecone](https://www.pinecone.io/) in production
4. At search time, the query first runs through a rule-based pre-parser that extracts structured constraints (year/decade, genre, certification, director/actor names, "similar to X" title references); the query is then embedded and matched against the database using cosine similarity, candidates violating hard constraints are dropped, and [Claude](https://www.anthropic.com/claude) reranks the survivors via an agentic loop that can optionally look up director or actor filmographies from TMDB before returning ranked results
5. The frontend separately fetches region-aware streaming availability for each result via [Watchmode](https://api.watchmode.com/) (primary) or TMDB (fallback), with the viewer's country inferred from their IP

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
| `RATE_LIMIT` | slowapi limit on `/recommend` (default `10/minute`). |
| `STREAMING_RATE_LIMIT` | slowapi limit on the `/streaming` endpoints (default `30/minute`). |
| `LOG_DIR` | Directory for request logs (default `logs`; point at a persistent volume in production). |
| `FORCE_FAST_MODEL` | When `true` (default) Claude uses Haiku for all rounds; set `false` to re-enable Opus escalation on tool use. |

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

You will be prompted for the number of movies to index. The default dataset is **5,000 movies**, which takes approximately 50 minutes (mostly TMDB API calls). If the pipeline is interrupted, re-running it will skip already-ingested movies and resume from where it left off.

## Running the server

```bash
source venv/bin/activate
uvicorn api.app:app --reload
```

Then open `app.html` in a browser.

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
