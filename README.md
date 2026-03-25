# MovieMatch

A semantic movie recommendation engine. Describe what you're in the mood for and get ranked matches from a database of top-rated films.

## How it works

1. Movie metadata (plot, keywords, cast, crew) is fetched from the [TMDB API](https://www.themoviedb.org/documentation/api) via the discover endpoint across three sort criteria (rating, popularity, revenue), ranked by a composite Bayesian score, and stored as local JSON files
2. A richtext string is compiled for each movie and embedded using a local [Sentence Transformers](https://www.sbert.net/) model (`all-mpnet-base-v2`)
3. Embeddings are stored in a vector database — [ChromaDB](https://www.trychroma.com/) locally or [Pinecone](https://www.pinecone.io/) in production
4. At search time, the query is embedded and matched against the database using cosine similarity; [Claude](https://www.anthropic.com/claude) then uses an agentic loop to optionally look up director or actor filmographies from TMDB before returning ranked results

## Setup

### Prerequisites

- Python 3.11+
- A [TMDB API read access token](https://developer.themoviedb.org/docs/getting-started)
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install requests python-dotenv anthropic sentence-transformers chromadb fastapi uvicorn

# For Pinecone (production):
pip install pinecone
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

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/recommend` | Get movie recommendations for a query |
| `GET` | `/health` | Health check |
| `POST` | `/admin/initialize` | Trigger pipeline initialization |
| `GET` | `/admin/status` | Get database stats and initialization status |
| `GET` | `/admin/logs` | Retrieve recent search logs |

> **Note:** The `/admin/*` endpoints have no authentication. Do not expose them publicly in production — restrict them via your reverse proxy or network configuration.

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
  app.py               # FastAPI app factory, CORS, lifespan
  limiter.py           # Rate limiter (slowapi)
  routes/
    search.py          # POST /recommend — SSE streaming endpoint
    admin.py           # POST /admin/initialize, GET /admin/status, GET /admin/logs
app.html               # Frontend UI
pipeline.py            # Initialization pipeline (ingest → embed → store); lazy single-movie ingestion
main.py                # Core search logic (embed query → retrieve → rerank)
claude.py              # Claude agentic loop: tool use (person search, filmography) + result reranking
embeddings.py          # Embedding generation and batch upsert
tmdb.py                # TMDB API integration
richtext.py            # Movie metadata → text for embedding
db.py                  # ChromaDB/Pinecone and Sentence Transformer singletons
config.py              # Configuration constants (dataset size, scoring weights, rate limits, quality thresholds, model selection)
logger.py              # JSON request logging; rotating file locally, stdout on Railway
migrate_to_pinecone.py # One-off script: copies all vectors from Chroma to Pinecone
```
