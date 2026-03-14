# MovieMatch

A semantic movie recommendation engine. Describe what you're in the mood for and get ranked matches from a database of top-rated films.

## How it works

1. Movie metadata (plot, keywords, cast, crew) is fetched from the [TMDB API](https://www.themoviedb.org/documentation/api) via the discover endpoint, sorted by rating with a minimum vote threshold, and stored as local JSON files
2. A richtext string is compiled for each movie and embedded using a local [Sentence Transformers](https://www.sbert.net/) model (`all-mpnet-base-v2`)
3. Embeddings are stored in a local [ChromaDB](https://www.trychroma.com/) vector database
4. At search time, the query is embedded and matched against the database using cosine similarity; the top candidates are reranked by [Claude](https://www.anthropic.com/claude) for relevance

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
```

### Environment variables

Create a `.env` file in the project root:

```
TMDB_READ_ACCESS_TOKEN=your_tmdb_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Initialization

Run the pipeline once to fetch metadata, build embeddings, and populate the database:

```bash
python pipeline.py
```

You will be prompted for the number of movies to index (up to 500).

## Running the server

```bash
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

### POST /recommend

```json
// Request
{ "query": "a feel-good coming of age story" }

// Response
{
  "query": "a feel-good coming of age story",
  "results": [
    {
      "title": "Stand by Me",
      "explanation": "A nostalgic and heartfelt coming of age story...",
      "movie_poster": "/eze1b4v9GSnwNLpSaO4gqJ7FaBT.jpg"
    }
  ]
}
```

## Project structure

```
api/
  app.py               # FastAPI app
  routes/
    search.py          # POST /recommend endpoint
    admin.py           # POST /admin/initialize, GET /admin/status endpoints
app.html               # Frontend UI
pipeline.py            # Initialization pipeline (ingest → embed → store)
search.py              # Core search logic (embed query → retrieve → rerank)
claude.py              # Claude reranking integration
embeddings.py          # Embedding generation and ChromaDB upsert
tmdb.py                # TMDB API integration
richtext.py            # Movie metadata → text for embedding
db.py                  # ChromaDB and Sentence Transformer singletons
config.py              # Configuration constants
```
