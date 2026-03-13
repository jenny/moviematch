# moviematch

A local CLI tool for semantic movie search. Query by mood, theme, or plot description and get ranked matches from a database of up to 500 top-rated films.

## How it works

1. Movie metadata (plot, keywords) is fetched from the [TMDB API](https://www.themoviedb.org/documentation/api) via the discover endpoint, sorted by rating with a minimum vote threshold, and stored as local JSON files
2. A richtext string is compiled for each movie and embedded using a local [Sentence Transformers](https://www.sbert.net/) model (`all-mpnet-base-v2`)
3. Embeddings are stored in a local [ChromaDB](https://www.trychroma.com/) vector database
4. Queries are embedded at search time and matched against the database using cosine similarity

## Setup

### Prerequisites

- Python 3.11+
- A [TMDB API read access token](https://developer.themoviedb.org/docs/getting-started)
- An [Anthropic API key](https://console.anthropic.com/) (for Claude integration)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install requests python-dotenv anthropic sentence-transformers chromadb
```

### Environment variables

Create a `.env` file in the project root:

```
TMDB_READ_ACCESS_TOKEN=your_tmdb_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Initialization

Run these scripts once in order to populate the local database:

```bash
# 1. Fetch movie metadata from TMDB (prompts for number of movies, up to 500)
python init_data_json.py

# 2. Build richtext strings for embedding
python init_richtext.py

# 3. Generate embeddings and store in ChromaDB
python init_embeddings.py
```

## Usage

```bash
python search.py
```

Enter a natural language query when prompted, e.g.:
- `"a heist movie with a twist ending"`
- `"feel-good films about friendship"`
- `"dark psychological thriller"`

## Project structure

```
init_data_json.py    # Fetch and filter movie metadata from TMDB
init_richtext.py     # Build richtext strings from movie metadata
init_embeddings.py   # Generate and store embeddings in ChromaDB
search.py            # CLI search interface
prompt_claude.py     # Claude API integration
```
