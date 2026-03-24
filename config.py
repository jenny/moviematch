import os
from dotenv import load_dotenv

load_dotenv()

# TMDB
TMDB_KEY = os.getenv("TMDB_READ_ACCESS_TOKEN")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_KEY}" if TMDB_KEY else ""
}
TMDB_MIN_VOTE_COUNT = 200
TMDB_RATE_LIMIT_SLEEP = 0.3  # seconds; keeps requests under TMDB's 40 req/10s limit
# Composite ranking weights (must sum to 1.0): Bayesian weighted rating vs. popularity
SCORE_WEIGHT_RATING = 0.6
SCORE_WEIGHT_POPULARITY = 0.4
# Quality thresholds for lazy ingestion of tool-discovered movies
MIN_INGEST_VOTE_AVERAGE = 6.0
MIN_INGEST_VOTE_COUNT = 100

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths
DATA_DIR = "data"
CHROMA_PATH = "./embeddings/chroma_db"

# ChromaDB
COLLECTION_NAME = "movies"

# Embedding model
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 64

# Claude
CLAUDE_MODEL = "claude-opus-4-6"          # used when tools are invoked
CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001"  # used for round 1 and tool-free queries
AGENT_MAX_TOOL_ROUNDS = 4  # search_person → get_filmography (×1-2) → return_results
# Token pricing in USD/token — update if Anthropic changes prices
HAIKU_INPUT_PRICE  = 0.80  / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00  / 1_000_000
OPUS_INPUT_PRICE   = 15.00 / 1_000_000
OPUS_OUTPUT_PRICE  = 75.00 / 1_000_000

# Filtering
CAST_LIMIT = 30
RICHTEXT_CAST_LIMIT = 5
CREW_JOBS = {"Director", "Executive Producer", "Producer"}

# Search
SEARCH_CANDIDATES = 20
SEARCH_DOC_TRUNCATE = 300

# Rate limiting (slowapi format, e.g. "10/minute")
RATE_LIMIT = "10/minute"

# CORS — comma-separated list of allowed origins; restrict in production
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")]
