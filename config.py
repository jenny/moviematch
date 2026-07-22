import os
from dotenv import load_dotenv

load_dotenv()

# TMDB
TMDB_KEY = os.getenv("TMDB_READ_ACCESS_TOKEN")

# Watchmode — primary source for streaming availability (watchmode.com, free tier: 1,000 req/month)
WATCHMODE_API_KEY = os.getenv("WATCHMODE_API_KEY")
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
# Fail fast if the weights are misconfigured — _composite_score assumes they sum to 1.0.
if abs((SCORE_WEIGHT_RATING + SCORE_WEIGHT_POPULARITY) - 1.0) > 1e-9:
    raise ValueError(
        f"SCORE_WEIGHT_RATING ({SCORE_WEIGHT_RATING}) + SCORE_WEIGHT_POPULARITY "
        f"({SCORE_WEIGHT_POPULARITY}) must sum to 1.0."
    )
# Quality thresholds for lazy ingestion of tool-discovered movies
MIN_INGEST_VOTE_AVERAGE = 6.0
MIN_INGEST_VOTE_COUNT = 100

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths
DATA_DIR = "data"
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Vector DB backend: "chroma" for local, "pinecone" for production
# Auto-detects Railway via RAILWAY_ENVIRONMENT; can always be overridden with VECTOR_DB
VECTOR_DB = os.getenv("VECTOR_DB", "pinecone" if os.getenv("RAILWAY_ENVIRONMENT") else "chroma")

# ChromaDB (used when VECTOR_DB=chroma)
CHROMA_PATH = "./embeddings/chroma_db"
COLLECTION_NAME = "movies"

# Pinecone (used when VECTOR_DB=pinecone)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "moviematch")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Embedding model
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # output dimension of all-mpnet-base-v2
EMBEDDING_BATCH_SIZE = 64

# Claude
CLAUDE_MODEL = "claude-opus-4-6"          # used when tools are invoked (if escalation enabled)
CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001"  # used for round 1 and tool-free queries
AGENT_MAX_TOOL_ROUNDS = 4  # search_person → get_filmography (×1-2) → return_results
# When True, Haiku is used for all rounds — no Opus escalation after tool calls.
# Set FORCE_FAST_MODEL=false in .env to re-enable Opus for tool-use rounds.
FORCE_FAST_MODEL = os.getenv("FORCE_FAST_MODEL", "true").lower() == "true"
FILMOGRAPHY_INGEST_LIMIT = 10  # max movies to lazily ingest per get_filmography call
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
SEARCH_CANDIDATES = 15
SEARCH_DOC_TRUNCATE = 200

# Reference-title anchoring — "movies like X" retrieval-by-example.
# When a referenced title resolves to a movie already in the vector DB, we retrieve
# by that movie's DOCUMENT embedding (its stored richtext) instead of the query token
# embedding, because a symmetric encoder (all-mpnet-base-v2) can't dereference a bare
# title into the film's content. See CLAUDE.md "Reference-title anchoring".
ANCHOR_FETCH_DEPTH = 50          # neighbors fetched per anchor BEFORE hard filters. Deep enough
                                 # to leave a usable pool after post-retrieval genre/year/cert
                                 # filtering (genres aren't filterable vector metadata — see the
                                 # "Deferred Ideas" note in CLAUDE.md for the precise fix).
ANCHOR_CANDIDATES_QUALIFIED = 25 # candidates handed to Claude when a reference query carries a soft
                                 # qualifier ("like X but funnier"): a wider slice gives the rerank
                                 # room to reorder within the anchor neighborhood. Non-qualified
                                 # (pivot) reference queries use SEARCH_CANDIDATES.

# Query pre-parsing
PERSON_LOOKUP_TIMEOUT_S = 1.5   # max seconds to wait for concurrent person TMDB pre-fetch;
                                 # tight on purpose — TMDB p50 ~500ms, p95 ~1s; limits worst-case streaming block
TITLE_LOOKUP_TIMEOUT_S = 1.5    # max seconds to wait for concurrent reference-title TMDB resolution;
                                 # mirrors PERSON_LOOKUP_TIMEOUT_S — bounds worst-case streaming block
PREPARSE_EXECUTOR_WORKERS = 2   # ThreadPoolExecutor pool size for concurrent person/title lookups

# Runtime certification backfill for filmography films not yet ingested (their sparse
# TMDB filmography payload carries no cert). Fanned out concurrently so N films add
# ~one TMDB round-trip, not N; bounded so a slow/hung TMDB can't stall the stream.
CERT_FETCH_WORKERS = 8          # concurrent fetch_certification calls
CERT_FETCH_TIMEOUT_S = 1.5      # max total wait; films that don't resolve stay unrated

# Rate limiting (slowapi format, e.g. "10/minute")
RATE_LIMIT = "10/minute"           # /recommend — hits Anthropic API, keep tight
STREAMING_RATE_LIMIT = "30/minute" # /streaming endpoints — Watchmode/TMDB only
REGION_RATE_LIMIT = "10/minute"    # /region — triggers an external ipinfo lookup
LOGIN_RATE_LIMIT = "5/minute"      # /admin/login — throttles password brute-force

# Streaming region — ISO 3166-1 alpha-2 country code used when no region is detected
DEFAULT_STREAMING_REGION = "US"

# Admin auth — set all three in .env to enable the admin panel
# ADMIN_SECRET_KEY should be a random 32-byte hex string (e.g. from `openssl rand -hex 32`)
ADMIN_USERNAME  = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD  = os.getenv("ADMIN_PASSWORD", "")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "")

# CORS — comma-separated list of allowed origins; restrict in production
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if o.strip()]

# Richtext field prefixes — shared between richtext.py (writer) and main.py (parser)
RICHTEXT_PREFIX_PLOT = "Plot: "
RICHTEXT_PREFIX_GENRES = "Genres: "
RICHTEXT_PREFIX_DIRECTOR = "Director: "
RICHTEXT_PREFIX_CAST = "Top Cast: "


def validate_config() -> None:
    """Fail fast at startup if required environment variables are missing.

    Called from the FastAPI lifespan before serving traffic. Pinecone keys are
    intentionally excluded — they're only required when VECTOR_DB=pinecone and
    are validated lazily by _get_pinecone_index() with a clear error message.
    """
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not TMDB_KEY:
        missing.append("TMDB_READ_ACCESS_TOKEN")
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Check your .env file."
        )
