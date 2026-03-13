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
TMDB_RATE_LIMIT_SLEEP = 0.25

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
CLAUDE_MODEL = "claude-opus-4-6"

# Filtering
CAST_LIMIT = 30
RICHTEXT_CAST_LIMIT = 5
CREW_JOBS = {"Director", "Executive Producer", "Producer"}

# Search
SEARCH_CANDIDATES = 20
SEARCH_DOC_TRUNCATE = 300
