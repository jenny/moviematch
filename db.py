import logging
import threading
import time

import chromadb
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config import (
    MODEL_NAME,
    VECTOR_DB,
    CHROMA_PATH, COLLECTION_NAME,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

_PINECONE_UPSERT_BATCH_SIZE = 100

_model = None
_chroma_collection = None
_pinecone_index = None
_model_lock = threading.Lock()
_chroma_lock = threading.Lock()
_pinecone_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    _model = SentenceTransformer(MODEL_NAME)
                except Exception as e:
                    raise RuntimeError(f"Failed to load Sentence Transformers model: {e}")
    return _model


# --- ChromaDB backend ---

def _get_chroma_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is None:
        with _chroma_lock:
            if _chroma_collection is None:
                try:
                    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
                    _chroma_collection = chroma.get_or_create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=None,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to connect to ChromaDB: {e}")
    return _chroma_collection


# --- Pinecone backend ---

def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                if not PINECONE_API_KEY:
                    raise ValueError("PINECONE_API_KEY is not set. Check your .env file.")
                try:
                    pc = Pinecone(api_key=PINECONE_API_KEY)
                    existing = {idx.name for idx in pc.list_indexes()}
                    if PINECONE_INDEX_NAME not in existing:
                        logger.info(
                            f"Creating Pinecone index '{PINECONE_INDEX_NAME}' "
                            f"(dim={EMBEDDING_DIMENSION}, metric=cosine)..."
                        )
                        pc.create_index(
                            name=PINECONE_INDEX_NAME,
                            dimension=EMBEDDING_DIMENSION,
                            metric="cosine",
                            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                        )
                        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
                            time.sleep(1)
                        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready.")
                    _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                except Exception as e:
                    raise RuntimeError(f"Failed to connect to Pinecone: {e}")
    return _pinecone_index


# --- Unified interface ---

def vector_count() -> int:
    """Return the total number of vectors in the configured backend."""
    if VECTOR_DB == "pinecone":
        return _get_pinecone_index().describe_index_stats().total_vector_count
    return _get_chroma_collection().count()


def vector_query(vector: list[float], top_k: int) -> list[dict]:
    """Query the vector store. Returns [{title, movie_poster, document}, ...]."""
    if VECTOR_DB == "pinecone":
        results = _get_pinecone_index().query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {
                "title": m.metadata.get("title", ""),
                "movie_poster": m.metadata.get("movie_poster", ""),
                "certification": m.metadata.get("certification", ""),
                "document": m.metadata.get("document", ""),
            }
            for m in results.matches
            if m.metadata and m.metadata.get("title")
        ]
    results = _get_chroma_collection().query(query_embeddings=[vector], n_results=top_k)
    return [
        {
            "title": meta.get("title", ""),
            "movie_poster": meta.get("movie_poster", ""),
            "certification": meta.get("certification", ""),
            "document": doc,
        }
        for meta, doc in zip(results["metadatas"][0], results["documents"][0])
        if meta.get("title")
    ]


def vector_fetch_by_ids(ids: list[str]) -> list[dict]:
    """Fetch records by ID without a similarity query.

    Returns [{title, movie_poster, certification, document}, ...].
    IDs not found in the store are silently omitted.
    """
    if not ids:
        return []
    if VECTOR_DB == "pinecone":
        result = _get_pinecone_index().fetch(ids=ids)
        return [
            {
                "title": v.metadata.get("title", ""),
                "movie_poster": v.metadata.get("movie_poster", ""),
                "certification": v.metadata.get("certification", ""),
                "document": v.metadata.get("document", ""),
            }
            for v in result.vectors.values()
            if v.metadata and v.metadata.get("title")
        ]
    result = _get_chroma_collection().get(ids=ids, include=["metadatas", "documents"])
    return [
        {
            "title": meta.get("title", ""),
            "movie_poster": meta.get("movie_poster", ""),
            "certification": meta.get("certification", ""),
            "document": doc or "",
        }
        for meta, doc in zip(result["metadatas"], result["documents"])
        if meta and meta.get("title")
    ]


def vector_upsert_batch(vectors: list[dict]) -> None:
    """Upsert vectors into the configured backend.

    Each dict must have: id (str), values (list[float]),
    title (str), movie_poster (str), document (str).
    """
    if not vectors:
        return
    actual_dim = len(vectors[0]["values"])
    if actual_dim != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: got {actual_dim}, expected {EMBEDDING_DIMENSION}. "
            f"Check that MODEL_NAME in config.py matches the model currently in use."
        )
    if VECTOR_DB == "pinecone":
        index = _get_pinecone_index()
        pinecone_vectors = [
            {
                "id": v["id"],
                "values": v["values"],
                "metadata": {
                    "title": v["title"],
                    "movie_poster": v["movie_poster"],
                    "certification": v.get("certification", ""),
                    "document": v["document"],
                },
            }
            for v in vectors
        ]
        for i in range(0, len(pinecone_vectors), _PINECONE_UPSERT_BATCH_SIZE):
            index.upsert(vectors=pinecone_vectors[i:i + _PINECONE_UPSERT_BATCH_SIZE])
    else:
        _get_chroma_collection().upsert(
            ids=[v["id"] for v in vectors],
            embeddings=[v["values"] for v in vectors],
            documents=[v["document"] for v in vectors],
            metadatas=[{"title": v["title"], "movie_poster": v["movie_poster"], "certification": v.get("certification", "")} for v in vectors],
        )
