import logging
import os
import json
import glob

from config import DATA_DIR, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)
from db import get_model, vector_upsert_batch
from tmdb import extract_certification


def embed_text(text: str) -> list[float]:
    return get_model().encode(text).tolist()


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    return get_model().encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()


def upsert_movie(movie_id: str) -> None:
    file_path = os.path.join(DATA_DIR, f"{movie_id}.json")
    with open(file_path, "r") as f:
        movie = json.load(f)
    if "richtext" not in movie:
        raise KeyError(f"Movie {movie.get('id', movie_id)} is missing richtext. Run richtext.py first.")
    embedding = embed_text(movie["richtext"])
    vector_upsert_batch([{
        "id": str(movie_id),
        "values": embedding,
        "title": movie["title"],
        "movie_poster": movie.get("poster_path") or "",
        "certification": extract_certification(movie),
        "document": movie["richtext"],
    }])


def load_all_richtexts() -> tuple[list[str], list[str], list[dict]]:
    ids, texts, metadatas = [], [], []
    for file_path in glob.glob(os.path.join(DATA_DIR, "*.json")):
        if "index" not in file_path:
            with open(file_path) as f:
                movie = json.load(f)
            if "richtext" not in movie:
                raise KeyError(f"Movie {movie.get('id', '?')} is missing richtext. Please run richtext.py before embeddings.py.")
            ids.append(str(movie["id"]))
            texts.append(movie["richtext"])
            metadatas.append({
                "title": movie["title"],
                "movie_poster": movie.get("poster_path") or "",
                "certification": extract_certification(movie),
            })
    return ids, texts, metadatas


def initialize_all_embeddings() -> int:
    ids, docs, metadatas = load_all_richtexts()
    embeddings = embed_texts(docs)
    vector_upsert_batch([
        {
            "id": id_,
            "values": emb,
            "title": meta["title"],
            "movie_poster": meta["movie_poster"],
            "certification": meta["certification"],
            "document": doc,
        }
        for id_, emb, doc, meta in zip(ids, embeddings, docs, metadatas)
    ])
    logger.info(f"Ingested {len(ids)} movies into vector store.")
    return len(ids)


if __name__ == "__main__":
    initialize_all_embeddings()
