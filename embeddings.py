import os
import json
import glob

from config import DATA_DIR, EMBEDDING_BATCH_SIZE
from db import get_model, get_or_create_collection


def embed_text(text: str) -> list[float]:
    return get_model().encode(text).tolist()


def embed_texts(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    return get_model().encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()


def upsert_movie(movie_id: str) -> None:
    file_path = os.path.join(DATA_DIR, f"{movie_id}.json")
    with open(file_path, "r") as f:
        movie = json.load(f)
    embedding = embed_text(movie["richtext"])
    get_or_create_collection().upsert(
        ids=[movie_id],
        embeddings=[embedding],
        documents=[movie["richtext"]],
        metadatas=[{"title": movie["title"], "movie_poster": movie.get("poster_path") or ""}]
    )


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
            metadatas.append({"title": movie["title"], "movie_poster": movie.get("poster_path") or ""})
    return ids, texts, metadatas


def initialize_all_embeddings() -> int:
    ids, docs, metadatas = load_all_richtexts()
    embeddings = embed_texts(docs)
    get_or_create_collection().upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
    print(f"Ingested {len(ids)} movies into ChromaDB.")
    return len(ids)


if __name__ == "__main__":
    initialize_all_embeddings()
