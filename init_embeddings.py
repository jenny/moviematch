import json
import glob

import chromadb
from sentence_transformers import SentenceTransformer

# load model (downloads automatically)
try:
    MODEL = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    raise RuntimeError(f"Failed to load Sentence Transformers model: {e}")

# load chromadb
try:
    chroma = chromadb.PersistentClient(path='./embeddings/chroma_db')
    collection = chroma.get_or_create_collection(
        name='movies',
        embedding_function=None,
        metadata={'hnsw:space': 'cosine'}  # cosine similarity
    )
except Exception as e:
    raise RuntimeError(f"Failed to connect to ChromaDB: {e}")

# embed a single string
def embed_text_single(text: str) -> list[float]:
    vector = MODEL.encode(text)
    return vector.tolist()

# upsert one movie
def upsert_movie_single(id: str):
    file = "data/" + id + ".json"
    with open(file, "r") as f:
        movie = json.load(f)
        richtext = movie["richtext"]
        embeddings = MODEL.encode(richtext).tolist()
        collection.upsert(ids=[id],
                        embeddings=[embeddings],
                        documents=[richtext],
                        metadatas=[{"title": movie["title"]}])

    #print(collection.count())
    #print(collection.peek())

# load all richtext strings into memory for batch embedding
def load_richtexts_batch() -> tuple[list[str], list[str], list[dict]]:
    ids, texts, metadatas = [], [], []
    for file in glob.glob(f"./data/*.json"):
        if("index" not in file):
            with open(file) as f:
                movie = json.load(f)
            if "richtext" not in movie:
                raise KeyError(f"Movie {movie.get('id', '?')} is missing richtext. Please run init_richtext.py before init_embeddings.py.")
            ids.append(str(movie["id"]))
            texts.append(movie["richtext"])
            metadatas.append({"title": movie["title"]})
    return ids, texts, metadatas

# batch embeddings
def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vectors = MODEL.encode(texts, batch_size=batch_size,
                           show_progress_bar=True)
    return vectors.tolist()

def initialize_embeddings_batch():
    ids, docs, metadatas = load_richtexts_batch()

    # Embed in batches of 64
    embeddings = get_embeddings_batch(docs, batch_size=64)

    # Upsert into Chroma (safe to re-run)
    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=docs, metadatas=metadatas)
    print(f"Ingested {len(ids)} movies into Chroma.")
    #print(f"\n\n{collection.count()}")
    #print(f"\n\n{collection.peek()}")

if __name__ == "__main__":     
    initialize_embeddings_batch()






