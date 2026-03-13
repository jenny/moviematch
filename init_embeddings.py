import json
import glob

import chromadb
from sentence_transformers import SentenceTransformer

# load model (downloads automatically)
MODEL = SentenceTransformer('all-mpnet-base-v2')

# load chromadb
chroma = chromadb.PersistentClient(path='./embeddings/chroma_db')
collection = chroma.get_or_create_collection(
    name='movies',
    embedding_function=None,
    metadata={'hnsw:space': 'cosine'}  # cosine similarity
)

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
        embeddings = MODEL.encode(richtext)
        collection.upsert(ids=[id],
                        embeddings=[embeddings],
                        documents=[richtext])

    #print(collection.count())
    #print(collection.peek())

# load all richtext strings into memory for batch embedding
def load_richtexts_batch() -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for file in glob.glob(f"./data/*.json"):
        if("index" not in file):
            with open(file) as f:
                movie = json.load(f)
            ids.append(str(movie["id"]))
            texts.append(movie["richtext"])
    return ids, texts # returns tuple

# batch embeddings
def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vectors = MODEL.encode(texts, batch_size=batch_size,
                           show_progress_bar=True)
    return vectors.tolist()

def initialize_embeddings_batch():
    richtexts = load_richtexts_batch()
    ids = richtexts[0]
    docs = richtexts[1]

    # Embed in batches of 64
    embeddings = get_embeddings_batch(docs, batch_size=64)

    # Upsert into Chroma (safe to re-run)
    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=docs)
    print(f"Ingested {len(ids)} movies into Chroma.")
    #print(f"\n\n{collection.count()}")
    #print(f"\n\n{collection.peek()}")

if __name__ == "__main__":     
    initialize_embeddings_batch()






