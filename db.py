import chromadb
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, CHROMA_PATH, COLLECTION_NAME

_model = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to load Sentence Transformers model: {e}")
    return _model


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        try:
            chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            _collection = chroma.get_collection(name=COLLECTION_NAME)
        except ValueError:
            raise RuntimeError("ChromaDB collection 'movies' not found. Please run embeddings.py first.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to ChromaDB: {e}")
    return _collection


def get_or_create_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        try:
            chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            _collection = chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to ChromaDB: {e}")
    return _collection
