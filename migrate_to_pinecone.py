"""One-off script: migrate all vectors from local ChromaDB → Pinecone.

Usage:
    VECTOR_DB=pinecone python migrate_to_pinecone.py

Reads every record from the local Chroma collection and upserts them into
Pinecone in batches using the existing vector_upsert_batch abstraction.
Requires PINECONE_API_KEY (and optionally PINECONE_INDEX_NAME) to be set,
either in the environment or in a .env file.
"""

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME, VECTOR_DB
from db import vector_upsert_batch, vector_count

BATCH_SIZE = 500


def main():
    if VECTOR_DB != "pinecone":
        raise SystemExit("Set VECTOR_DB=pinecone before running this script.")

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION_NAME)

    total = collection.count()
    if total == 0:
        raise SystemExit("Local Chroma collection is empty — nothing to migrate.")

    print(f"Migrating {total} vectors from Chroma → Pinecone...")

    offset = 0
    migrated = 0
    while offset < total:
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        ids        = batch["ids"]
        embeddings = batch["embeddings"]
        documents  = batch["documents"]
        metadatas  = batch["metadatas"]

        vectors = [
            {
                "id":          ids[i],
                "values":      embeddings[i],
                "title":       metadatas[i].get("title", ""),
                "movie_poster": metadatas[i].get("movie_poster", ""),
                "document":    documents[i],
            }
            for i in range(len(ids))
        ]

        vector_upsert_batch(vectors)
        migrated += len(vectors)
        offset   += BATCH_SIZE
        print(f"  {migrated}/{total}")

    pinecone_count = vector_count()
    print(f"Done. Pinecone now has {pinecone_count} vectors.")


if __name__ == "__main__":
    main()
