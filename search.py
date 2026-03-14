from config import SEARCH_CANDIDATES, SEARCH_DOC_TRUNCATE
from db import get_model, get_collection
from claude import rerank


def search(query: str) -> list[dict]:
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError("ChromaDB collection is empty. Please run embeddings.py first.")

    try:
        q_embeddings = get_model().encode(query)
        n_results = min(SEARCH_CANDIDATES, collection.count())
        q_results = collection.query(query_embeddings=q_embeddings, n_results=n_results)
    except Exception as e:
        raise RuntimeError(f"Error querying ChromaDB: {e}")

    metadatas = q_results["metadatas"][0]
    documents = q_results["documents"][0]

    candidates = []
    for i in range(len(metadatas)):
        title = metadatas[i].get("title")
        if not title:
            print(f"Warning: missing title for result {i}, skipping.")
            continue
        candidates.append({"title": title, "movie_poster": metadatas[i].get("movie_poster") or "", "document": documents[i][:SEARCH_DOC_TRUNCATE]})

    if not candidates:
        return []

    poster_by_title = {c["title"]: c["movie_poster"] for c in candidates}
    reranked = rerank(query, candidates)
    for result in reranked:
        result["movie_poster"] = poster_by_title.get(result["title"], "")
    return reranked


if __name__ == "__main__":
    query = input("Enter a search query: ")
    try:
        results = search(query)
    except RuntimeError as e:
        print(f"Error: {e}")
    else:
        if not results:
            print("No relevant matches found. Please try a different query.")
        else:
            print(f"\nFound {len(results)} relevant matches:\n")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"   {result['explanation']}\n")
