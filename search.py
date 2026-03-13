import chromadb
from sentence_transformers import SentenceTransformer

from prompt_claude import rerank

# load model (downloads automatically)
try:
    MODEL = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    raise RuntimeError(f"Failed to load Sentence Transformers model: {e}")

try:
    chroma = chromadb.PersistentClient(path='./embeddings/chroma_db')
    collection = chroma.get_collection(name='movies')
except ValueError:
    raise RuntimeError("ChromaDB collection 'movies' not found. Please run init_embeddings.py first.")
except Exception as e:
    raise RuntimeError(f"Failed to connect to ChromaDB: {e}")

def search(query):
    if collection.count() == 0:
        print("ChromaDB collection is empty. Please run init_embeddings.py first.")
        return

    try:
        q_embeddings = MODEL.encode(query)
        n_results = min(10, collection.count())
        q_results = collection.query(query_embeddings=q_embeddings, n_results=n_results)
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return

    q_results_metadatas = q_results["metadatas"][0]
    q_results_documents = q_results["documents"][0]

    candidates = []
    for i in range(len(q_results_metadatas)):
        title = q_results_metadatas[i].get("title")
        if not title:
            print(f"Warning: missing title for result {i}, skipping.")
            continue
        candidates.append({"title": title, "document": q_results_documents[i][:300]})

    if not candidates:
        print("No valid candidates found. Try re-running init_embeddings.py.")
        return

    print(f"\nSearching {len(candidates)} candidates for \"{query}\"...\n")

    try:
        reranked = rerank(query, candidates)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return

    if not reranked:
        print("No relevant matches found. Please try a different query.")
        return

    print(f"Found {len(reranked)} relevant matches:\n")
    for i, result in enumerate(reranked, start=1):
        print(f"{i}. {result['title']}")
        print(f"   {result['explanation']}\n")


if __name__ == "__main__":
    query = input("Enter a search query: ")
    search(query)
