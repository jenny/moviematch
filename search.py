import chromadb
from sentence_transformers import SentenceTransformer

from prompt_claude import rerank

# load model (downloads automatically)
MODEL = SentenceTransformer('all-mpnet-base-v2')

chroma = chromadb.PersistentClient(path='./embeddings/chroma_db')
collection = chroma.get_collection(
    name='movies'
)

def search(query):
    q_embeddings = MODEL.encode(query)

    q_results = collection.query(query_embeddings=q_embeddings, n_results=20)

    q_results_metadatas = q_results["metadatas"][0]
    q_results_documents = q_results["documents"][0]

    candidates = [
        {"title": q_results_metadatas[i]["title"], "document": q_results_documents[i]}
        for i in range(len(q_results_metadatas))
    ]

    print(f"\nSearching {len(candidates)} candidates for \"{query}\"...\n")

    reranked = rerank(query, candidates)

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
