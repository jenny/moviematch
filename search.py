import chromadb
from sentence_transformers import SentenceTransformer

# load model (downloads automatically)
MODEL = SentenceTransformer('all-mpnet-base-v2')

chroma = chromadb.PersistentClient(path='./embeddings/chroma_db')
collection = chroma.get_collection(
    name='movies'
)

def search(query):
    q_embeddings = MODEL.encode(query)
    all_results = [] # {"title": distance}

    '''
    # todo: include "contains" results
    contains_results = collection.get(where_document={"$contains": query})["documents"]
    for cr in contains_results:
        title = cr.split('\n', maxsplit=1)[0]
        all_results.append({title: 1})
    '''

    q_results = collection.query(query_embeddings=q_embeddings)

    q_results_metadatas = q_results["metadatas"][0]
    q_results_distances = q_results["distances"][0]
    print(f"\n\nFound {len(q_results_metadatas)} matches for query \"{query}\"\n")

    for i in range(len(q_results_metadatas)):
        title = q_results_metadatas[i]["title"]
        distance = q_results_distances[i]
        all_results.append({"title": title, "distance": distance})

    for i in all_results:
        t = i["title"]
        s = i["distance"]
        print(f"{t}, distance: {s}")
    
    
if __name__ == "__main__":
    query = input("Enter a search query: ")
    search(query)




