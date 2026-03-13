from tmdb import ingest_index, ingest_movie
from richtext import compile_all_richtexts
from embeddings import initialize_all_embeddings


def initialize_all(n: int) -> dict:
    print(f"Starting full pipeline initialization for {n} movies...")

    movie_ids = ingest_index(n)

    for movie_id in movie_ids:
        ingest_movie(movie_id)

    compile_all_richtexts()

    embedded_count = initialize_all_embeddings()

    return {
        "movie_count": n,
        "indexed_count": len(movie_ids),
        "embedded_count": embedded_count
    }


if __name__ == "__main__":
    n = int(input("How many movies to initialize? "))
    result = initialize_all(n)
    print(f"\nPipeline complete: {result}")
