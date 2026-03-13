from tmdb import ingest_index, ingest_movie
from richtext import compile_all_richtexts
from embeddings import initialize_all_embeddings


def initialize_all(n: int) -> dict:
    print(f"Starting full pipeline initialization for {n} movies...")

    movie_ids = ingest_index(n)

    failed_ids = []
    for movie_id in movie_ids:
        try:
            ingest_movie(movie_id)
        except Exception as e:
            print(f"Failed to ingest movie {movie_id}: {e}")
            failed_ids.append(movie_id)

    if failed_ids:
        print(f"Warning: {len(failed_ids)} movies failed to ingest: {failed_ids}")

    compile_all_richtexts()

    embedded_count = initialize_all_embeddings()

    return {
        "movie_count": len(movie_ids) - len(failed_ids),
        "indexed_count": len(movie_ids),
        "embedded_count": embedded_count,
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids
    }


if __name__ == "__main__":
    n = int(input("How many movies to initialize? "))
    result = initialize_all(n)
    print(f"\nPipeline complete: {result}")
