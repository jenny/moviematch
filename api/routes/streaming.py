from fastapi import APIRouter

from tmdb import search_movie_by_title, fetch_watch_providers

router = APIRouter()


@router.get("/streaming")
def streaming_providers(title: str, year: str = ""):
    """Return US streaming providers for a movie. Looks up TMDB ID by title + optional year."""
    movie_id = search_movie_by_title(title, year)
    if movie_id is None:
        return {"providers": []}
    providers = fetch_watch_providers(movie_id)
    return {"providers": providers}
