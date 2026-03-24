import pytest
from richtext import build_richtext
from config import RICHTEXT_CAST_LIMIT


FULL_MOVIE = {
    "title": "The Godfather",
    "overview": "The aging patriarch of an organized crime dynasty transfers control to his son.",
    "release_date": "1972-03-24",
    "keywords": {"keywords": [{"name": "mafia"}, {"name": "crime family"}]},
    "genres": [{"name": "Crime"}, {"name": "Drama"}],
    "credits": {
        "crew": [{"name": "Francis Ford Coppola", "job": "Director"}],
        "cast": [
            {"name": "Marlon Brando"},
            {"name": "Al Pacino"},
            {"name": "James Caan"},
            {"name": "Robert Duvall"},
            {"name": "Diane Keaton"},
            {"name": "Richard Castellano"},  # beyond RICHTEXT_CAST_LIMIT=5, should be excluded
        ],
    },
}


def test_full_movie_contains_all_sections():
    result = build_richtext(FULL_MOVIE)
    assert "Plot: The aging patriarch" in result
    assert "Themes and Keywords: mafia, crime family" in result
    assert "Genres: Crime, Drama" in result
    assert "Director: Francis Ford Coppola" in result
    assert "Top Cast: Marlon Brando" in result
    assert "Title: The Godfather (1972)" in result


def test_cast_capped_at_richtext_limit():
    movie = {
        **FULL_MOVIE,
        "credits": {
            "crew": [],
            "cast": [{"name": f"Actor {i}"} for i in range(RICHTEXT_CAST_LIMIT + 5)],
        },
    }
    result = build_richtext(movie)
    cast_line = next(line for line in result.splitlines() if line.startswith("Top Cast:"))
    names = [n.strip() for n in cast_line.removeprefix("Top Cast: ").split(",") if n.strip()]
    assert len(names) == RICHTEXT_CAST_LIMIT


def test_cast_beyond_limit_excluded():
    result = build_richtext(FULL_MOVIE)
    assert "Richard Castellano" not in result


def test_missing_overview_defaults_to_empty():
    movie = {**FULL_MOVIE, "overview": ""}
    result = build_richtext(movie)
    assert result.startswith("Plot: \n")


def test_missing_keywords_key():
    movie = {k: v for k, v in FULL_MOVIE.items() if k != "keywords"}
    result = build_richtext(movie)
    assert "Themes and Keywords: \n" in result


def test_empty_keywords_list():
    movie = {**FULL_MOVIE, "keywords": {"keywords": []}}
    result = build_richtext(movie)
    assert "Themes and Keywords: \n" in result


def test_missing_genres_key():
    movie = {k: v for k, v in FULL_MOVIE.items() if k != "genres"}
    result = build_richtext(movie)
    assert "Genres: \n" in result


def test_empty_genres_list():
    movie = {**FULL_MOVIE, "genres": []}
    result = build_richtext(movie)
    assert "Genres: \n" in result


def test_no_director_in_crew():
    movie = {
        **FULL_MOVIE,
        "credits": {
            "crew": [{"name": "John Smith", "job": "Producer"}],
            "cast": [],
        },
    }
    result = build_richtext(movie)
    assert "Director:" not in result


def test_multiple_directors_both_appear():
    movie = {
        **FULL_MOVIE,
        "credits": {
            "crew": [
                {"name": "Director One", "job": "Director"},
                {"name": "Director Two", "job": "Director"},
            ],
            "cast": [],
        },
    }
    result = build_richtext(movie)
    assert "Director: Director One" in result
    assert "Director: Director Two" in result


def test_missing_credits_key():
    movie = {k: v for k, v in FULL_MOVIE.items() if k != "credits"}
    result = build_richtext(movie)
    assert "Director:" not in result
    assert "Top Cast: \n" in result


def test_missing_release_date_shows_unknown():
    movie = {k: v for k, v in FULL_MOVIE.items() if k != "release_date"}
    result = build_richtext(movie)
    assert "(Unknown)" in result


def test_empty_release_date_shows_unknown():
    movie = {**FULL_MOVIE, "release_date": ""}
    result = build_richtext(movie)
    assert "(Unknown)" in result


def test_release_year_extracted_from_date():
    movie = {**FULL_MOVIE, "title": "Inception", "release_date": "2010-07-16"}
    result = build_richtext(movie)
    assert "Title: Inception (2010)" in result
