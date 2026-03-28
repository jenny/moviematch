import pytest
from unittest.mock import patch, MagicMock
from main import _parse_document, _fetch_candidates

# Representative richtext document matching the format produced by build_richtext()
FULL_DOC = (
    "Plot: The aging patriarch of an organized crime dynasty transfers control to his son.\n\n"
    "Themes and Keywords: mafia, organized crime, family\n\n"
    "Genres: Crime, Drama\n\n"
    "Director: Francis Ford Coppola\n\n"
    "Top Cast: Marlon Brando, Al Pacino, James Caan, Robert Duvall, Diane Keaton\n\n"
    "Title: The Godfather (1972)\n\n"
)


class TestParseDocument:
    def test_extracts_overview(self):
        result = _parse_document(FULL_DOC)
        assert result["overview"] == (
            "The aging patriarch of an organized crime dynasty transfers control to his son."
        )

    def test_extracts_director(self):
        result = _parse_document(FULL_DOC)
        assert result["director"] == "Francis Ford Coppola"

    def test_extracts_cast_as_list(self):
        result = _parse_document(FULL_DOC)
        assert result["cast"] == [
            "Marlon Brando", "Al Pacino", "James Caan", "Robert Duvall", "Diane Keaton"
        ]

    def test_extracts_year(self):
        result = _parse_document(FULL_DOC)
        assert result["year"] == "1972"

    def test_missing_director_returns_empty_string(self):
        doc = FULL_DOC.replace("Director: Francis Ford Coppola\n\n", "")
        result = _parse_document(doc)
        assert result["director"] == ""

    def test_missing_cast_returns_empty_list(self):
        doc = FULL_DOC.replace(
            "Top Cast: Marlon Brando, Al Pacino, James Caan, Robert Duvall, Diane Keaton\n\n", ""
        )
        result = _parse_document(doc)
        assert result["cast"] == []

    def test_missing_year_returns_empty_string(self):
        doc = FULL_DOC.replace("(1972)", "")
        result = _parse_document(doc)
        assert result["year"] == ""

    def test_empty_document_returns_all_defaults(self):
        result = _parse_document("")
        assert result == {"year": "", "overview": "", "director": "", "cast": []}

    def test_cast_strips_surrounding_whitespace(self):
        doc = FULL_DOC.replace(
            "Top Cast: Marlon Brando, Al Pacino, James Caan, Robert Duvall, Diane Keaton",
            "Top Cast:  Marlon Brando ,  Al Pacino ",
        )
        result = _parse_document(doc)
        assert result["cast"] == ["Marlon Brando", "Al Pacino"]

    def test_single_cast_member(self):
        doc = FULL_DOC.replace(
            "Top Cast: Marlon Brando, Al Pacino, James Caan, Robert Duvall, Diane Keaton",
            "Top Cast: Marlon Brando",
        )
        result = _parse_document(doc)
        assert result["cast"] == ["Marlon Brando"]

    def test_empty_cast_line_returns_empty_list(self):
        doc = FULL_DOC.replace(
            "Top Cast: Marlon Brando, Al Pacino, James Caan, Robert Duvall, Diane Keaton",
            "Top Cast: ",
        )
        result = _parse_document(doc)
        assert result["cast"] == []


def _make_vector_match(title: str, doc: str = "") -> dict:
    return {"title": title, "movie_poster": "", "document": doc}


def _mock_model():
    model = MagicMock()
    model.encode.return_value.tolist.return_value = [0.0] * 384
    return model


class TestFetchCandidatesDeduplicate:
    def test_duplicate_vector_titles_returns_first_only(self):
        matches = [
            _make_vector_match("Inception", "Plot: First entry.\n\nDirector: Nolan\n\nTitle: Inception (2010)\n\n"),
            _make_vector_match("Inception", "Plot: Duplicate entry.\n\nDirector: Nolan\n\nTitle: Inception (2010)\n\n"),
        ]
        with patch("main.vector_count", return_value=2), \
             patch("main.vector_query", return_value=matches), \
             patch("main.get_model", return_value=_mock_model()):
            candidates, _, _ = _fetch_candidates("sci-fi dream heist")

        assert len(candidates) == 1
        assert candidates[0]["title"] == "Inception"

    def test_three_results_with_one_duplicate_title(self):
        matches = [
            _make_vector_match("A"),
            _make_vector_match("B"),
            _make_vector_match("A"),  # duplicate
        ]
        with patch("main.vector_count", return_value=3), \
             patch("main.vector_query", return_value=matches), \
             patch("main.get_model", return_value=_mock_model()):
            candidates, _, _ = _fetch_candidates("some query")

        titles = [c["title"] for c in candidates]
        assert titles == ["A", "B"]

    def test_no_duplicates_passes_through_unchanged(self):
        matches = [_make_vector_match("A"), _make_vector_match("B"), _make_vector_match("C")]
        with patch("main.vector_count", return_value=3), \
             patch("main.vector_query", return_value=matches), \
             patch("main.get_model", return_value=_mock_model()):
            candidates, _, _ = _fetch_candidates("some query")

        assert [c["title"] for c in candidates] == ["A", "B", "C"]
