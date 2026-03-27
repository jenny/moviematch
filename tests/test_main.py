import pytest
from main import _parse_document

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
