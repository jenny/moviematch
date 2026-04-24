"""
tests/test_query_parser.py — Unit tests for query_parser.py.

Covers:
  - parse_query: decade normalization, year ranges, genre inclusion/exclusion,
    certifications, person patterns, slang triggers, "in the style of" dual-purpose,
    relative date hints
  - apply_hard_filters: year, genre, certification filtering with conservative
    empty-metadata handling
  - resolve_persons: TMDB calls mocked; success/failure paths
"""

import datetime
import pytest
from unittest.mock import patch, MagicMock

from query_parser import parse_query, apply_hard_filters, resolve_persons, ParsedQuery


# ---------------------------------------------------------------------------
# parse_query — year / decade
# ---------------------------------------------------------------------------

class TestParseQueryYears:
    def test_full_decade(self):
        p = parse_query("90s comedies")
        assert p.year_min == 1990
        assert p.year_max == 1999

    def test_abbreviated_80s(self):
        p = parse_query("80s action movies")
        assert p.year_min == 1980
        assert p.year_max == 1989

    def test_abbreviated_00s(self):
        # "00s" is ambiguous but should resolve to 2000–2009
        p = parse_query("00s films")
        assert p.year_min == 2000
        assert p.year_max == 2009

    def test_full_year_decade(self):
        p = parse_query("1970s cinema")
        assert p.year_min == 1970
        assert p.year_max == 1979

    def test_early_decade(self):
        p = parse_query("early 2000s thriller")
        assert p.year_min == 2000
        assert p.year_max == 2004

    def test_late_decade(self):
        p = parse_query("late 90s drama")
        assert p.year_min == 1996
        assert p.year_max == 1999

    def test_mid_decade(self):
        p = parse_query("mid 80s")
        assert p.year_min == 1983
        assert p.year_max == 1987

    def test_after_year(self):
        p = parse_query("after 1985")
        assert p.year_min == 1986
        assert p.year_max is None

    def test_before_year(self):
        p = parse_query("before 2000")
        assert p.year_max == 1999
        assert p.year_min is None

    def test_from_year(self):
        p = parse_query("movies from 2010")
        assert p.year_min == 2010
        assert p.year_max is None

    def test_in_exact_year(self):
        p = parse_query("in 1994")
        assert p.year_min == 1994
        assert p.year_max == 1994

    def test_decade_takes_precedence_over_from(self):
        # "from 2010s" — decade fires first, from should not override
        p = parse_query("from the 2010s")
        assert p.year_min == 2010
        assert p.year_max == 2019


# ---------------------------------------------------------------------------
# parse_query — genre
# ---------------------------------------------------------------------------

class TestParseQueryGenres:
    def test_required_genre(self):
        p = parse_query("action movies")
        assert "Action" in p.required_genres
        assert not p.excluded_genres

    def test_excluded_genre_no_prefix(self):
        p = parse_query("no documentaries")
        assert "Documentary" in p.excluded_genres
        assert "Documentary" not in p.required_genres

    def test_excluded_genre_without_prefix(self):
        p = parse_query("horror without comedy")
        assert "Horror" in p.required_genres
        assert "Comedy" in p.excluded_genres

    def test_sci_fi_alias(self):
        p = parse_query("sci-fi thriller")
        assert "Science Fiction" in p.required_genres

    def test_multiple_genres(self):
        p = parse_query("sci-fi with no horror")
        assert "Science Fiction" in p.required_genres
        assert "Horror" in p.excluded_genres

    def test_case_insensitive_genre(self):
        p = parse_query("COMEDY films")
        assert "Comedy" in p.required_genres


# ---------------------------------------------------------------------------
# parse_query — certifications
# ---------------------------------------------------------------------------

class TestParseQueryCertifications:
    def test_no_r_rated(self):
        p = parse_query("no R-rated movies")
        assert "R" in p.excluded_certifications
        assert not p.allowed_certifications

    def test_family_friendly(self):
        p = parse_query("family friendly films")
        assert p.allowed_certifications == ["G", "PG"]

    def test_pg13_only(self):
        p = parse_query("PG-13 only")
        assert "PG-13" in p.allowed_certifications

    def test_cert_allowlist_explicit(self):
        p = parse_query("PG movies")
        assert "PG" in p.allowed_certifications


# ---------------------------------------------------------------------------
# parse_query — person names
# ---------------------------------------------------------------------------

class TestParseQueryPersons:
    def test_directed_by(self):
        p = parse_query("directed by Bong Joon-ho")
        assert "Bong Joon-ho" in p.person_names
        assert p.person_department == "directing"

    def test_films_by(self):
        p = parse_query("films by Wes Anderson")
        assert "Wes Anderson" in p.person_names
        assert p.person_department == "directing"

    def test_starring(self):
        p = parse_query("starring Meryl Streep")
        assert "Meryl Streep" in p.person_names
        assert p.person_department == "cast"

    def test_something_with(self):
        p = parse_query("something with Tom Hanks")
        assert "Tom Hanks" in p.person_names
        assert p.person_department == "cast"

    def test_name_movies_auto(self):
        p = parse_query("Christopher Nolan movies")
        assert "Christopher Nolan" in p.person_names
        assert p.person_department == "auto"

    def test_slang_joint(self):
        p = parse_query("Spike Lee joint")
        assert "Spike Lee" in p.person_names

    def test_slang_flick(self):
        p = parse_query("a Coen Brothers flick")
        assert "Coen Brothers" in p.person_names

    def test_slang_pic(self):
        p = parse_query("a Wes Anderson pic")
        assert "Wes Anderson" in p.person_names

    def test_single_word_name_not_extracted(self):
        # Single-word stage names require 2+ tokens
        p = parse_query("a Spielberg film")
        assert p.person_names == []

    def test_no_person_for_generic_query(self):
        p = parse_query("great action movies")
        assert p.person_names == []

    def test_department_not_overridden_by_auto(self):
        # Explicit directing context should win over auto
        p = parse_query("directed by Christopher Nolan movies")
        assert p.person_department == "directing"


# ---------------------------------------------------------------------------
# parse_query — title references
# ---------------------------------------------------------------------------

class TestParseQueryTitleRefs:
    def test_something_like(self):
        p = parse_query("something like Inception")
        assert "Inception" in p.reference_titles
        # Single-word title should NOT be added to person_names
        assert "Inception" not in p.person_names

    def test_similar_to(self):
        p = parse_query("similar to The Godfather")
        assert "The Godfather" in p.reference_titles
        # "similar to" is title-only, even if title matches _NAME_TOKEN
        assert "The Godfather" not in p.person_names

    def test_in_the_style_of_director(self):
        # "in the style of X" where X is a director name → both ref_title and person
        p = parse_query("in the style of Wes Anderson")
        assert "Wes Anderson" in p.reference_titles
        assert "Wes Anderson" in p.person_names

    def test_in_the_style_of_single_word_title(self):
        # Single-word "title" won't match _NAME_TOKEN (requires 2+ words)
        p = parse_query("in the style of Inception")
        assert "Inception" in p.reference_titles
        assert "Inception" not in p.person_names

    def test_something_like_does_not_promote_to_person(self):
        # "something like The Godfather" should NOT add to person_names
        # (only "in the style of" triggers dual-purpose)
        p = parse_query("something like The Godfather")
        assert "The Godfather" in p.reference_titles
        assert "The Godfather" not in p.person_names


# ---------------------------------------------------------------------------
# parse_query — relative date hints
# ---------------------------------------------------------------------------

class TestParseQueryRelativeDates:
    def test_older_hint(self):
        p = parse_query("something like Inception but older")
        assert "older" in p.relative_date_hints

    def test_classic_hint(self):
        p = parse_query("classic noir films")
        assert "older" in p.relative_date_hints

    def test_newer_hint(self):
        p = parse_query("a more recent thriller")
        assert "newer" in p.relative_date_hints

    def test_recent_hint(self):
        p = parse_query("recent comedies")
        assert "newer" in p.relative_date_hints

    def test_last_n_years_hard_filter(self):
        p = parse_query("something good from the last 10 years")
        expected_min = datetime.date.today().year - 10
        assert p.year_min == expected_min

    def test_last_n_years_does_not_duplicate_hints(self):
        # "last N years" sets year_min directly; should not also appear as a hint
        p = parse_query("from the last 5 years")
        assert p.year_min == datetime.date.today().year - 5
        # Hints for newer/older are separate; last_N_years doesn't add a hint
        assert "newer" not in p.relative_date_hints

    def test_no_hints_for_generic_query(self):
        p = parse_query("great movies")
        assert p.relative_date_hints == []


# ---------------------------------------------------------------------------
# parse_query — has_filters / has_persons helpers
# ---------------------------------------------------------------------------

class TestParsedQueryHelpers:
    def test_has_filters_with_year(self):
        p = parse_query("90s films")
        assert p.has_filters()

    def test_has_filters_with_hints(self):
        p = parse_query("something older")
        assert p.has_filters()

    def test_has_filters_false_for_generic(self):
        p = parse_query("great movies")
        assert not p.has_filters()

    def test_has_persons_true(self):
        p = parse_query("Christopher Nolan movies")
        assert p.has_persons()

    def test_has_persons_false(self):
        p = parse_query("90s comedies")
        assert not p.has_persons()


# ---------------------------------------------------------------------------
# apply_hard_filters
# ---------------------------------------------------------------------------

def _candidate(title, year="", genres=None, certification=""):
    return {
        "title": title,
        "year": year,
        "genres": genres or [],
        "certification": certification,
        "document": "",
        "movie_poster": "",
        "overview": "",
        "director": "",
        "cast": [],
    }


class TestApplyHardFilters:
    def test_no_filters_returns_all(self):
        parsed = ParsedQuery()
        candidates = [_candidate("A"), _candidate("B")]
        assert apply_hard_filters(candidates, parsed) == candidates

    # --- Year ---
    def test_year_below_min_dropped(self):
        parsed = ParsedQuery(year_min=1990)
        candidates = [_candidate("Old", year="1985"), _candidate("New", year="1995")]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["New"]

    def test_year_above_max_dropped(self):
        parsed = ParsedQuery(year_max=1999)
        candidates = [_candidate("Old", year="1995"), _candidate("New", year="2005")]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["Old"]

    def test_exact_year_boundary_kept(self):
        parsed = ParsedQuery(year_min=1990, year_max=1990)
        candidates = [_candidate("Match", year="1990"), _candidate("Miss", year="1991")]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["Match"]

    def test_empty_year_kept_conservatively(self):
        parsed = ParsedQuery(year_min=1990, year_max=1999)
        candidates = [_candidate("No Year", year=""), _candidate("Too Old", year="1985")]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["No Year"]

    # --- Genre ---
    def test_excluded_genre_dropped(self):
        parsed = ParsedQuery(excluded_genres=["Documentary"])
        candidates = [
            _candidate("Doc", genres=["Documentary"]),
            _candidate("Action", genres=["Action"]),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["Action"]

    def test_required_genre_no_match_dropped(self):
        parsed = ParsedQuery(required_genres=["Comedy"])
        candidates = [
            _candidate("Action", genres=["Action"]),
            _candidate("Comedy", genres=["Comedy"]),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["Comedy"]

    def test_empty_genres_kept_conservatively(self):
        # Candidates with no genre metadata should not be dropped by required_genres filter
        parsed = ParsedQuery(required_genres=["Comedy"])
        candidates = [
            _candidate("No Genres", genres=[]),
            _candidate("Wrong Genre", genres=["Action"]),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["No Genres"]

    def test_excluded_genre_with_empty_genres_kept(self):
        # No genre metadata → keep conservatively even if excluded_genres is set
        parsed = ParsedQuery(excluded_genres=["Horror"])
        candidates = [_candidate("No Genres", genres=[])]
        result = apply_hard_filters(candidates, parsed)
        assert result == candidates

    # --- Certification ---
    def test_excluded_cert_dropped(self):
        parsed = ParsedQuery(excluded_certifications=["R"])
        candidates = [
            _candidate("R Movie", certification="R"),
            _candidate("PG Movie", certification="PG"),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["PG Movie"]

    def test_cert_not_in_allowlist_dropped(self):
        parsed = ParsedQuery(allowed_certifications=["G", "PG"])
        candidates = [
            _candidate("G Movie", certification="G"),
            _candidate("R Movie", certification="R"),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["G Movie"]

    def test_empty_cert_kept_conservatively(self):
        parsed = ParsedQuery(allowed_certifications=["PG"])
        candidates = [
            _candidate("No Cert", certification=""),
            _candidate("R Movie", certification="R"),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["No Cert"]

    def test_all_filters_combined(self):
        parsed = ParsedQuery(
            year_min=1990,
            year_max=1999,
            required_genres=["Comedy"],
            excluded_certifications=["R"],
        )
        candidates = [
            _candidate("Good", year="1995", genres=["Comedy"], certification="PG"),
            _candidate("Wrong Year", year="2005", genres=["Comedy"], certification="PG"),
            _candidate("Wrong Genre", year="1995", genres=["Action"], certification="PG"),
            _candidate("Wrong Cert", year="1995", genres=["Comedy"], certification="R"),
        ]
        result = apply_hard_filters(candidates, parsed)
        assert [c["title"] for c in result] == ["Good"]


# ---------------------------------------------------------------------------
# resolve_persons
# ---------------------------------------------------------------------------

class TestResolvePersons:
    def _make_parsed(self, names, department="auto"):
        p = ParsedQuery()
        p.person_names = list(names)
        p.person_department = department
        return p

    def _make_tmdb_person(self, name, person_id, known_for_dept="Directing"):
        return {"id": person_id, "name": name, "known_for_department": known_for_dept}

    def _make_movie(self, title, movie_id=1):
        return {"id": movie_id, "title": title, "vote_average": 7.5, "vote_count": 500}

    def test_successful_resolution_sets_filmography(self):
        parsed = self._make_parsed(["Christopher Nolan"], "directing")
        person = self._make_tmdb_person("Christopher Nolan", 525)
        movies = [self._make_movie("Inception", 27205), self._make_movie("Memento", 77)]

        # resolve_persons uses a deferred `from tmdb import ...` inside the function,
        # so we patch the tmdb module directly.
        with patch("tmdb.search_person", return_value=[person]), \
             patch("tmdb.get_filmography", return_value=movies):
            resolve_persons(parsed)

        assert len(parsed.person_filmographies) == 1
        pf = parsed.person_filmographies[0]
        assert pf["name"] == "Christopher Nolan"
        assert pf["person_id"] == 525
        assert pf["department"] == "directing"
        assert "Inception" in pf["titles"]
        assert "Memento" in pf["titles"]
        # Raw movie objects stored for background ingestion
        assert len(pf["movies"]) == 2
        assert parsed.is_person_focused is True

    def test_no_tmdb_results_skips_gracefully(self):
        parsed = self._make_parsed(["Unknown Person"])

        with patch("tmdb.search_person", return_value=[]):
            resolve_persons(parsed)

        assert parsed.person_filmographies == []
        assert parsed.is_person_focused is False

    def test_tmdb_exception_handled_gracefully(self):
        parsed = self._make_parsed(["Error Person"])

        with patch("tmdb.search_person", side_effect=Exception("TMDB down")):
            resolve_persons(parsed)  # should not raise

        assert parsed.person_filmographies == []
        assert parsed.is_person_focused is False

    def test_department_auto_defers_to_tmdb_known_for(self):
        parsed = self._make_parsed(["Meryl Streep"], "auto")
        person = self._make_tmdb_person("Meryl Streep", 123, known_for_dept="Acting")

        with patch("tmdb.search_person", return_value=[person]), \
             patch("tmdb.get_filmography", return_value=[]) as mock_get:
            resolve_persons(parsed)

        # "Acting" known_for_department → should resolve to "cast"
        mock_get.assert_called_once_with(123, "cast")

    def test_explicit_directing_department_takes_precedence(self):
        parsed = self._make_parsed(["Someone"], "directing")
        person = self._make_tmdb_person("Someone", 1, known_for_dept="Acting")

        with patch("tmdb.search_person", return_value=[person]), \
             patch("tmdb.get_filmography", return_value=[]) as mock_get:
            resolve_persons(parsed)

        # Explicit query context should override TMDB known_for_department
        mock_get.assert_called_once_with(1, "directing")

    def test_multiple_persons_all_resolved(self):
        parsed = self._make_parsed(["Alice Director", "Bob Actor"], "auto")
        persons = [
            self._make_tmdb_person("Alice Director", 1, "Directing"),
            self._make_tmdb_person("Bob Actor", 2, "Acting"),
        ]
        movies = [self._make_movie("Movie A")]

        call_count = 0
        def fake_search(name):
            nonlocal call_count
            result = [persons[call_count]]
            call_count += 1
            return result

        with patch("tmdb.search_person", side_effect=fake_search), \
             patch("tmdb.get_filmography", return_value=movies):
            resolve_persons(parsed)

        assert len(parsed.person_filmographies) == 2
        assert parsed.is_person_focused is True
