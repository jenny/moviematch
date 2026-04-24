import pytest
from unittest.mock import patch, MagicMock, call
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

    def test_extracts_genres(self):
        result = _parse_document(FULL_DOC)
        assert result["genres"] == ["Crime", "Drama"]

    def test_missing_genres_returns_empty_list(self):
        doc = FULL_DOC.replace("Genres: Crime, Drama\n\n", "")
        result = _parse_document(doc)
        assert result["genres"] == []

    def test_empty_document_returns_all_defaults(self):
        result = _parse_document("")
        assert result == {"year": "", "overview": "", "genres": [], "director": "", "cast": []}

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


# ---------------------------------------------------------------------------
# search_stream: pre-parse integration
# ---------------------------------------------------------------------------

def _make_candidate(title, year="", genres=None, certification=""):
    return {
        "title": title,
        "year": year,
        "genres": genres or [],
        "certification": certification,
        "movie_poster": "",
        "document": f"Plot: A movie.\n\nTitle: {title}",
        "overview": "A movie.",
        "director": "",
        "cast": [],
    }


class TestSearchStreamPreParse:
    """Integration tests for the pre-parse + filter block in search_stream."""

    def _mock_fetch_candidates(self, candidates):
        """Return a patch target that yields the given candidates."""
        return patch("main._fetch_candidates", return_value=(candidates, 10, 5))

    def _mock_rerank_stream(self, results):
        """Yield result dicts then a __usage sentinel."""
        def _gen(*args, **kwargs):
            yield from results
            yield {"__usage": {"input_tokens": 10, "output_tokens": 5, "rounds": 1, "tools_called": []}}
        return patch("main.rerank_stream", side_effect=_gen)

    def test_year_filter_removes_out_of_range_candidates(self):
        """Candidates outside the parsed year range should not reach rerank_stream."""
        candidates = [
            _make_candidate("Good", year="1995"),
            _make_candidate("Too Old", year="1985"),
            _make_candidate("Too New", year="2005"),
        ]
        with self._mock_fetch_candidates(candidates), \
             patch("main.rerank_stream") as mock_rerank, \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"):
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery(year_min=1990, year_max=1999)
            mock_parse.return_value = mock_parsed

            mock_rerank.return_value = iter([
                {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}}
            ])

            from main import search_stream
            list(search_stream("90s movies"))

        # Only the in-range candidate should have been passed to rerank_stream
        call_candidates = mock_rerank.call_args[0][1]
        assert len(call_candidates) == 1
        assert call_candidates[0]["title"] == "Good"

    def test_person_future_timeout_falls_back_to_claude_tools(self):
        """A timed-out person future should log a warning and rerank still fires."""
        from concurrent.futures import Future, TimeoutError as FuturesTimeoutError

        future = MagicMock(spec=Future)
        future.result.side_effect = FuturesTimeoutError()

        candidates = [_make_candidate("Memento", year="2000")]

        with self._mock_fetch_candidates(candidates), \
             patch("main.rerank_stream") as mock_rerank, \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor") as mock_executor:
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery()
            mock_parsed.person_names = ["Christopher Nolan"]
            mock_parsed.person_filmographies = []
            mock_parse.return_value = mock_parsed
            mock_executor.submit.return_value = future

            mock_rerank.return_value = iter([
                {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}}
            ])

            from main import search_stream
            list(search_stream("Christopher Nolan movies"))

        # rerank_stream should still be called despite the timeout
        mock_rerank.assert_called_once()

    def test_successful_person_future_triggers_background_ingestion(self):
        """After a successful person future, background ingestion threads are started."""
        raw_movies = [{"id": 27205, "title": "Inception", "vote_average": 8.8, "vote_count": 2000}]
        candidates = [_make_candidate("Inception", year="2010")]

        with self._mock_fetch_candidates(candidates), \
             patch("main.rerank_stream") as mock_rerank, \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor") as mock_executor, \
             patch("main.threading.Thread") as mock_thread:
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery()
            mock_parsed.person_names = ["Christopher Nolan"]
            mock_parsed.person_filmographies = [
                {"name": "Christopher Nolan", "person_id": 525,
                 "department": "directing", "titles": ["Inception"],
                 "movies": raw_movies}
            ]
            mock_parsed.is_person_focused = True
            mock_parse.return_value = mock_parsed

            future = MagicMock()
            future.result.return_value = None  # success
            mock_executor.submit.return_value = future

            mock_rerank.return_value = iter([
                {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}}
            ])

            from main import search_stream
            list(search_stream("Christopher Nolan movies"))

        # A background thread should have been started for ingestion
        mock_thread.assert_called_once()
        thread_kwargs = mock_thread.call_args
        assert thread_kwargs[1].get("daemon") is True

    def test_metadata_lookup_is_case_insensitive(self):
        """If Claude returns a title with different casing than the vector candidate,
        metadata (year, poster, etc.) should still be attached correctly."""
        candidates = [_make_candidate("Inception", year="2010")]
        candidates[0]["movie_poster"] = "/poster.jpg"
        candidates[0]["overview"] = "A dream heist."

        with self._mock_fetch_candidates(candidates), \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"):
            from query_parser import ParsedQuery
            mock_parse.return_value = ParsedQuery()

            # Claude returns lowercase title — different casing from the candidate
            with patch("main.rerank_stream") as mock_rerank:
                mock_rerank.return_value = iter([
                    {"title": "inception", "explanation": "A classic."},
                    {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}},
                ])

                from main import search_stream
                results = [r for r in search_stream("dream heist") if "__meta" not in r]

        assert len(results) == 1
        assert results[0]["year"] == "2010"
        assert results[0]["movie_poster"] == "/poster.jpg"
        assert results[0]["overview"] == "A dream heist."

    def test_filmography_only_title_gets_poster_and_year(self):
        """A title from the filmography that is NOT among the vector candidates
        should still receive poster_path and release year from the TMDB movie dict."""
        candidates = [_make_candidate("Memento", year="2000")]

        filmography_movies = [
            {
                "id": 999,
                "title": "Following",
                "vote_average": 7.5,
                "vote_count": 500,
                "release_date": "1998-09-11",
                "poster_path": "/following.jpg",
            }
        ]

        with self._mock_fetch_candidates(candidates), \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"):
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery()
            mock_parsed.person_names = ["Christopher Nolan"]
            mock_parsed.person_filmographies = [
                {"name": "Christopher Nolan", "person_id": 525,
                 "department": "directing", "titles": ["Memento", "Following"],
                 "movies": filmography_movies}
            ]
            mock_parse.return_value = mock_parsed

            with patch("main.rerank_stream") as mock_rerank:
                # Claude recommends the filmography-only title "Following"
                mock_rerank.return_value = iter([
                    {"title": "Following", "explanation": "Nolan's debut."},
                    {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}},
                ])

                from main import search_stream
                results = [r for r in search_stream("Nolan films") if "__meta" not in r]

        assert len(results) == 1
        assert results[0]["title"] == "Following"
        assert results[0]["movie_poster"] == "/following.jpg"
        assert results[0]["year"] == "1998"

    def test_filmography_seed_does_not_overwrite_candidate_metadata(self):
        """If a title appears in both candidates and filmography, the richer
        vector DB metadata (from the candidate) should take precedence."""
        candidates = [_make_candidate("Memento", year="2000")]
        candidates[0]["movie_poster"] = "/memento_vector.jpg"
        candidates[0]["overview"] = "A man with no short-term memory."

        filmography_movies = [
            {
                "id": 1,
                "title": "Memento",          # same title as candidate
                "vote_average": 8.4,
                "vote_count": 1500,
                "release_date": "1999-09-05",
                "poster_path": "/memento_tmdb.jpg",  # different poster — should NOT win
            }
        ]

        with self._mock_fetch_candidates(candidates), \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"):
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery()
            mock_parsed.person_filmographies = [
                {"name": "Christopher Nolan", "person_id": 525,
                 "department": "directing", "titles": ["Memento"],
                 "movies": filmography_movies}
            ]
            mock_parse.return_value = mock_parsed

            with patch("main.rerank_stream") as mock_rerank:
                mock_rerank.return_value = iter([
                    {"title": "Memento", "explanation": "Memory loss thriller."},
                    {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}},
                ])

                from main import search_stream
                results = [r for r in search_stream("Nolan films") if "__meta" not in r]

        assert len(results) == 1
        # Vector DB metadata wins over filmography TMDB metadata
        assert results[0]["movie_poster"] == "/memento_vector.jpg"
        assert results[0]["overview"] == "A man with no short-term memory."
