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

    @pytest.fixture(autouse=True)
    def _no_real_cert_fetch(self):
        """Runtime cert backfill for not-yet-ingested filmography films must never hit
        the real TMDB API in tests. Default to empty; individual tests override as needed."""
        with patch("main.fetch_certification", return_value="") as m:
            yield m

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
             patch("main.threading.Thread") as mock_thread, \
             patch("main.vector_fetch_by_ids", return_value=[]):
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
        """A filmography title NOT in the vector DB (vector_fetch_by_ids returns empty)
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
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]):
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

    def test_filmography_title_in_vector_db_gets_full_metadata(self):
        """A filmography title not in the top-N candidates but present in the vector DB
        should receive full rich metadata (overview, genres, director, cast) via the
        vector_fetch_by_ids batch lookup, not just the sparse poster+year from TMDB."""
        candidates = [_make_candidate("Memento", year="2000")]

        filmography_movies = [
            {
                "id": 999,
                "title": "Following",
                "vote_average": 7.5,
                "vote_count": 500,
                "release_date": "1998-09-11",
                "poster_path": "/following_tmdb.jpg",  # sparse TMDB poster — should lose
            }
        ]

        full_doc = (
            "Plot: A young writer follows strangers.\n\n"
            "Genres: Thriller\n\n"
            "Director: Christopher Nolan\n\n"
            "Top Cast: Jeremy Theobald\n\n"
            "Title: Following (1998)\n\n"
        )

        with self._mock_fetch_candidates(candidates), \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids") as mock_fetch_by_ids:
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery()
            mock_parsed.person_names = ["Christopher Nolan"]
            mock_parsed.person_filmographies = [
                {"name": "Christopher Nolan", "person_id": 525,
                 "department": "directing", "titles": ["Memento", "Following"],
                 "movies": filmography_movies}
            ]
            mock_parse.return_value = mock_parsed

            mock_fetch_by_ids.return_value = [{
                "title": "Following",
                "movie_poster": "/following_db.jpg",
                "certification": "R",
                "document": full_doc,
            }]

            with patch("main.rerank_stream") as mock_rerank:
                mock_rerank.return_value = iter([
                    {"title": "Following", "explanation": "Nolan's debut."},
                    {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}},
                ])

                from main import search_stream
                results = [r for r in search_stream("Nolan films") if "__meta" not in r]

        assert len(results) == 1
        assert results[0]["title"] == "Following"
        assert results[0]["overview"] == "A young writer follows strangers."
        assert results[0]["genres"] == ["Thriller"]
        assert results[0]["director"] == "Christopher Nolan"
        assert results[0]["cast"] == ["Jeremy Theobald"]
        assert results[0]["certification"] == "R"
        assert results[0]["movie_poster"] == "/following_db.jpg"
        mock_fetch_by_ids.assert_called_once_with(["999"])

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
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]):
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

    def test_not_ingested_filmography_film_gets_cert_via_runtime_fetch(self, _no_real_cert_fetch):
        """A filmography film not yet in the vector DB has no cert in its sparse TMDB
        payload — the runtime batched fetch_certification backfills it so the rating
        still renders on first view."""
        _no_real_cert_fetch.return_value = "PG-13"  # override the autouse default
        candidates = [_make_candidate("Memento", year="2000")]
        filmography_movies = [{
            "id": 999, "title": "Following", "vote_average": 7.5, "vote_count": 500,
            "release_date": "1998-09-11", "poster_path": "/following.jpg",
        }]

        with self._mock_fetch_candidates(candidates), \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]):  # not ingested
            from query_parser import ParsedQuery
            mock_parsed = ParsedQuery()
            mock_parsed.person_names = ["Christopher Nolan"]
            mock_parsed.person_filmographies = [
                {"name": "Christopher Nolan", "person_id": 525, "department": "directing",
                 "titles": ["Memento", "Following"], "movies": filmography_movies}
            ]
            mock_parse.return_value = mock_parsed

            with patch("main.rerank_stream") as mock_rerank:
                mock_rerank.return_value = iter([
                    {"title": "Following", "explanation": "Nolan's debut."},
                    {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 1, "tools_called": []}},
                ])
                from main import search_stream
                results = [r for r in search_stream("Nolan films") if "__meta" not in r]

        assert results[0]["certification"] == "PG-13"
        _no_real_cert_fetch.assert_called_once_with(999)

    def test_mid_stream_filmography_sentinel_seeds_metadata(self, _no_real_cert_fetch):
        """A title Claude discovers via a mid-stream get_filmography call arrives as a
        __filmography sentinel; search_stream must seed its poster/cert so the subsequent
        result item isn't blank on both fields."""
        candidates = [_make_candidate("Memento", year="2000")]
        discovered = [{
            "id": 777, "title": "Insomnia", "release_date": "2002-05-24",
            "poster_path": "/insomnia.jpg",
        }]
        _no_real_cert_fetch.return_value = "R"

        with self._mock_fetch_candidates(candidates), \
             patch("main.parse_query") as mock_parse, \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]):  # discovered film not ingested
            from query_parser import ParsedQuery
            mock_parse.return_value = ParsedQuery()

            with patch("main.rerank_stream") as mock_rerank:
                # Sentinel precedes the result item that references the discovered title.
                mock_rerank.return_value = iter([
                    {"__filmography": discovered},
                    {"title": "Insomnia", "explanation": "Discovered via get_filmography."},
                    {"__usage": {"input_tokens": 5, "output_tokens": 3, "rounds": 2, "tools_called": ["get_filmography"]}},
                ])
                from main import search_stream
                results = [r for r in search_stream("something") if "__meta" not in r]

        assert len(results) == 1
        assert results[0]["title"] == "Insomnia"
        assert results[0]["movie_poster"] == "/insomnia.jpg"   # seeded — not blank
        assert results[0]["year"] == "2002"
        assert results[0]["certification"] == "R"              # seeded via runtime fetch
        _no_real_cert_fetch.assert_called_once_with(777)


# ---------------------------------------------------------------------------
# Reference-title anchoring (retrieval-by-document) + always-append guarantee
# ---------------------------------------------------------------------------

class TestAnchorRetrieval:
    def _parsed_with_refs(self, refs):
        from query_parser import ParsedQuery
        p = ParsedQuery()
        p.reference_titles = [r["title"] for r in refs]
        p.reference_movie_ids = [dict(r) for r in refs]
        return p

    def test_anchor_retrieves_document_neighbors_and_drops_self(self):
        from main import _fetch_candidates_anchored
        from config import ANCHOR_FETCH_DEPTH
        parsed = self._parsed_with_refs([{"title": "Inception", "id": 27205}])
        anchor_doc = {
            "title": "Inception", "movie_poster": "", "certification": "",
            "document": "Plot: A dream heist.\n\nTitle: Inception (2010)\n\n",
        }
        neighbors = [
            _make_vector_match("Inception", "Plot: self.\n\nTitle: Inception (2010)\n\n"),
            _make_vector_match("Tenet", "Plot: time.\n\nTitle: Tenet (2020)\n\n"),
            _make_vector_match("Memento", "Plot: memory.\n\nTitle: Memento (2000)\n\n"),
        ]
        model = _mock_model()
        with patch("main.vector_fetch_by_ids", return_value=[anchor_doc]), \
             patch("main.vector_count", return_value=100), \
             patch("main.get_model", return_value=model), \
             patch("main.vector_query", return_value=neighbors) as mock_vq:
            candidates, emb_ms, chroma_ms = _fetch_candidates_anchored("more like Inception", parsed)

        titles = [c["title"] for c in candidates]
        assert "Inception" not in titles                 # the anchor itself is dropped
        assert titles == ["Tenet", "Memento"]
        model.encode.assert_called_once_with(anchor_doc["document"])  # anchored on the DOCUMENT
        assert mock_vq.call_args[0][1] == min(ANCHOR_FETCH_DEPTH, 100)  # deep fetch
        assert "_doc" in parsed.reference_movie_ids[0]    # in-DB marker attached

    def test_cold_reference_falls_back_to_token_path(self):
        from main import _fetch_candidates_anchored
        parsed = self._parsed_with_refs([{"title": "Obscure Film", "id": 999999}])
        sentinel = ([{"title": "X"}], 7, 3)
        with patch("main.vector_fetch_by_ids", return_value=[]), \
             patch("main._fetch_candidates", return_value=sentinel) as mock_fc:
            result = _fetch_candidates_anchored("more like Obscure Film", parsed)
        assert result == sentinel
        mock_fc.assert_called_once()
        assert "_doc" not in parsed.reference_movie_ids[0]   # stays cold → will be ingested

    def test_multi_reference_interleaves_and_dedupes(self):
        from main import _fetch_candidates_anchored
        parsed = self._parsed_with_refs(
            [{"title": "Inception", "id": 1}, {"title": "Interstellar", "id": 2}]
        )
        doc1 = {"title": "Inception", "movie_poster": "", "certification": "",
                "document": "Plot: a.\n\nTitle: Inception (2010)\n\n"}
        doc2 = {"title": "Interstellar", "movie_poster": "", "certification": "",
                "document": "Plot: b.\n\nTitle: Interstellar (2014)\n\n"}
        list1 = [_make_vector_match("Tenet"), _make_vector_match("Shared")]
        list2 = [_make_vector_match("Gravity"), _make_vector_match("Shared")]
        with patch("main.vector_fetch_by_ids", side_effect=[[doc1], [doc2]]), \
             patch("main.vector_count", return_value=100), \
             patch("main.get_model", return_value=_mock_model()), \
             patch("main.vector_query", side_effect=[list1, list2]):
            candidates, _, _ = _fetch_candidates_anchored("like Inception and Interstellar", parsed)
        # Round-robin: Tenet (L1 r1), Gravity (L2 r1), Shared (L1 r2), Shared dup dropped.
        assert [c["title"] for c in candidates] == ["Tenet", "Gravity", "Shared"]

    def test_candidate_limit_widens_for_soft_qualifier(self):
        from main import _candidate_limit
        from config import SEARCH_CANDIDATES, ANCHOR_CANDIDATES_QUALIFIED
        from query_parser import ParsedQuery
        assert _candidate_limit(ParsedQuery()) == SEARCH_CANDIDATES
        p = ParsedQuery()
        p.has_soft_qualifier = True
        assert _candidate_limit(p) == ANCHOR_CANDIDATES_QUALIFIED


class TestReferenceGuarantee:
    def _mock_anchored(self, candidates):
        return patch("main._fetch_candidates_anchored", return_value=(candidates, 10, 5))

    def _mock_rerank_stream(self, results):
        def _gen(*a, **k):
            yield from results
            yield {"__usage": {"input_tokens": 1, "output_tokens": 1, "rounds": 1, "tools_called": []}}
        return patch("main.rerank_stream", side_effect=_gen)

    def _parsed_with_ref_doc(self):
        from query_parser import ParsedQuery
        p = ParsedQuery()
        p.reference_titles = ["Inception"]
        p.reference_movie_ids = [{
            "title": "Inception", "id": 27205,
            "_doc": {
                "title": "Inception", "movie_poster": "/p.jpg", "certification": "PG-13",
                "document": "Plot: dream heist.\n\nGenres: Science Fiction\n\nTitle: Inception (2010)\n\n",
            },
        }]
        return p

    def test_referenced_film_appended_when_missing(self):
        candidates = [_make_candidate("Tenet")]
        parsed = self._parsed_with_ref_doc()
        with self._mock_anchored(candidates), \
             self._mock_rerank_stream([{"title": "Tenet", "explanation": "similar"}]), \
             patch("main.parse_query", return_value=parsed), \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]), \
             patch("main.threading.Thread"):
            from main import search_stream
            out = [r for r in search_stream("more movies like Inception") if "__meta" not in r]
        titles = [r["title"] for r in out]
        assert "Inception" in titles
        inc = next(r for r in out if r["title"] == "Inception")
        assert inc["explanation"] == "The film you referenced."
        assert inc["movie_poster"] == "/p.jpg"
        assert inc["year"] == "2010"

    def test_referenced_film_not_duplicated_when_present(self):
        candidates = [_make_candidate("Inception")]
        parsed = self._parsed_with_ref_doc()
        with self._mock_anchored(candidates), \
             self._mock_rerank_stream([{"title": "Inception", "explanation": "the one"}]), \
             patch("main.parse_query", return_value=parsed), \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]), \
             patch("main.threading.Thread"):
            from main import search_stream
            out = [r for r in search_stream("more like Inception") if "__meta" not in r]
        titles = [r["title"] for r in out]
        assert titles.count("Inception") == 1
        assert out[0]["explanation"] == "the one"   # Claude's card kept, guarantee not re-added

    def test_cold_reference_not_appended(self):
        # Gate: a cold reference (resolved id but not in DB) is NOT guaranteed —
        # it gets background-ingested and only appears once anchored on a later search.
        from query_parser import ParsedQuery
        parsed = ParsedQuery()
        parsed.reference_titles = ["Obscure"]
        parsed.reference_movie_ids = [{"title": "Obscure", "id": 42}]   # no _doc → cold
        candidates = [_make_candidate("Something")]
        with self._mock_anchored(candidates), \
             self._mock_rerank_stream([{"title": "Something", "explanation": "x"}]), \
             patch("main.parse_query", return_value=parsed), \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]), \
             patch("main.threading.Thread"):
            from main import search_stream
            out = [r for r in search_stream("more like Obscure") if "__meta" not in r]
        assert [r["title"] for r in out] == ["Something"]   # Obscure not appended

    def test_cold_reference_triggers_background_ingestion(self):
        from query_parser import ParsedQuery
        parsed = ParsedQuery()
        parsed.reference_titles = ["Obscure"]
        parsed.reference_movie_ids = [{"title": "Obscure", "id": 42}]   # no _doc → cold
        candidates = [_make_candidate("Something")]
        with self._mock_anchored(candidates), \
             self._mock_rerank_stream([{"title": "Something", "explanation": "x"}]), \
             patch("main.parse_query", return_value=parsed), \
             patch("main._person_fetch_executor"), \
             patch("main.vector_fetch_by_ids", return_value=[]), \
             patch("main._ingest_reference_background") as mock_ingest, \
             patch("main.threading.Thread") as mock_thread:
            from main import search_stream
            list(search_stream("more like Obscure"))
        # A daemon thread targeting reference ingestion is started for the cold ref.
        assert any(
            kw.get("target") is mock_ingest or (a and a[0] is mock_ingest)
            for a, kw in [(c.args, c.kwargs) for c in mock_thread.call_args_list]
        )
