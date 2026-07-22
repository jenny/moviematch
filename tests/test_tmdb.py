import math
import pytest
from unittest.mock import patch, MagicMock
from tmdb import _composite_score, extract_certification, select_top_n, filter_cast, filter_crew, search_movie_by_title, fetch_watch_providers, get_filmography
from config import (
    TMDB_MIN_VOTE_COUNT,
    SCORE_WEIGHT_RATING,
    SCORE_WEIGHT_POPULARITY,
    CAST_LIMIT,
    CREW_JOBS,
)


def make_movie(id=1, title="Test Movie", vote_count=1000, vote_average=7.5, popularity=50.0):
    return {
        "id": id,
        "title": title,
        "vote_count": vote_count,
        "vote_average": vote_average,
        "popularity": popularity,
    }


class TestExtractCertification:
    def _movie(self, releases):
        return {"release_dates": {"results": [{"iso_3166_1": "US", "release_dates": releases}]}}

    def test_returns_theatrical_certification(self):
        movie = self._movie([
            {"type": 3, "certification": "PG-13", "release_date": "2010-07-16"},
        ])
        assert extract_certification(movie) == "PG-13"

    def test_prefers_theatrical_over_other_types(self):
        movie = self._movie([
            {"type": 4, "certification": "R", "release_date": "2010-01-01"},  # digital
            {"type": 3, "certification": "PG-13", "release_date": "2010-07-16"},  # theatrical
        ])
        assert extract_certification(movie) == "PG-13"

    def test_falls_back_to_any_cert_when_no_theatrical(self):
        movie = self._movie([
            {"type": 5, "certification": "R", "release_date": "2010-01-01"},  # physical
        ])
        assert extract_certification(movie) == "R"

    def test_returns_empty_when_no_us_entry(self):
        movie = {"release_dates": {"results": [{"iso_3166_1": "GB", "release_dates": [{"type": 3, "certification": "15"}]}]}}
        assert extract_certification(movie) == ""

    def test_returns_empty_when_all_certs_are_blank(self):
        movie = self._movie([{"type": 3, "certification": "", "release_date": "2010-07-16"}])
        assert extract_certification(movie) == ""

    def test_returns_empty_when_release_dates_key_absent(self):
        assert extract_certification({}) == ""


class TestCompositeScore:
    def test_matches_manual_calculation(self):
        movie = make_movie(vote_count=1000, vote_average=8.0, popularity=100.0)
        mean_rating = 7.0
        log_max_pop = math.log1p(100.0)

        v, R, p, m = 1000, 8.0, 100.0, TMDB_MIN_VOTE_COUNT
        wr = (v / (v + m)) * R + (m / (v + m)) * mean_rating
        expected = SCORE_WEIGHT_RATING * (wr / 10.0) + SCORE_WEIGHT_POPULARITY * 1.0

        assert _composite_score(movie, mean_rating, log_max_pop) == pytest.approx(expected)

    def test_zero_log_max_pop_gives_zero_popularity_component(self):
        movie = make_movie(popularity=999.0)
        score = _composite_score(movie, mean_rating=7.0, log_max_pop=0.0)

        v, R, m = 1000, 7.5, TMDB_MIN_VOTE_COUNT
        wr = (v / (v + m)) * R + (m / (v + m)) * 7.0
        expected = SCORE_WEIGHT_RATING * (wr / 10.0)

        assert score == pytest.approx(expected)

    def test_zero_vote_count_regresses_fully_to_mean(self):
        mean_rating = 6.5
        movie = make_movie(vote_count=0, vote_average=9.0, popularity=50.0)
        log_max_pop = math.log1p(50.0)
        score = _composite_score(movie, mean_rating, log_max_pop)

        # With v=0: wr = mean_rating regardless of R
        norm_pop = 1.0  # log1p(50) / log1p(50)
        expected = SCORE_WEIGHT_RATING * (mean_rating / 10.0) + SCORE_WEIGHT_POPULARITY * norm_pop

        assert score == pytest.approx(expected)

    def test_higher_vote_average_scores_higher(self):
        mean_rating = 7.0
        log_max_pop = math.log1p(50.0)
        low = _composite_score(make_movie(vote_average=5.0), mean_rating, log_max_pop)
        high = _composite_score(make_movie(vote_average=9.0), mean_rating, log_max_pop)
        assert high > low

    def test_higher_popularity_scores_higher(self):
        mean_rating = 7.0
        log_max_pop = math.log1p(500.0)
        low = _composite_score(make_movie(popularity=10.0), mean_rating, log_max_pop)
        high = _composite_score(make_movie(popularity=500.0), mean_rating, log_max_pop)
        assert high > low

    def test_missing_fields_default_to_zero(self):
        movie = {"id": 1, "title": "Sparse"}
        mean_rating = 7.0
        log_max_pop = math.log1p(100.0)
        score = _composite_score(movie, mean_rating, log_max_pop)

        # vote_count=0, vote_average=0, popularity=0 → wr = mean_rating, norm_pop = 0
        expected = SCORE_WEIGHT_RATING * (mean_rating / 10.0)

        assert score == pytest.approx(expected)

    def test_large_vote_count_approaches_actual_rating(self):
        # With very large vote_count, Bayesian weight approaches the actual vote_average
        movie = make_movie(vote_count=1_000_000, vote_average=8.5)
        mean_rating = 5.0
        log_max_pop = math.log1p(50.0)
        score = _composite_score(movie, mean_rating, log_max_pop)

        # wr ≈ R = 8.5 when v >> m
        approx_wr = 8.5
        norm_pop = 1.0
        approx_expected = SCORE_WEIGHT_RATING * (approx_wr / 10.0) + SCORE_WEIGHT_POPULARITY * norm_pop

        assert score == pytest.approx(approx_expected, abs=0.01)


class TestSelectTopN:
    def test_returns_correct_count(self):
        candidates = {i: make_movie(id=i) for i in range(5)}
        result = select_top_n(candidates, n=3)
        assert len(result) == 3

    def test_returns_ids_not_dicts(self):
        candidates = {1: make_movie(id=1)}
        result = select_top_n(candidates, n=1)
        assert result == [1]

    def test_highest_rated_ranked_first(self):
        candidates = {
            1: make_movie(id=1, vote_average=9.0, popularity=100.0, vote_count=5000),
            2: make_movie(id=2, vote_average=4.0, popularity=10.0, vote_count=500),
            3: make_movie(id=3, vote_average=6.0, popularity=30.0, vote_count=1000),
        }
        result = select_top_n(candidates, n=3)
        assert result[0] == 1

    def test_n_larger_than_candidates_returns_all(self):
        candidates = {1: make_movie(id=1), 2: make_movie(id=2)}
        result = select_top_n(candidates, n=100)
        assert len(result) == 2
        assert set(result) == {1, 2}


class TestFilterCast:
    def test_truncates_to_cast_limit(self):
        movie = {"credits": {"cast": [{"name": f"Actor {i}"} for i in range(CAST_LIMIT + 10)]}}
        result = filter_cast(movie)
        assert len(result["credits"]["cast"]) == CAST_LIMIT

    def test_preserves_order_within_limit(self):
        cast = [{"name": f"Actor {i}"} for i in range(3)]
        movie = {"credits": {"cast": cast}}
        result = filter_cast(movie)
        assert result["credits"]["cast"] == cast

    def test_no_truncation_when_under_limit(self):
        cast = [{"name": "Only Actor"}]
        movie = {"credits": {"cast": cast}}
        result = filter_cast(movie)
        assert result["credits"]["cast"] == cast


class TestFilterCrew:
    def test_removes_disallowed_jobs(self):
        movie = {"credits": {"crew": [
            {"name": "Alice", "job": "Director"},
            {"name": "Bob", "job": "Gaffer"},
            {"name": "Carol", "job": "Sound Mixer"},
        ]}}
        result = filter_crew(movie)
        jobs = {m["job"] for m in result["credits"]["crew"]}
        assert jobs.issubset(CREW_JOBS)

    def test_keeps_allowed_jobs(self):
        movie = {"credits": {"crew": [
            {"name": "Alice", "job": "Director"},
            {"name": "Bob", "job": "Producer"},
        ]}}
        result = filter_crew(movie)
        names = {m["name"] for m in result["credits"]["crew"]}
        assert names == {"Alice", "Bob"}

    def test_sorted_by_job(self):
        movie = {"credits": {"crew": [
            {"name": "Bob", "job": "Producer"},
            {"name": "Alice", "job": "Director"},
        ]}}
        result = filter_crew(movie)
        jobs = [m["job"] for m in result["credits"]["crew"]]
        assert jobs == sorted(jobs)

    def test_empty_crew_returns_empty(self):
        movie = {"credits": {"crew": []}}
        result = filter_crew(movie)
        assert result["credits"]["crew"] == []


class TestSearchMovieByTitle:
    # _require_tmdb_key() is patched to avoid needing a real key in CI
    def test_returns_id_of_first_result(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"id": 238, "title": "The Godfather"}]}
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp):
            result = search_movie_by_title("The Godfather", "1972")
        assert result == 238

    def test_returns_none_when_no_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp):
            result = search_movie_by_title("Unknown Movie XYZ")
        assert result is None

    def test_returns_none_on_request_exception(self):
        with patch("tmdb._require_tmdb_key"), patch("requests.get", side_effect=Exception("timeout")):
            result = search_movie_by_title("The Godfather")
        assert result is None

    def test_year_omitted_when_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"id": 1}]}
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp) as mock_get:
            search_movie_by_title("Some Movie", "")
        params = mock_get.call_args.kwargs["params"]
        assert "year" not in params

    def test_year_included_when_provided(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"id": 1}]}
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp) as mock_get:
            search_movie_by_title("Some Movie", "1994")
        params = mock_get.call_args.kwargs["params"]
        assert params["year"] == "1994"


class TestFetchWatchProviders:
    def test_returns_flatrate_providers_for_us(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": {
                "US": {
                    "flatrate": [
                        {"provider_name": "Netflix", "logo_path": "/netflix.jpg"},
                        {"provider_name": "Max", "logo_path": "/max.jpg"},
                    ]
                }
            }
        }
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp):
            result = fetch_watch_providers(238)
        assert result == [
            {"name": "Netflix", "type": "sub", "logo": "https://image.tmdb.org/t/p/w45/netflix.jpg"},
            {"name": "Max", "type": "sub", "logo": "https://image.tmdb.org/t/p/w45/max.jpg"},
        ]

    def test_returns_empty_list_when_no_us_entry(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"GB": {"flatrate": [{"provider_name": "BFI", "logo_path": "/bfi.jpg"}]}}}
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp):
            result = fetch_watch_providers(238)
        assert result == []

    def test_returns_empty_list_when_no_flatrate(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"US": {"rent": []}}}
        with patch("tmdb._require_tmdb_key"), patch("requests.get", return_value=mock_resp):
            result = fetch_watch_providers(238)
        assert result == []

    def test_returns_empty_list_on_request_exception(self):
        with patch("tmdb._require_tmdb_key"), patch("requests.get", side_effect=Exception("timeout")):
            result = fetch_watch_providers(238)
        assert result == []


class TestGetFilmography:
    def _mock_response(self, crew=None, cast=None):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"crew": crew or [], "cast": cast or []}
        return mock_resp

    def _director_entry(self, id=1, title="Movie A", poster="/a.jpg", release="2010-01-01"):
        return {"id": id, "title": title, "job": "Director",
                "poster_path": poster, "release_date": release,
                "vote_average": 7.5, "vote_count": 500}

    def _cast_entry(self, id=1, title="Movie B", poster="/b.jpg", release="2012-05-20", character="Hero"):
        return {"id": id, "title": title, "character": character,
                "poster_path": poster, "release_date": release,
                "vote_average": 7.0, "vote_count": 300}

    def test_directing_includes_poster_path(self):
        mock_resp = self._mock_response(crew=[self._director_entry()])
        with patch("tmdb._require_tmdb_key"), \
             patch("requests.get", return_value=mock_resp), \
             patch("tmdb.time.sleep"):
            movies = get_filmography(1, department="directing")
        assert movies[0]["poster_path"] == "/a.jpg"

    def test_cast_includes_poster_path(self):
        mock_resp = self._mock_response(cast=[self._cast_entry()])
        with patch("tmdb._require_tmdb_key"), \
             patch("requests.get", return_value=mock_resp), \
             patch("tmdb.time.sleep"):
            movies = get_filmography(1, department="cast")
        assert movies[0]["poster_path"] == "/b.jpg"

    def test_missing_poster_path_defaults_to_empty_string(self):
        # Build a TMDB crew entry that has no poster_path key at all
        entry = {"id": 99, "title": "No Poster Film", "job": "Director",
                 "release_date": "2005-03-01", "vote_average": 6.5, "vote_count": 200}
        mock_resp = self._mock_response(crew=[entry])
        with patch("tmdb._require_tmdb_key"), \
             patch("requests.get", return_value=mock_resp), \
             patch("tmdb.time.sleep"):
            movies = get_filmography(99, department="directing")
        assert movies[0]["poster_path"] == ""

    def test_cast_ranks_by_composite_not_raw_average(self):
        # Regression: a single-vote 10.0 credit (e.g. a WWE special) must not
        # outrank a genuinely popular film. Raw vote_average sorting put obscure
        # one-vote titles at the top, burying real movies past the 30-item cap.
        blockbuster = {"id": 1, "title": "Jumanji", "character": "Hero",
                       "poster_path": "/j.jpg", "release_date": "2017-12-20",
                       "vote_average": 6.8, "vote_count": 14570, "popularity": 120.0}
        one_vote_ten = {"id": 2, "title": "Obscure WWE Special", "character": "Self",
                        "poster_path": "/w.jpg", "release_date": "1998-01-01",
                        "vote_average": 10.0, "vote_count": 1, "popularity": 0.3}
        mock_resp = self._mock_response(cast=[one_vote_ten, blockbuster])
        # Unique person_id — get_filmography is lru_cached, so reusing an id from
        # another test would return that test's cached result instead of this mock.
        with patch("tmdb._require_tmdb_key"), \
             patch("requests.get", return_value=mock_resp), \
             patch("tmdb.time.sleep"):
            movies = get_filmography(70001, department="cast")
        assert movies[0]["title"] == "Jumanji"

    def test_cast_caps_at_30_keeping_top_ranked(self):
        # With more than 30 credits, the cap must keep the highest composite-scored
        # films. Build 35 popular films plus one single-vote 10.0 outlier; the
        # outlier must be dropped, not retained at the expense of a real film.
        cast = [
            {"id": i, "title": f"Popular {i}", "character": "Role",
             "poster_path": "", "release_date": "2015-01-01",
             "vote_average": 7.0, "vote_count": 5000, "popularity": 80.0}
            for i in range(35)
        ]
        cast.append({"id": 999, "title": "One Vote Wonder", "character": "Self",
                     "poster_path": "", "release_date": "1990-01-01",
                     "vote_average": 10.0, "vote_count": 1, "popularity": 0.1})
        mock_resp = self._mock_response(cast=cast)
        # Unique person_id to sidestep get_filmography's lru_cache (see above).
        with patch("tmdb._require_tmdb_key"), \
             patch("requests.get", return_value=mock_resp), \
             patch("tmdb.time.sleep"):
            movies = get_filmography(70002, department="cast")
        assert len(movies) == 30
        assert "One Vote Wonder" not in {m["title"] for m in movies}


class TestWarmup:
    """Startup TMDB connection primer — best-effort, must never raise or block startup."""

    def test_primes_configuration_endpoint(self):
        from tmdb import warmup
        with patch("tmdb.TMDB_KEY", "fake"), patch("requests.get") as mock_get:
            warmup()
        assert mock_get.called
        assert "/configuration" in mock_get.call_args[0][0]

    def test_noop_without_key(self):
        from tmdb import warmup
        with patch("tmdb.TMDB_KEY", ""), patch("requests.get") as mock_get:
            warmup()
        mock_get.assert_not_called()

    def test_swallows_errors(self):
        """A cold-connection failure at startup must not propagate."""
        from tmdb import warmup
        with patch("tmdb.TMDB_KEY", "fake"), patch("requests.get", side_effect=Exception("cold")):
            warmup()  # must not raise
