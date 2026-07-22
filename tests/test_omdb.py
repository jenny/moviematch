from unittest.mock import patch, MagicMock

import omdb


# A representative full OMDb response (trimmed to the fields fetch_ratings reads).
OMDB_FULL = {
    "Title": "Inception",
    "Year": "2010",
    "Ratings": [
        {"Source": "Internet Movie Database", "Value": "8.8/10"},
        {"Source": "Rotten Tomatoes", "Value": "87%"},
        {"Source": "Metacritic", "Value": "74/100"},
    ],
    "Metascore": "74",
    "imdbRating": "8.8",
    "imdbID": "tt1375666",
    "Response": "True",
}


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestParsers:
    def test_parse_percent(self):
        assert omdb._parse_percent("87%") == 87
        assert omdb._parse_percent("100%") == 100
        assert omdb._parse_percent("N/A") is None
        assert omdb._parse_percent("") is None

    def test_parse_ratio(self):
        assert omdb._parse_ratio("8.8/10") == 8.8
        assert omdb._parse_ratio("8.8") == 8.8
        assert omdb._parse_ratio("N/A") is None
        assert omdb._parse_ratio("") is None

    def test_parse_int(self):
        assert omdb._parse_int("74") == 74
        assert omdb._parse_int("74/100") == 74
        assert omdb._parse_int("N/A") is None
        assert omdb._parse_int("") is None


class TestFetchRatings:
    def setup_method(self):
        omdb._cache.clear()

    def test_failsoft_returns_empty_when_key_unset(self):
        """No OMDB_API_KEY → return {} without making any HTTP call."""
        with patch("omdb.OMDB_API_KEY", None), \
             patch("requests.get") as mock_get:
            assert omdb.fetch_ratings("Inception", "2010") == {}
        mock_get.assert_not_called()

    def test_parses_all_scores_and_imdb_id(self):
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", return_value=_mock_response(OMDB_FULL)):
            result = omdb.fetch_ratings("Inception", "2010")
        assert result == {"rt": 87, "imdb": 8.8, "metacritic": 74, "imdb_id": "tt1375666"}

    def test_omits_missing_rt(self):
        payload = dict(OMDB_FULL, Ratings=[{"Source": "Internet Movie Database", "Value": "8.8/10"}])
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", return_value=_mock_response(payload)):
            result = omdb.fetch_ratings("Some Movie", "")
        assert "rt" not in result
        assert result["imdb"] == 8.8

    def test_omits_na_metascore(self):
        payload = dict(OMDB_FULL, Metascore="N/A")
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", return_value=_mock_response(payload)):
            result = omdb.fetch_ratings("Some Movie", "")
        assert "metacritic" not in result

    def test_not_found_returns_empty_and_caches_miss(self):
        payload = {"Response": "False", "Error": "Movie not found!"}
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", return_value=_mock_response(payload)) as mock_get:
            r1 = omdb.fetch_ratings("Nonexistent XYZ", "")
            r2 = omdb.fetch_ratings("Nonexistent XYZ", "")  # cache hit
        assert r1 == {}
        assert r2 == {}
        assert mock_get.call_count == 1  # miss was cached — only one HTTP call

    def test_caches_successful_result(self):
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", return_value=_mock_response(OMDB_FULL)) as mock_get:
            r1 = omdb.fetch_ratings("Inception", "2010")
            r2 = omdb.fetch_ratings("Inception", "2010")  # cache hit
        assert r1 == r2
        assert mock_get.call_count == 1

    def test_does_not_cache_on_exception(self):
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", side_effect=[Exception("timeout"), _mock_response(OMDB_FULL)]) as mock_get:
            r1 = omdb.fetch_ratings("Inception", "2010")  # fails → {}
            r2 = omdb.fetch_ratings("Inception", "2010")  # retry hits API
        assert r1 == {}
        assert r2["rt"] == 87
        assert mock_get.call_count == 2

    def test_cache_key_isolated_by_year(self):
        """Same title, different year must not share a cache entry."""
        payload_2010 = dict(OMDB_FULL, imdbRating="8.8")
        payload_other = dict(OMDB_FULL, imdbRating="6.0", imdbID="tt9999999")

        def fake_get(url, **kwargs):
            year = kwargs.get("params", {}).get("y")
            return _mock_response(payload_2010 if year == "2010" else payload_other)

        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", side_effect=fake_get):
            r2010 = omdb.fetch_ratings("Inception", "2010")
            rother = omdb.fetch_ratings("Inception", "1999")
        assert r2010["imdb"] == 8.8
        assert rother["imdb"] == 6.0
        assert rother["imdb_id"] == "tt9999999"

    def test_refetches_after_ttl_expires(self):
        with patch("omdb.OMDB_API_KEY", "key"), \
             patch("requests.get", return_value=_mock_response(OMDB_FULL)) as mock_get:
            with patch("omdb.time.time", return_value=0.0):
                omdb.fetch_ratings("Inception", "2010")
            with patch("omdb.time.time", return_value=float(omdb._CACHE_TTL + 1)):
                result = omdb.fetch_ratings("Inception", "2010")  # TTL expired
        assert result["rt"] == 87
        assert mock_get.call_count == 2
