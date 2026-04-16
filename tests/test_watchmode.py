import json
import time
from unittest.mock import patch, MagicMock

import watchmode


def _reset_counters():
    """Reset in-memory counters between tests so they don't accumulate across the suite."""
    watchmode._api_calls = 0
    watchmode._api_calls_month = 0
    watchmode._cache_hits = 0


class TestSearchTitle:
    def setup_method(self):
        watchmode._cache.clear()
        _reset_counters()

    def test_returns_id_for_first_result(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2004}, {"id": 100, "year": 2005}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Some Movie") == 99

    def test_prefers_year_match_when_provided(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2004}, {"id": 100, "year": 2005}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Some Movie", "2005") == 100

    def test_falls_back_to_first_result_when_year_not_matched(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2004}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Some Movie", "1999") == 99

    def test_returns_none_when_no_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": []}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Nonexistent Movie XYZ") is None

    def test_returns_none_on_request_exception(self):
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", side_effect=Exception("timeout")):
            assert watchmode.search_title("Any Movie") is None


class TestFetchProviders:
    def setup_method(self):
        watchmode._cache.clear()
        _reset_counters()

    def test_includes_sub_rent_buy_free_sources(self):
        sources = [
            {"source_id": 203, "name": "Netflix", "type": "sub"},
            {"source_id": 300, "name": "Amazon", "type": "rent"},
            {"source_id": 387, "name": "Max", "type": "free"},
            {"source_id": 500, "name": "SomeChannel", "type": "tve"},  # excluded
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = sources
        logo_map = {203: "https://cdn.watchmode.com/netflix_100px.jpg", 387: "https://cdn.watchmode.com/max_100px.jpg"}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp), \
             patch("watchmode._source_logos", logo_map), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        names = {p["name"] for p in result}
        assert names == {"Netflix", "Amazon", "Max"}
        assert all("type" in p for p in result)

    def test_deduplicates_keeping_best_type(self):
        # Netflix appears as both sub and rent — sub should win
        sources = [
            {"source_id": 203, "name": "Netflix", "type": "rent"},
            {"source_id": 203, "name": "Netflix", "type": "sub"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = sources
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp), \
             patch("watchmode._source_logos", {}), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        assert len(result) == 1
        assert result[0]["type"] == "sub"

    def test_returns_none_logo_when_source_not_in_cache(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"source_id": 999, "name": "Obscure+", "type": "sub"}]
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp), \
             patch("watchmode._source_logos", {}), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        assert result == [{"name": "Obscure+", "type": "sub", "logo": None}]

    def test_returns_empty_list_on_request_exception(self):
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", side_effect=Exception("timeout")), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        assert result == []


class TestTTLCache:
    def setup_method(self):
        watchmode._cache.clear()
        _reset_counters()

    def test_search_title_returns_cached_result_on_second_call(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2010}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp) as mock_get:
            result1 = watchmode.search_title("Inception", "2010")
            result2 = watchmode.search_title("Inception", "2010")  # cache hit
        assert result1 == 99
        assert result2 == 99
        assert mock_get.call_count == 1  # only one real API call

    def test_search_title_caches_not_found(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": []}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp) as mock_get:
            result1 = watchmode.search_title("Unknown Movie XYZ", "")
            result2 = watchmode.search_title("Unknown Movie XYZ", "")  # cache hit (None)
        assert result1 is None
        assert result2 is None
        assert mock_get.call_count == 1

    def test_search_title_does_not_cache_on_exception(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2010}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", side_effect=[Exception("timeout"), mock_resp]) as mock_get:
            result1 = watchmode.search_title("Inception", "2010")  # fails, not cached
            result2 = watchmode.search_title("Inception", "2010")  # retry hits API
        assert result1 is None
        assert result2 == 99
        assert mock_get.call_count == 2

    def test_search_title_refetches_after_ttl_expires(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2010}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp) as mock_get:
            with patch("watchmode.time.time", return_value=0.0):
                watchmode.search_title("Inception", "2010")
            with patch("watchmode.time.time", return_value=float(watchmode._CACHE_TTL + 1)):
                result = watchmode.search_title("Inception", "2010")  # TTL expired
        assert result == 99
        assert mock_get.call_count == 2

    def test_fetch_providers_returns_cached_result_on_second_call(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"source_id": 203, "name": "Netflix", "type": "sub"}]
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp) as mock_get, \
             patch("watchmode._source_logos", {203: "https://cdn.example.com/netflix.jpg"}), \
             patch("watchmode._source_logos_loaded", True):
            result1 = watchmode.fetch_providers(12345)
            result2 = watchmode.fetch_providers(12345)  # cache hit
        assert result1 == result2
        assert mock_get.call_count == 1


class TestMonthlyCounter:
    def setup_method(self):
        watchmode._cache.clear()
        _reset_counters()

    def test_load_reads_count_when_month_matches(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"month": "2099-01", "count": 42}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)), \
             patch("watchmode._get_current_month", return_value="2099-01"):
            watchmode._load_persistent_counter()
        assert watchmode._api_calls_month == 42

    def test_load_resets_count_when_month_changed(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"month": "2020-01", "count": 99}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)), \
             patch("watchmode._get_current_month", return_value="2020-02"):
            watchmode._load_persistent_counter()
        assert watchmode._api_calls_month == 0

    def test_load_starts_at_zero_when_no_file(self, tmp_path):
        with patch("watchmode._COUNTER_FILE", str(tmp_path / "missing.json")):
            watchmode._load_persistent_counter()
        assert watchmode._api_calls_month == 0

    def test_increment_updates_both_counters(self):
        with patch("watchmode._persist_counter"):
            watchmode._increment_api_calls()
            watchmode._increment_api_calls()
        assert watchmode._api_calls == 2
        assert watchmode._api_calls_month == 2

    def test_get_stats_returns_monthly_count(self):
        watchmode._api_calls_month = 7
        stats = watchmode.get_stats()
        assert stats["api_calls_month"] == 7
        assert "api_calls_session" in stats
        assert stats["monthly_limit"] == 1000

    def test_persist_writes_month_and_count(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        watchmode._api_calls_month = 5
        with patch("watchmode._COUNTER_FILE", str(counter_file)), \
             patch("watchmode._get_current_month", return_value="2099-03"):
            watchmode._persist_counter()
        data = json.loads(counter_file.read_text())
        assert data == {"month": "2099-03", "count": 5}
