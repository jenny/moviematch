from unittest.mock import patch, MagicMock

import watchmode


class TestSearchTitle:
    def test_returns_id_for_first_result(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2004}, {"id": 100, "year": 2005}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Some Movie") == 99

    def test_prefers_year_match_when_provided(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2004}, {"id": 100, "year": 2005}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Some Movie", "2005") == 100

    def test_falls_back_to_first_result_when_year_not_matched(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": [{"id": 99, "year": 2004}]}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Some Movie", "1999") == 99

    def test_returns_none_when_no_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"title_results": []}
        with patch("watchmode.WATCHMODE_API_KEY", "key"), patch("requests.get", return_value=mock_resp):
            assert watchmode.search_title("Nonexistent Movie XYZ") is None

    def test_returns_none_on_request_exception(self):
        with patch("watchmode.WATCHMODE_API_KEY", "key"), patch("requests.get", side_effect=Exception("timeout")):
            assert watchmode.search_title("Any Movie") is None


class TestFetchProviders:
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
             patch("requests.get", return_value=mock_resp), \
             patch("watchmode._source_logos", {}), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        assert result == [{"name": "Obscure+", "type": "sub", "logo": None}]

    def test_returns_empty_list_on_request_exception(self):
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("requests.get", side_effect=Exception("timeout")), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        assert result == []
