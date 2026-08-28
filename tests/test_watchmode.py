import json
import logging
import time
from unittest.mock import patch, MagicMock

import pytest
import requests

import watchmode


@pytest.fixture(autouse=True)
def _isolate_counter_file(tmp_path, monkeypatch):
    """Point the persistent counter at a tmp file for every test in this module.

    _increment_api_calls() writes to disk on each simulated API call, and not every
    test stubs _persist_counter (the redaction tests deliberately let the call path
    run until requests raises). Without this, a plain `pytest` run rewrote the
    developer's real logs/watchmode_calls.json and inflated the live counter.
    """
    monkeypatch.setattr(watchmode, "_COUNTER_FILE", str(tmp_path / "watchmode_calls.json"))


def _reset_counters():
    """Reset in-memory counters between tests so they don't accumulate across the suite."""
    watchmode._api_calls = 0
    watchmode._counts.clear()
    watchmode._cache_hits = 0


class TestLoadSourceLogos:
    def setup_method(self):
        watchmode._source_logos.clear()
        watchmode._source_logos_loaded = False
        _reset_counters()

    def test_stores_valid_logo_urls(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 203, "logo_100px": "https://cdn.watchmode.com/provider_logos/netflix_100px.jpg"},
        ]
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            watchmode._load_source_logos()
        assert 203 in watchmode._source_logos
        assert watchmode._source_logos[203].endswith("netflix_100px.jpg")

    def test_rejects_null_filename_logo_urls(self):
        """Watchmode sometimes returns '.../provider_logos/null' for missing logos."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 1, "logo_100px": "https://cdn.watchmode.com/provider_logos/null"},
            {"id": 2, "logo_100px": "https://cdn.watchmode.com/provider_logos/null/"},
        ]
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            watchmode._load_source_logos()
        assert 1 not in watchmode._source_logos
        assert 2 not in watchmode._source_logos

    def test_skips_sources_with_no_logo(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 99, "logo_100px": None},
        ]
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp):
            watchmode._load_source_logos()
        assert 99 not in watchmode._source_logos


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
        assert result == [
            {"name": "Obscure+", "type": "sub", "logo": None, "url": None, "price": None}
        ]

    def test_returns_empty_list_on_request_exception(self):
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", side_effect=Exception("timeout")), \
             patch("watchmode._source_logos_loaded", True):
            result = watchmode.fetch_providers(12345)
        assert result == []

    def test_cache_key_isolated_by_country(self):
        """US and GB lookups for the same title_id must not share a cache entry."""
        us_sources = [{"source_id": 203, "name": "Netflix US", "type": "sub"}]
        gb_sources = [{"source_id": 999, "name": "MUBI", "type": "sub"}]

        def fake_get(url, **kwargs):
            resp = MagicMock()
            if "regions=US" in url or kwargs.get("params", {}).get("regions") == "US":
                resp.json.return_value = us_sources
            else:
                resp.json.return_value = gb_sources
            return resp

        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", side_effect=fake_get), \
             patch("watchmode._source_logos", {}), \
             patch("watchmode._source_logos_loaded", True):
            us_result = watchmode.fetch_providers(12345, "US")
            gb_result = watchmode.fetch_providers(12345, "GB")

        assert us_result[0]["name"] == "Netflix US"
        assert gb_result[0]["name"] == "MUBI"


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

    def test_load_reads_keyed_counts(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"2099-01": 42, "2099-02": 7}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {"2099-01": 42, "2099-02": 7}

    def test_load_starts_empty_when_no_file(self, tmp_path):
        # Seeded here, not left to setup_method: these "starts empty" tests used to pass
        # only because the fixture pre-cleared _counts, so they would have passed against
        # a _load_persistent_counter() that did nothing at all.
        watchmode._counts.update({"2099-01": 99})
        with patch("watchmode._COUNTER_FILE", str(tmp_path / "missing.json")):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {}

    def test_load_migrates_legacy_single_count_format(self, tmp_path):
        """The old {"month", "count"} file must not be silently discarded on deploy."""
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"month": "2099-01", "count": 42}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {"2099-01": 42}

    def test_load_rejects_legacy_entry_with_a_non_month_label(self, tmp_path):
        """The legacy branch writes straight into _counts, so it needs the same key
        check as the keyed branch — otherwise a malformed label seeds exactly the junk
        key the keyed branch refuses, and the panel paints it as the current month."""
        watchmode._counts.update({"2099-01": 99})
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"month": "not-a-month", "count": 9}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {}

    def test_load_survives_corrupt_file(self, tmp_path):
        watchmode._counts.update({"2099-01": 99})
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text("{not json")
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {}

    def test_load_survives_unexpected_shape(self, tmp_path):
        watchmode._counts.update({"2099-01": 99})
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps(["not", "a", "dict"]))
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {}

    def test_load_skips_malformed_entries(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"2099-01": 42, "2099-02": "bogus"}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {"2099-01": 42}

    def test_load_rejects_non_month_keys(self, tmp_path):
        """A stray key sorts after every real month (letters beat digits), so it would
        land last in get_stats()'s ordering — and the panel treats the last entry as the
        current month, painting severity and "so far" onto a junk row."""
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"2099-01": 42, "count": 9, "2099-1": 3, "": 1}))
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._load_persistent_counter()
        assert watchmode._counts == {"2099-01": 42}

    def test_increment_updates_session_and_current_month(self):
        with patch("watchmode._persist_counter"), \
             patch("watchmode._get_current_month", return_value="2099-05"):
            watchmode._increment_api_calls()
            watchmode._increment_api_calls()
        assert watchmode._api_calls == 2
        assert watchmode._counts == {"2099-05": 2}

    def test_rollover_while_process_stays_alive(self):
        """The regression this whole change exists for.

        A long-lived container (Railway runs for weeks) is never re-imported, so the
        old design carried the previous month's total forward and then re-stamped it
        with the new month. Here the process stays up across the boundary: the new
        month must start at 1, and the old month's total must survive intact.
        """
        with patch("watchmode._persist_counter"):
            with patch("watchmode._get_current_month", return_value="2099-07"):
                for _ in range(847):
                    watchmode._increment_api_calls()
            # ---- calendar month changes; same process, no restart ----
            with patch("watchmode._get_current_month", return_value="2099-08"):
                watchmode._increment_api_calls()
                stats = watchmode.get_stats()

        assert watchmode._counts["2099-08"] == 1, "new month must not inherit the old total"
        assert watchmode._counts["2099-07"] == 847, "previous month must be preserved"
        assert stats["api_calls_month"] == 1

    def test_get_stats_reports_zero_for_new_month_before_any_call(self):
        """After a rollover the panel should read 0 immediately, not the old month's total."""
        watchmode._counts.update({"2099-07": 847})
        with patch("watchmode._get_current_month", return_value="2099-08"):
            stats = watchmode.get_stats()
        assert stats["api_calls_month"] == 0

    def test_get_stats_returns_monthly_count(self):
        watchmode._counts.update({"2099-09": 7})
        with patch("watchmode._get_current_month", return_value="2099-09"):
            stats = watchmode.get_stats()
        assert stats["api_calls_month"] == 7
        assert "api_calls_session" in stats
        assert stats["monthly_limit"] == 1000

    def test_persist_writes_keyed_counts(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        watchmode._counts.update({"2099-03": 5, "2099-04": 2})
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._persist_counter()
        data = json.loads(counter_file.read_text())
        assert data == {"2099-03": 5, "2099-04": 2}

    def test_increment_does_not_prune_the_key_it_is_incrementing(self):
        """_prune_history() used to run between creating the month's key and using it.
        With retention full and the current month sorting oldest — a clock rollback, or
        a file seeded with future-dated keys — prune deleted the key the next line
        touched, raising KeyError inside the request path."""
        for i in range(1, watchmode._COUNTER_HISTORY_MONTHS + 1):
            watchmode._counts[f"2200-{i:02d}"] = 10
        with patch("watchmode._persist_counter"), \
             patch("watchmode._get_current_month", return_value="2100-01"):
            watchmode._increment_api_calls()  # must not raise
        assert watchmode._counts["2100-01"] == 1
        assert len(watchmode._counts) == watchmode._COUNTER_HISTORY_MONTHS

    def test_persist_is_atomic_and_leaves_no_temp_file(self, tmp_path):
        counter_file = tmp_path / "watchmode_calls.json"
        watchmode._counts.update({"2099-03": 5})
        with patch("watchmode._COUNTER_FILE", str(counter_file)):
            watchmode._persist_counter()
        assert json.loads(counter_file.read_text()) == {"2099-03": 5}
        assert list(tmp_path.glob("*.tmp")) == [], "temp file left behind after a good write"

    def test_persist_failure_leaves_the_previous_file_intact(self, tmp_path):
        """The whole point of the temp-file dance: a crash mid-write must not destroy
        the retained history, which cannot be reconstructed."""
        counter_file = tmp_path / "watchmode_calls.json"
        counter_file.write_text(json.dumps({"2099-01": 847}))
        watchmode._counts.update({"2099-02": 5})
        with patch("watchmode._COUNTER_FILE", str(counter_file)), \
             patch("watchmode.os.replace", side_effect=OSError("boom")):
            watchmode._persist_counter()  # must not raise
        assert json.loads(counter_file.read_text()) == {"2099-01": 847}
        assert list(tmp_path.glob("*.tmp")) == [], "partial temp file left behind"

    def test_persist_failure_does_not_raise(self, tmp_path):
        """A read-only filesystem (no volume mounted) must not break user requests."""
        watchmode._counts.update({"2099-03": 5})
        with patch("watchmode._COUNTER_FILE", str(tmp_path / "sub" / "counts.json")), \
             patch("watchmode.open", side_effect=OSError("read-only filesystem")):
            watchmode._persist_counter()  # must not raise

    def test_history_is_pruned_to_retention_limit(self):
        """The file is unbounded otherwise — one key per month, forever."""
        with patch("watchmode._persist_counter"):
            for month in range(1, 15):  # 14 distinct months
                with patch("watchmode._get_current_month", return_value=f"2099-{month:02d}"):
                    watchmode._increment_api_calls()
        assert len(watchmode._counts) == watchmode._COUNTER_HISTORY_MONTHS
        assert "2099-01" not in watchmode._counts, "oldest months should be dropped"
        assert "2099-14" in watchmode._counts, "newest month must be kept"


class TestUsageHistory:
    """get_stats()["history"] — the series behind the admin panel's monthly chart."""

    def setup_method(self):
        watchmode._cache.clear()
        _reset_counters()

    def test_history_is_oldest_first(self):
        watchmode._counts.update({"2099-03": 300, "2099-01": 100, "2099-02": 200})
        with patch("watchmode._get_current_month", return_value="2099-03"):
            history = watchmode.get_stats()["history"]
        assert history == [
            {"month": "2099-01", "calls": 100},
            {"month": "2099-02", "calls": 200},
            {"month": "2099-03", "calls": 300},
        ]

    def test_current_month_included_at_zero_before_any_call(self):
        """It has no _counts key until its first call; omitting it would drop the
        newest bar off the chart for the first hours of every month."""
        watchmode._counts.update({"2099-07": 847})
        with patch("watchmode._get_current_month", return_value="2099-08"):
            history = watchmode.get_stats()["history"]
        assert history[-1] == {"month": "2099-08", "calls": 0}
        assert history[0] == {"month": "2099-07", "calls": 847}

    def test_history_has_current_month_only_when_no_data(self):
        with patch("watchmode._get_current_month", return_value="2099-08"):
            assert watchmode.get_stats()["history"] == [{"month": "2099-08", "calls": 0}]

    def test_history_never_exceeds_retention(self):
        """_counts can already hold a full 12 months of *past* data, and get_stats
        zero-fills the current month on top — which made a 13th entry after every
        rollover, until the new month's first call."""
        for i in range(1, watchmode._COUNTER_HISTORY_MONTHS + 1):
            watchmode._counts[f"2099-{i:02d}"] = 10
        with patch("watchmode._get_current_month", return_value="2100-01"):
            history = watchmode.get_stats()["history"]
        assert len(history) == watchmode._COUNTER_HISTORY_MONTHS
        assert history[-1] == {"month": "2100-01", "calls": 0}, "newest must be the current month"
        assert history[0]["month"] == "2099-02", "the oldest month is the one dropped"

    def test_history_does_not_mutate_stored_counts(self):
        """The zero-fill is for display only — it must not create a real counter key,
        or _prune_history would start evicting real months to make room for empties."""
        watchmode._counts.update({"2099-07": 847})
        with patch("watchmode._get_current_month", return_value="2099-08"):
            watchmode.get_stats()
        assert watchmode._counts == {"2099-07": 847}


class TestApiKeyRedaction:
    """Watchmode authenticates by query param, so a raw exception leaks the key.

    requests puts the full request URL — query string included — into every
    HTTPError message, so an unredacted `except ... {e}` would print apiKey=... to
    the logs (stdout in production). Every Watchmode failure log must scrub it.
    """

    KEY = "wm_s3cret_key_value"

    def _leaky_error(self, path):
        return requests.exceptions.HTTPError(
            f"401 Client Error: Unauthorized for url: "
            f"https://api.watchmode.com/v1{path}?apiKey={self.KEY}"
        )

    def setup_method(self):
        watchmode._cache.clear()
        watchmode._source_logos.clear()
        watchmode._source_logos_loaded = False
        _reset_counters()

    def test_search_failure_redacts_key(self, caplog):
        with patch("config.WATCHMODE_API_KEY", self.KEY), \
             patch("requests.get", side_effect=self._leaky_error("/search/")):
            with caplog.at_level(logging.WARNING, logger="watchmode"):
                assert watchmode.search_title("Inception", "2010") is None
        assert self.KEY not in caplog.text
        assert "***" in caplog.text

    def test_sources_failure_redacts_key(self, caplog):
        watchmode._source_logos_loaded = True  # skip the catalog call
        with patch("config.WATCHMODE_API_KEY", self.KEY), \
             patch("requests.get", side_effect=self._leaky_error("/title/123/sources/")):
            with caplog.at_level(logging.WARNING, logger="watchmode"):
                assert watchmode.fetch_providers(123, "US") == []
        assert self.KEY not in caplog.text
        assert "***" in caplog.text

    def test_source_logo_catalog_failure_redacts_key(self, caplog):
        with patch("config.WATCHMODE_API_KEY", self.KEY), \
             patch("requests.get", side_effect=self._leaky_error("/sources/")):
            with caplog.at_level(logging.WARNING, logger="watchmode"):
                watchmode._load_source_logos()
        assert self.KEY not in caplog.text
        assert "***" in caplog.text


class TestProviderDeepLinks:
    """web_url / price pass-through, and the format-level dedupe that comes with it.

    Watchmode emits one row per format (SD/HD/4K) for the same provider, each with its own
    price, so these rows are not duplicates to be discarded — they have to be merged.
    """

    def setup_method(self):
        watchmode._cache.clear()
        _reset_counters()

    def _fetch(self, sources, logo_map=None):
        mock_resp = MagicMock()
        mock_resp.json.return_value = sources
        with patch("watchmode.WATCHMODE_API_KEY", "key"), \
             patch("watchmode._persist_counter"), \
             patch("requests.get", return_value=mock_resp), \
             patch("watchmode._source_logos", logo_map or {}), \
             patch("watchmode._source_logos_loaded", True):
            return watchmode.fetch_providers(12345)

    def test_surfaces_web_url_and_price(self):
        result = self._fetch([
            {"source_id": 203, "name": "Netflix", "type": "sub",
             "web_url": "https://www.netflix.com/title/70131314", "price": None},
            {"source_id": 307, "name": "Fandango", "type": "rent",
             "web_url": "https://athome.fandango.com/x/1", "price": 4.99},
        ])
        by_name = {p["name"]: p for p in result}
        assert by_name["Netflix"]["url"] == "https://www.netflix.com/title/70131314"
        assert by_name["Netflix"]["price"] is None
        assert by_name["Fandango"]["url"] == "https://athome.fandango.com/x/1"
        assert by_name["Fandango"]["price"] == 4.99

    def test_same_type_multiple_formats_keeps_cheapest_price(self):
        # Real shape: one row per format, same provider, same type, different prices.
        result = self._fetch([
            {"source_id": 307, "name": "Fandango", "type": "rent",
             "web_url": "https://athome.fandango.com/x/1", "price": 5.99, "format": "4K"},
            {"source_id": 307, "name": "Fandango", "type": "rent",
             "web_url": "https://athome.fandango.com/x/1", "price": 3.99, "format": "SD"},
            {"source_id": 307, "name": "Fandango", "type": "rent",
             "web_url": "https://athome.fandango.com/x/1", "price": 4.99, "format": "HD"},
        ])
        assert len(result) == 1
        assert result[0]["price"] == 3.99

    def test_better_type_wins_over_cheaper_price(self):
        # A $0 rent row must not displace the subscription entry.
        result = self._fetch([
            {"source_id": 203, "name": "Netflix", "type": "rent", "price": 0.99},
            {"source_id": 203, "name": "Netflix", "type": "sub", "price": None},
        ])
        assert len(result) == 1
        assert result[0]["type"] == "sub"
        assert result[0]["price"] is None

    def test_missing_url_backfilled_from_sibling_row(self):
        result = self._fetch([
            {"source_id": 307, "name": "Fandango", "type": "rent", "price": 5.99},
            {"source_id": 307, "name": "Fandango", "type": "rent",
             "web_url": "https://athome.fandango.com/x/1", "price": 6.99},
        ])
        assert result[0]["url"] == "https://athome.fandango.com/x/1"
        assert result[0]["price"] == 5.99  # cheaper row still wins on price

    def test_paid_plan_placeholder_is_not_a_url(self):
        """Free-tier ios_url/android_url read as prose; the same must never reach an href."""
        result = self._fetch([
            {"source_id": 203, "name": "Netflix", "type": "sub",
             "web_url": "Deeplinks available for paid plans only."},
        ])
        assert result[0]["url"] is None

    def test_non_http_scheme_rejected(self):
        result = self._fetch([
            {"source_id": 203, "name": "Evil", "type": "sub", "web_url": "javascript:alert(1)"},
        ])
        assert result[0]["url"] is None

    def test_absent_fields_default_to_none(self):
        result = self._fetch([{"source_id": 203, "name": "Netflix", "type": "sub"}])
        assert result[0]["url"] is None
        assert result[0]["price"] is None

    def test_non_numeric_price_ignored(self):
        result = self._fetch([
            {"source_id": 307, "name": "Fandango", "type": "rent", "price": "N/A"},
        ])
        assert result[0]["price"] is None
