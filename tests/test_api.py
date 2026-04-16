import json
import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.app import app


def parse_sse(response_text: str) -> list[dict]:
    """Parse SSE response body into a list of event data dicts."""
    events = []
    for chunk in response_text.strip().split("\n\n"):
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def make_meta(usage=None, error=None):
    return {"__meta": {
        "embedding_ms": 10,
        "chroma_ms": 5,
        "claude_ms": 200,
        "usage": usage or {
            "input_tokens": 100, "output_tokens": 50,
            "haiku_input_tokens": 100, "haiku_output_tokens": 50,
            "opus_input_tokens": 0, "opus_output_tokens": 0,
            "rounds": 1, "tools_called": [],
        },
        "error": error,
    }}


@pytest.fixture(scope="module")
def client():
    with patch("db.get_model", return_value=MagicMock()), \
         patch("api.routes.search.log_request"):  # prevent test runs from writing to search.log
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSearchEndpoint:
    def test_happy_path_streams_results_and_done(self, client):
        stream = iter([
            {"title": "Inception", "explanation": "Mind-bending", "movie_poster": ""},
            make_meta(),
        ])
        with patch("api.routes.search.search_stream", return_value=stream):
            response = client.post("/recommend", json={"query": "mind-bending sci-fi"})

        assert response.status_code == 200
        events = parse_sse(response.text)
        result_events = [e for e in events if e["type"] == "result"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(result_events) == 1
        assert result_events[0]["title"] == "Inception"
        assert len(done_events) == 1
        assert done_events[0]["result_count"] == 1

    def test_multiple_results_all_appear(self, client):
        stream = iter([
            {"title": "A", "explanation": "First", "movie_poster": ""},
            {"title": "B", "explanation": "Second", "movie_poster": ""},
            {"title": "C", "explanation": "Third", "movie_poster": ""},
            make_meta(),
        ])
        with patch("api.routes.search.search_stream", return_value=stream):
            response = client.post("/recommend", json={"query": "something"})

        events = parse_sse(response.text)
        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 3
        assert [e["title"] for e in result_events] == ["A", "B", "C"]

    def test_no_results_done_event_has_message(self, client):
        stream = iter([make_meta(usage={
            "input_tokens": 50, "output_tokens": 20,
            "haiku_input_tokens": 50, "haiku_output_tokens": 20,
            "opus_input_tokens": 0, "opus_output_tokens": 0,
            "rounds": 1, "tools_called": [],
        })])
        with patch("api.routes.search.search_stream", return_value=stream):
            response = client.post("/recommend", json={"query": "obscure query"})

        events = parse_sse(response.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events[0]["result_count"] == 0
        assert done_events[0]["message"] == "No relevant matches found."

    def test_claude_error_emits_error_event(self, client):
        stream = iter([make_meta(usage=None, error="rate limit exceeded")])
        with patch("api.routes.search.search_stream", return_value=stream):
            response = client.post("/recommend", json={"query": "any query"})

        events = parse_sse(response.text)
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1

    def test_result_events_include_type_field(self, client):
        stream = iter([
            {"title": "The Matrix", "explanation": "Reality is a simulation", "movie_poster": ""},
            make_meta(),
        ])
        with patch("api.routes.search.search_stream", return_value=stream):
            response = client.post("/recommend", json={"query": "simulation"})

        events = parse_sse(response.text)
        assert all("type" in e for e in events)

    def test_empty_query_returns_422(self, client):
        response = client.post("/recommend", json={"query": ""})
        assert response.status_code == 422

    def test_query_over_500_chars_returns_422(self, client):
        response = client.post("/recommend", json={"query": "x" * 501})
        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/recommend", json={})
        assert response.status_code == 422


class TestLogoProxy:
    def test_proxies_image_from_allowed_cdn(self, client):
        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG\r\n"
        mock_resp.headers = {"Content-Type": "image/png"}
        with patch("api.routes.streaming.requests.get", return_value=mock_resp):
            response = client.get("/logo-proxy?url=https%3A%2F%2Fcdn.watchmode.com%2Fprovider_logos%2Fnetflix.png")
        assert response.status_code == 200
        assert response.content == b"\x89PNG\r\n"
        assert response.headers["content-type"] == "image/png"
        assert "max-age=2592000" in response.headers["cache-control"]

    def test_serves_from_cache_on_second_request(self, client):
        # Clear the image cache so we start fresh
        import api.routes.streaming as streaming_mod
        with streaming_mod._logo_image_cache_lock:
            streaming_mod._logo_image_cache.clear()

        mock_resp = MagicMock()
        mock_resp.content = b"JPEG"
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        url = "https://cdn.watchmode.com/provider_logos/hulu.jpg"
        encoded = "https%3A%2F%2Fcdn.watchmode.com%2Fprovider_logos%2Fhulu.jpg"
        with patch("api.routes.streaming.requests.get", return_value=mock_resp) as mock_get:
            client.get(f"/logo-proxy?url={encoded}")
            client.get(f"/logo-proxy?url={encoded}")
        # CDN should only be contacted once; second request served from cache
        assert mock_get.call_count == 1

    def test_rejects_disallowed_host(self, client):
        response = client.get("/logo-proxy?url=https%3A%2F%2Fevil.example.com%2Flogo.png")
        assert response.status_code == 400

    def test_returns_502_when_cdn_fetch_fails(self, client):
        with patch("api.routes.streaming.requests.get", side_effect=Exception("timeout")):
            response = client.get("/logo-proxy?url=https%3A%2F%2Fcdn.watchmode.com%2Fprovider_logos%2Ffail.png")
        assert response.status_code == 502


class TestStreamingEndpoint:
    def test_returns_providers_via_watchmode_when_key_set(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", "fake-key"), \
             patch("api.routes.streaming.watchmode.search_title", return_value=12345), \
             patch("api.routes.streaming.watchmode.fetch_providers", return_value=[
                 {"name": "Paramount+", "type": "sub", "logo": "https://cdn.watchmode.com/provider_logos/paramountplus_100px.jpg"}
             ]), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.get("/streaming?title=Million+Dollar+Baby&year=2004")

        assert response.status_code == 200
        data = response.json()
        assert data["providers"] == [
            {"name": "Paramount+", "type": "sub", "logo": "/logo-proxy?url=https%3A%2F%2Fcdn.watchmode.com%2Fprovider_logos%2Fparamountplus_100px.jpg"}
        ]
        assert "certification" in data

    def test_returns_rent_providers_when_no_subscription(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", "fake-key"), \
             patch("api.routes.streaming.watchmode.search_title", return_value=12345), \
             patch("api.routes.streaming.watchmode.fetch_providers", return_value=[
                 {"name": "Amazon", "type": "rent", "logo": None},
                 {"name": "VUDU", "type": "buy", "logo": None},
             ]), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.get("/streaming?title=The+Royal+Tenenbaums&year=2001")

        assert response.status_code == 200
        providers = response.json()["providers"]
        assert len(providers) == 2
        assert all(p["type"] in ("rent", "buy") for p in providers)

    def test_returns_empty_when_watchmode_finds_no_title(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", "fake-key"), \
             patch("api.routes.streaming.watchmode.search_title", return_value=None), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.get("/streaming?title=Unknown+Movie+XYZ")

        assert response.status_code == 200
        data = response.json()
        assert data["providers"] == []
        assert data["certification"] == ""

    def test_falls_back_to_tmdb_when_no_watchmode_key(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", None), \
             patch("api.routes.streaming.search_movie_by_title", return_value=238), \
             patch("api.routes.streaming.fetch_watch_providers", return_value=[
                 {"name": "Netflix", "type": "sub", "logo": "https://image.tmdb.org/t/p/w45/netflix.jpg"}
             ]), \
             patch("api.routes.streaming.fetch_certification", return_value="R"):
            response = client.get("/streaming?title=The+Godfather&year=1972")

        assert response.status_code == 200
        data = response.json()
        assert data["providers"] == [
            {"name": "Netflix", "type": "sub", "logo": "/logo-proxy?url=https%3A%2F%2Fimage.tmdb.org%2Ft%2Fp%2Fw45%2Fnetflix.jpg"}
        ]
        assert data["certification"] == "R"

    def test_tmdb_fallback_returns_empty_when_movie_not_found(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", None), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.get("/streaming?title=Unknown+Movie+XYZ")

        assert response.status_code == 200
        data = response.json()
        assert data["providers"] == []
        assert data["certification"] == ""

    def test_year_param_is_optional(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", None), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.get("/streaming?title=Some+Movie")

        assert response.status_code == 200
        assert "certification" in response.json()

    def test_missing_title_returns_422(self, client):
        response = client.get("/streaming")
        assert response.status_code == 422


class TestStreamingBatchEndpoint:
    def test_returns_providers_for_multiple_titles(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", "fake-key"), \
             patch("api.routes.streaming.watchmode.search_title", side_effect=[111, 222]), \
             patch("api.routes.streaming.watchmode.fetch_providers", side_effect=[
                 [{"name": "Netflix", "type": "sub", "logo": None}],
                 [{"name": "Hulu", "type": "sub", "logo": None}],
             ]), \
             patch("api.routes.streaming.search_movie_by_title", side_effect=[550, 603]), \
             patch("api.routes.streaming.fetch_certification", side_effect=["R", "R"]):
            response = client.post("/streaming/batch", json={
                "titles": [
                    {"title": "Inception", "year": "2010"},
                    {"title": "The Matrix", "year": "1999"},
                ]
            })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "Inception"
        assert data["results"][0]["providers"] == [{"name": "Netflix", "type": "sub", "logo": None}]
        assert data["results"][0]["certification"] == "R"
        assert data["results"][1]["title"] == "The Matrix"
        assert data["results"][1]["providers"] == [{"name": "Hulu", "type": "sub", "logo": None}]
        assert data["results"][1]["certification"] == "R"

    def test_exceeding_ten_titles_returns_422(self, client):
        response = client.post("/streaming/batch", json={
            "titles": [{"title": f"Movie {i}", "year": "2020"} for i in range(11)]
        })
        assert response.status_code == 422

    def test_empty_titles_returns_empty_results(self, client):
        response = client.post("/streaming/batch", json={"titles": []})
        assert response.status_code == 200
        assert response.json() == {"results": []}

    def test_watchmode_not_found_returns_empty_providers(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", "fake-key"), \
             patch("api.routes.streaming.watchmode.search_title", return_value=None), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.post("/streaming/batch", json={
                "titles": [{"title": "Unknown Movie XYZ", "year": ""}]
            })
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["providers"] == []
        assert result["certification"] == ""

    def test_falls_back_to_tmdb_when_no_watchmode_key(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", None), \
             patch("api.routes.streaming.search_movie_by_title", return_value=238), \
             patch("api.routes.streaming.fetch_watch_providers", return_value=[
                 {"name": "Netflix", "type": "sub", "logo": None}
             ]), \
             patch("api.routes.streaming.fetch_certification", return_value="R"):
            response = client.post("/streaming/batch", json={
                "titles": [{"title": "The Godfather", "year": "1972"}]
            })
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["providers"] == [{"name": "Netflix", "type": "sub", "logo": None}]
        assert result["certification"] == "R"

    def test_missing_titles_field_returns_422(self, client):
        response = client.post("/streaming/batch", json={})
        assert response.status_code == 422

    def test_year_is_optional_in_batch(self, client):
        with patch("api.routes.streaming.WATCHMODE_API_KEY", "fake-key"), \
             patch("api.routes.streaming.watchmode.search_title", return_value=999), \
             patch("api.routes.streaming.watchmode.fetch_providers", return_value=[]), \
             patch("api.routes.streaming.search_movie_by_title", return_value=None):
            response = client.post("/streaming/batch", json={
                "titles": [{"title": "Some Movie"}]
            })
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["year"] == ""
        assert "certification" in result


class TestHints:
    def test_returns_list_of_strings(self, client):
        response = client.get("/hints.json")
        assert response.status_code == 200
        hints = response.json()
        assert isinstance(hints, list)
        assert len(hints) > 0
        assert all(isinstance(h, str) for h in hints)


class TestAdminLogs:
    def test_location_resolved_from_client_ip(self, client):
        from api.auth import require_admin
        app.dependency_overrides[require_admin] = lambda: None
        log_entry = json.dumps({
            "timestamp": "2026-04-15T12:00:00+00:00",
            "query": "sci-fi thriller",
            "client_ip": "1.2.3.4",
            "status": "ok",
            "result_count": 3,
            "total_ms": 800,
        })
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".log", delete=False, dir="/tmp"
            ) as f:
                f.write(log_entry + "\n")
                tmp_path = f.name

            with patch("api.routes.admin.LOG_DIR", os.path.dirname(tmp_path)), \
                 patch("api.routes.admin._resolve_location", return_value=("94103", "San Francisco, California")) as mock_resolve:
                # Patch the log filename to match the temp file name.
                with patch("os.path.join", side_effect=lambda *a: tmp_path if a[-1] == "search.log" else os.path.join(*a)):
                    response = client.get("/admin/logs")
        finally:
            os.unlink(tmp_path)
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["location"] == "94103"
        assert data["entries"][0]["location_detail"] == "San Francisco, California"
        mock_resolve.assert_called_once_with("1.2.3.4")

    def test_missing_client_ip_omits_location(self, client):
        from api.auth import require_admin
        app.dependency_overrides[require_admin] = lambda: None
        log_entry = json.dumps({
            "timestamp": "2026-04-15T12:00:00+00:00",
            "query": "drama",
            "status": "ok",
            "result_count": 2,
            "total_ms": 500,
        })
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".log", delete=False, dir="/tmp"
            ) as f:
                f.write(log_entry + "\n")
                tmp_path = f.name

            with patch("api.routes.admin.LOG_DIR", os.path.dirname(tmp_path)), \
                 patch("api.routes.admin._resolve_location") as mock_resolve:
                with patch("os.path.join", side_effect=lambda *a: tmp_path if a[-1] == "search.log" else os.path.join(*a)):
                    response = client.get("/admin/logs")
        finally:
            os.unlink(tmp_path)
            app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_resolve.assert_not_called()


class TestResolveLocation:
    def test_returns_postal_with_detail_when_available(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"postal": "94103", "city": "San Francisco", "region": "California", "country": "US"}
        with patch("api.routes.admin.httpx.get", return_value=mock_resp):
            primary, detail = _resolve_location("1.2.3.4")
        assert primary == "94103"
        assert detail == "San Francisco, California"

    def test_postal_detail_omits_missing_city(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"postal": "94103", "city": "", "region": "California", "country": "US"}
        with patch("api.routes.admin.httpx.get", return_value=mock_resp):
            primary, detail = _resolve_location("1.2.3.4")
        assert primary == "94103"
        assert detail == "California"

    def test_falls_back_to_city_when_no_postal(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"postal": "", "city": "San Francisco", "region": "California", "country": "US"}
        with patch("api.routes.admin.httpx.get", return_value=mock_resp):
            primary, detail = _resolve_location("1.2.3.4")
        assert primary == "San Francisco"
        assert detail is None  # no extra line for city-level resolution

    def test_falls_back_to_region_when_no_city(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"postal": "", "city": "", "region": "California", "country": "US"}
        with patch("api.routes.admin.httpx.get", return_value=mock_resp):
            primary, detail = _resolve_location("1.2.3.4")
        assert primary == "California"
        assert detail is None

    def test_falls_back_to_country_as_last_resort(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"postal": "", "city": "", "region": "", "country": "US"}
        with patch("api.routes.admin.httpx.get", return_value=mock_resp):
            primary, detail = _resolve_location("1.2.3.4")
        assert primary == "US"
        assert detail is None

    def test_returns_none_tuple_for_private_ip(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        with patch("api.routes.admin.httpx.get") as mock_get:
            result = _resolve_location("127.0.0.1")
        assert result == (None, None)
        mock_get.assert_not_called()

    def test_returns_none_tuple_on_network_failure(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        with patch("api.routes.admin.httpx.get", side_effect=Exception("timeout")):
            assert _resolve_location("1.2.3.4") == (None, None)
        # Transient failures must not be cached so the next call can retry.
        assert "1.2.3.4" not in _ip_location_cache

    def test_caches_result_on_repeated_calls(self):
        from api.routes.admin import _resolve_location, _ip_location_cache
        _ip_location_cache.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"postal": "10001", "city": "New York", "region": "New York", "country": "US"}
        with patch("api.routes.admin.httpx.get", return_value=mock_resp) as mock_get:
            _resolve_location("5.6.7.8")
            _resolve_location("5.6.7.8")
        mock_get.assert_called_once()  # second call hit cache


class TestAdminStatus:
    def test_returns_expected_shape(self, client):
        from api.auth import require_admin
        from api.app import app
        app.dependency_overrides[require_admin] = lambda: None
        try:
            with patch("api.routes.admin.vector_count", return_value=42):
                response = client.get("/admin/status")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["chroma_count"] == 42
        assert "movie_count" in data
        assert "initializing" in data

    def test_unauthenticated_returns_401(self, client):
        response = client.get("/admin/status")
        assert response.status_code == 401
