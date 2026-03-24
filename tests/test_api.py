import json
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
    with patch("db.get_model", return_value=MagicMock()):
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


class TestAdminStatus:
    def test_returns_expected_shape(self, client):
        with patch("api.routes.admin.get_or_create_collection") as mock_coll:
            mock_coll.return_value.count.return_value = 42
            response = client.get("/admin/status")

        assert response.status_code == 200
        data = response.json()
        assert data["chroma_count"] == 42
        assert "movie_count" in data
        assert "initializing" in data
