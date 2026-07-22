"""Behavioural tests for the __mock__ offline contract.

Typing __mock__ bypasses /recommend and renders five hardcoded results. The contract is
that it makes **no backend call at all** — historically runMockSearch() still fired
batchPrefetchStreamingProviders(), which POSTs to /streaming/batch and hits the
Watchmode API, so an "offline" mock quietly burned real API quota.

Guarding that statically is not enough: the failure can also reappear via the *cache
key*. streamingCache is keyed on userCountry, which /region mutates asynchronously, so
any approach that pre-seeds the cache races the region lookup and leaks a live fetch on
a late resolve. Only driving a real browser catches that, so these tests intercept every
request the page makes.

Uses Playwright with a minimal static HTTP server — no backend or API keys required.

Setup (once):
    pip install pytest-playwright
    playwright install chromium
Run:
    pytest tests/test_mock_search.py
"""

import http.server
import os
import threading

import pytest
from playwright.sync_api import Page

ROOT = os.path.dirname(os.path.dirname(__file__))
MOCK = "__mock__"

# Any request whose path contains one of these is a backend call the mock must not make.
BACKEND_PATHS = ("/streaming", "/ratings", "/recommend")


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the project root statically; redirects / → /app.html."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", ""):
            self.path = "/app.html"
        super().do_GET()

    def log_message(self, *args):
        pass  # suppress server output during tests


@pytest.fixture(scope="session")  # session-scoped: pytest-base-url overrides a narrower scope
def base_url():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def mock_page(page: Page, base_url: str):
    """Run a __mock__ search, recording every request made after the page loads.

    Returns (page, calls). `calls` excludes the initial page load so that only
    query-triggered traffic is under test.
    """
    page.goto(base_url)
    page.wait_for_timeout(300)  # let the page-load /region call settle

    calls: list[str] = []
    page.on("request", lambda r: calls.append(r.url))

    page.fill("input[type=text]", MOCK)
    page.press("input[type=text]", "Enter")
    page.wait_for_selector(".card:nth-child(5)", timeout=5000)
    page.wait_for_timeout(800)  # a stray prefetch would fire in this window
    return page, calls


def _backend_calls(calls: list[str]) -> list[str]:
    return [c for c in calls if any(p in c for p in BACKEND_PATHS)]


def test_mock_search_makes_no_backend_call(mock_page):
    """The core regression: no /streaming, /ratings, or /recommend traffic."""
    _, calls = mock_page
    assert _backend_calls(calls) == []


def test_opening_every_overlay_makes_no_backend_call(mock_page):
    """Overlay open is the second path that can fall through to a live fetch.

    renderStreamingProviders()/renderScores() short-circuit on the inline arrays; if a
    fixture lacked one, opening its overlay would fetch even though the search did not.
    """
    page, calls = mock_page
    for i in range(5):
        page.query_selector_all(".card")[i].click()
        page.wait_for_timeout(250)
        page.keyboard.press("Escape")
        page.wait_for_timeout(150)
    assert _backend_calls(calls) == []


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, "Netflix"),                                  # sub → logo chips
        (2, "Prime Video"),                              # free → logo chips
        (3, "Available to rent on"),                     # rent/buy only → text line
        (4, "Not available for streaming in"),           # empty list → region notice
    ],
)
def test_streaming_render_branches(mock_page, index, expected):
    """Each fixture drives a different paintProviders() branch, rendered from inline data."""
    page, _ = mock_page
    page.query_selector_all(".card")[index].click()
    page.wait_for_timeout(250)
    assert expected in page.inner_text("#streamingProviders")


def test_reviews_render_from_inline_scores(mock_page):
    """Fixture 4 carries a partial score set (no Metacritic) — chips still render."""
    page, _ = mock_page
    page.query_selector_all(".card")[4].click()
    page.wait_for_timeout(250)
    assert page.query_selector_all("#ratingsScores .overlay-score")
