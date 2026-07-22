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
        (3, "Apple TV"),                                 # rent/buy only → promoted chips
        (4, "Not available for streaming in"),           # empty list → region notice
    ],
)
def test_streaming_render_branches(mock_page, index, expected):
    """Each fixture drives a different paintProviders() branch, rendered from inline data."""
    page, _ = mock_page
    page.query_selector_all(".card")[index].click()
    page.wait_for_timeout(250)
    assert expected in page.inner_text("#streamingProviders")


@pytest.mark.parametrize(
    "index, expected_label",
    [
        (0, "Streaming"),                    # subscription available
        (2, "Streaming"),                    # free (ad-supported) counts as streaming
        (3, "Available to rent or buy"),     # rent + buy, nothing streamable
        (4, "Streaming"),                    # nothing at all — heading stays put
    ],
)
def test_section_heading_reflects_availability(mock_page, index, expected_label):
    """The heading itself carries the rent/buy caveat, so it must track the branch taken.

    It's rewritten in place, so a stale label from a previously-opened overlay would
    mislabel the next film — hence checking the streamable cases too, not just fixture 3.
    Compared case-insensitively: .overlay-label is text-transform: uppercase, so inner_text
    reports the rendered casing rather than the string we set.
    """
    page, _ = mock_page
    page.query_selector_all(".card")[index].click()
    page.wait_for_timeout(250)
    assert page.inner_text("#streamingLabel").strip().casefold() == expected_label.casefold()


def test_heading_resets_between_overlays(mock_page):
    """Open the rent-only film, then a streamable one: the heading must not stick."""
    page, _ = mock_page
    page.query_selector_all(".card")[3].click()
    page.wait_for_timeout(250)
    assert "rent" in page.inner_text("#streamingLabel").casefold()
    page.keyboard.press("Escape")
    page.wait_for_timeout(150)
    page.query_selector_all(".card")[0].click()
    page.wait_for_timeout(250)
    assert page.inner_text("#streamingLabel").strip().casefold() == "streaming"


def test_provider_chips_are_deep_links(mock_page):
    """Subscription chips link out to the title on that provider, safely."""
    page, _ = mock_page
    page.query_selector_all(".card")[0].click()
    page.wait_for_timeout(250)
    links = page.query_selector_all("#streamingProviders a.overlay-provider")
    assert len(links) == 2
    for link in links:
        assert link.get_attribute("href", ).startswith("https://")
        assert link.get_attribute("target") == "_blank"
        # noreferrer matters as much as noopener — these are third-party destinations.
        assert link.get_attribute("rel") == "noopener noreferrer"


def test_rent_chip_shows_price(mock_page):
    """Fixture 3 has a priced rent entry and an unpriced buy entry — both must render."""
    page, _ = mock_page
    page.query_selector_all(".card")[3].click()
    page.wait_for_timeout(250)
    text = page.inner_text("#streamingProviders")
    assert "$3.99" in text                       # Apple TV, priced
    assert "Prime Video" in text                 # unpriced chip still renders
    prices = page.query_selector_all("#streamingProviders .overlay-provider-price")
    assert len(prices) == 1, "only the priced entry should show an amount"


def test_chip_hover_title_distinguishes_rent_from_buy(mock_page):
    """The chips look identical, so the hover title is the only rent/buy signal."""
    page, _ = mock_page
    page.query_selector_all(".card")[3].click()
    page.wait_for_timeout(250)
    titles = [a.get_attribute("title")
              for a in page.query_selector_all("#streamingProviders a.overlay-provider")]
    assert titles == ["Rent on Apple TV", "Buy on Prime Video"]


def test_streamable_chip_hover_title_says_watch(mock_page):
    page, _ = mock_page
    page.query_selector_all(".card")[0].click()
    page.wait_for_timeout(250)
    titles = [a.get_attribute("title")
              for a in page.query_selector_all("#streamingProviders a.overlay-provider")]
    assert titles == ["Watch on Netflix", "Watch on Max"]


def test_reviews_render_from_inline_scores(mock_page):
    """Fixture 4 carries a partial score set (no Metacritic) — chips still render."""
    page, _ = mock_page
    page.query_selector_all(".card")[4].click()
    page.wait_for_timeout(250)
    assert page.query_selector_all("#ratingsScores .overlay-score")
