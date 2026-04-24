"""Browser back-button / History API tests.

Uses Playwright with a minimal static HTTP server so no running backend or
real API keys are required. All searches use the __mock__ query, which
bypasses /recommend entirely and renders five hardcoded results through the
same appendResult() path as live searches.

Setup (once):
    pip install pytest-playwright
    playwright install chromium
Run:
    pytest tests/test_history.py
"""

import http.server
import os
import re
import threading

import pytest
from playwright.sync_api import Page, expect

ROOT = os.path.dirname(os.path.dirname(__file__))
MOCK = "__mock__"


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the project root statically; redirects / → /app.html; swallows POSTs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        # Strip query string before path check so /?q=... also serves app.html
        if self.path.split("?")[0] in ("/", ""):
            self.path = "/app.html"
        super().do_GET()

    def do_POST(self):
        # Satisfy background /streaming/batch calls so they don't error.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass  # suppress server output during tests


@pytest.fixture(scope="session")
def base_url():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _run_mock(page: Page, base_url: str) -> None:
    """Navigate to the app, submit __mock__, and wait for ALL five result cards.

    Waiting for the 5th card (rather than the 1st) ensures runMockSearch() has
    finished appending results and called history.replaceState with the full
    results array before any test interaction starts.
    """
    page.goto(base_url)
    page.fill("#queryInput", MOCK)
    page.click("#submitBtn")
    page.locator(".card").nth(4).wait_for()  # 0-indexed; 4 = 5th card


# ── Search-level navigation ───────────────────────────────────────────────────


def test_search_updates_url(page: Page, base_url: str):
    """Submitting a search updates the URL to ?q=<query>."""
    _run_mock(page, base_url)
    expect(page).to_have_url(re.compile(r"\?q=__mock__"))


def test_back_from_search_clears_results(page: Page, base_url: str):
    """Back from the first search returns to a blank page with no result cards."""
    _run_mock(page, base_url)
    page.go_back()
    expect(page.locator(".card")).to_have_count(0)


def test_back_between_searches_restores_results(page: Page, base_url: str):
    """Back after a second search restores the previous search results.

    Uses a direct second mock search rather than an overlay pivot so the test
    doesn't depend on the pivot query producing results on the test server.
    """
    _run_mock(page, base_url)
    # Submit a second search directly; pushState fires synchronously before fetch
    page.fill("#queryInput", "sci-fi movies")
    page.click("#submitBtn")
    page.wait_for_url(re.compile(r"\?q=sci-fi"))
    page.go_back()
    page.wait_for_url(re.compile(r"\?q=__mock__"))
    expect(page.locator(".card")).to_have_count(5)


# ── Overlay-level navigation ──────────────────────────────────────────────────


def test_overlay_opens_with_detail_url(page: Page, base_url: str):
    """Opening the first result card appends &detail=0 to the URL."""
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    expect(page).to_have_url(re.compile(r"\?q=__mock__&detail=0"))


def test_esc_closes_overlay(page: Page, base_url: str):
    """ESC dismisses the overlay and reverts the URL to ?q=<query>."""
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    page.keyboard.press("Escape")
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))
    expect(page).to_have_url(re.compile(r"\?q=__mock__$"))


def test_back_closes_overlay_preserves_results(page: Page, base_url: str):
    """Back while the overlay is open closes it; the result cards remain."""
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    page.go_back()
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))
    expect(page.locator(".card")).to_have_count(5)
    expect(page).to_have_url(re.compile(r"\?q=__mock__$"))


def test_pivot_from_overlay_back_reopens_overlay(page: Page, base_url: str):
    """Back after pivoting from an open overlay reopens that overlay.

    Stack after pivot:
      [{empty}, {search:__mock__}, {overlay:__mock__,0}, {search:<pivot>}]
    Back consumes the pivot entry and lands on the overlay entry.
    """
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    page.locator(".overlay-title-pivot").click()
    # pushState fires synchronously before the fetch; wait for URL change rather
    # than result cards (the pivot query won't produce cards on the test server)
    page.wait_for_url(re.compile(r"\?q=More"))
    page.go_back()
    page.wait_for_selector("#overlayBackdrop.open")


# ── Carousel navigation ───────────────────────────────────────────────────────


def test_carousel_does_not_add_history_entries(page: Page, base_url: str):
    """Carousel navigation inside the overlay does not change the URL or add history entries."""
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    initial_url = page.url
    page.keyboard.press("ArrowRight")
    assert page.url == initial_url  # URL must not change
    # Back should go directly to the search state, not a second overlay entry
    page.go_back()
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))


# ── Deep-link support ─────────────────────────────────────────────────────────


def test_deep_link_auto_runs_search(page: Page, base_url: str):
    """Loading the app with ?q=__mock__ auto-runs the mock search."""
    page.goto(f"{base_url}/?q=__mock__")
    page.locator(".card").nth(4).wait_for()
    assert page.locator(".card").count() == 5
