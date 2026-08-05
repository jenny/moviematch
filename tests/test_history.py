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
        # Safety net only: __mock__ no longer POSTs anything (see test_mock_search.py),
        # but this keeps an unexpected background call from erroring the page under test.
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
    """Opening the first result card appends &detail=<Title (Year)> to the URL.

    The param is the "Title (Year)" key (encodeURIComponent leaves parens literal, space→%20),
    not the numeric index — so the URL is a stable, shareable pointer to the film.
    """
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    expect(page).to_have_url(re.compile(r"\?q=__mock__&detail=Inception%20\(2010\)"))


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


def test_carousel_updates_url_without_adding_history(page: Page, base_url: str):
    """Carousel navigation updates the URL to name the open card, but adds NO history entry.

    The URL sync uses replaceState, so the address bar stays copy-shareable at every card
    while back still closes the overlay in a single step (not stepping back through cards).
    """
    _run_mock(page, base_url)
    page.locator(".card").first.click()
    page.wait_for_selector("#overlayBackdrop.open")
    expect(page).to_have_url(re.compile(r"detail=Inception%20\(2010\)"))
    # Carousel forward: URL now names card 1 (Parasite) — the replaceState sync.
    page.keyboard.press("ArrowRight")
    expect(page).to_have_url(re.compile(r"detail=Parasite%20\(2019\)"))
    # But no new history entry was pushed: one back closes the overlay to the search state,
    # rather than stepping back to the previous card.
    page.go_back()
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))
    expect(page).to_have_url(re.compile(r"\?q=__mock__$"))
    expect(page.locator(".card")).to_have_count(5)


# ── Overlay scroll reset ──────────────────────────────────────────────────────


def _open_and_scroll_to_bottom(page: Page, card_index: int):
    """Open the overlay on a card and scroll its .overlay-scroll to the bottom.

    Returns the .overlay-scroll locator. Asserts the content actually overflows
    (scrollTop > 0) so the test fails loudly if the panel ever stops scrolling,
    rather than passing vacuously.
    """
    page.locator(".card").nth(card_index).click()
    page.wait_for_selector("#overlayBackdrop.open")
    scroll = page.locator(".overlay-scroll")
    scroll.evaluate("el => el.scrollTop = el.scrollHeight")
    assert scroll.evaluate("el => el.scrollTop") > 0, "overlay content did not overflow"
    return scroll


def test_carousel_resets_overlay_scroll(page: Page, base_url: str):
    """Navigating to another movie via the carousel resets the scroll to top.

    Relies on cards 0 and 1 (Inception, Parasite) carrying long overviews so the
    overlay panel overflows; the helper asserts scrollTop > 0 to catch regressions.
    """
    _run_mock(page, base_url)
    scroll = _open_and_scroll_to_bottom(page, 0)
    page.keyboard.press("ArrowRight")
    assert scroll.evaluate("el => el.scrollTop") == 0


def test_reopen_different_movie_resets_overlay_scroll(page: Page, base_url: str):
    """Close a scrolled overlay, open a different movie → it opens at the top.

    Regression for the display:none case: the reset in renderOverlay() is a
    no-op while the backdrop is hidden, so openOverlay() must reset scrollTop
    again after .open makes the panel visible.
    """
    _run_mock(page, base_url)
    scroll = _open_and_scroll_to_bottom(page, 0)
    page.keyboard.press("Escape")
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))
    page.locator(".card").nth(1).click()
    page.wait_for_selector("#overlayBackdrop.open")
    assert scroll.evaluate("el => el.scrollTop") == 0


# ── Deep-link support ─────────────────────────────────────────────────────────


def test_deep_link_auto_runs_search(page: Page, base_url: str):
    """Loading the app with ?q=__mock__ auto-runs the mock search."""
    page.goto(f"{base_url}/?q=__mock__")
    page.locator(".card").nth(4).wait_for()
    assert page.locator(".card").count() == 5


def test_shared_detail_link_auto_opens_card(page: Page, base_url: str):
    """A shared ?q=…&detail=<Title (Year)> link auto-opens that movie's overlay.

    This is the payoff of the share button: the recipient lands with the film's card open.
    Matched by "Title (Year)" key, so it opens the right film regardless of result order.
    """
    page.goto(f"{base_url}/?q=__mock__&detail=Parasite%20(2019)")
    page.wait_for_selector("#overlayBackdrop.open")
    expect(page.locator(".overlay-title")).to_contain_text("Parasite")
    # Back closes the overlay and leaves the full result feed behind it.
    page.go_back()
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))
    expect(page.locator(".card")).to_have_count(5)


def test_carousel_works_from_deep_linked_card(page: Page, base_url: str):
    """Opening via a shared link is indistinguishable from a click — the carousel still steps.

    Guards the design: the deep link only computes the *starting* index; overlayIndex stays
    the source of truth, so arrow navigation moves to the neighbouring films as normal.
    """
    page.goto(f"{base_url}/?q=__mock__&detail=Parasite%20(2019)")  # card index 1
    page.wait_for_selector("#overlayBackdrop.open")
    expect(page.locator(".overlay-title")).to_contain_text("Parasite")
    page.keyboard.press("ArrowRight")  # → index 2
    expect(page.locator(".overlay-title")).to_contain_text("Everything Everywhere All at Once")
    expect(page).to_have_url(re.compile(r"detail=Everything"))
    page.keyboard.press("ArrowLeft")  # ← back to index 1
    expect(page.locator(".overlay-title")).to_contain_text("Parasite")


def test_unknown_detail_link_lands_on_results(page: Page, base_url: str):
    """A ?detail= that matches no result degrades gracefully: results show, no overlay opens."""
    page.goto(f"{base_url}/?q=__mock__&detail=Nonexistent%20(1999)")
    page.locator(".card").nth(4).wait_for()
    expect(page.locator("#overlayBackdrop")).not_to_have_class(re.compile(r"open"))
    assert page.locator(".card").count() == 5


# ── Clear-query button ────────────────────────────────────────────────────────


def test_clear_button_hidden_when_empty(page: Page, base_url: str):
    """The clear × is hidden on load while the input is empty."""
    page.goto(base_url)
    expect(page.locator("#clearBtn")).to_be_hidden()


def test_clear_button_appears_with_text(page: Page, base_url: str):
    """Typing into the input reveals the clear ×."""
    page.goto(base_url)
    page.fill("#queryInput", "space movies")
    expect(page.locator("#clearBtn")).to_be_visible()


def test_clear_button_empties_and_refocuses_input(page: Page, base_url: str):
    """Clicking the clear × empties the input, hides itself, and refocuses."""
    page.goto(base_url)
    page.fill("#queryInput", "space movies")
    page.click("#clearBtn")
    expect(page.locator("#queryInput")).to_have_value("")
    expect(page.locator("#clearBtn")).to_be_hidden()
    expect(page.locator("#queryInput")).to_be_focused()


def test_clear_button_hidden_after_deep_link_back(page: Page, base_url: str):
    """Back to the blank page (empty state) hides the clear × again."""
    _run_mock(page, base_url)
    expect(page.locator("#clearBtn")).to_be_visible()  # __mock__ populated the input
    page.go_back()
    expect(page.locator("#queryInput")).to_have_value("")
    expect(page.locator("#clearBtn")).to_be_hidden()
