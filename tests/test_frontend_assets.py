"""Static checks on the inline SVG sprite and mock fixtures in app.html.

The Reviews chips render each provider glyph as `<use href="#icon-x"/>` pointing at a
`<symbol>` defined in the sprite near the top of app.html. If the two drift apart the
glyph renders *invisibly* — no console error, no layout shift, just a missing logo —
so it is easy to ship broken. These tests parse app.html directly (no browser needed)
and assert the sprite and the SCORE_PROVIDERS config stay in sync.

The mock-fixture tests guard the __mock__ offline contract. They are static so they run
without Playwright; tests/test_mock_search.py covers the same contract behaviourally.
"""

import re
from pathlib import Path

import pytest

APP_HTML = Path(__file__).resolve().parent.parent / "app.html"


@pytest.fixture(scope="module")
def html() -> str:
    return APP_HTML.read_text()


def _symbol_viewboxes(html: str) -> dict[str, tuple[float, float]]:
    """Map every `<symbol id=...>` in the sprite to its viewBox (width, height)."""
    out: dict[str, tuple[float, float]] = {}
    for sym_id, view_box in re.findall(r'<symbol\s+id="([^"]+)"\s+viewBox="([^"]+)"', html):
        parts = [float(v) for v in view_box.split()]
        out[sym_id] = (parts[2], parts[3])
    return out


def _score_providers(html: str) -> dict[str, dict]:
    """Parse the SCORE_PROVIDERS JS object literal into {provider: {symbol, width}}."""
    block = re.search(r"const SCORE_PROVIDERS = \{(.*?)\n\s*\};", html, re.S)
    assert block, "SCORE_PROVIDERS object literal not found in app.html"
    out: dict[str, dict] = {}
    for key, symbol, width in re.findall(
        r'(\w+):\s*\{[^}]*symbol:\s*"([^"]+)",\s*width:\s*(\d+)', block.group(1)
    ):
        out[key] = {"symbol": symbol, "width": int(width)}
    return out


def test_score_providers_parsed(html):
    """Guard the parser itself — a silent regex miss would void every test below."""
    providers = _score_providers(html)
    assert set(providers) == {"rt", "imdb", "metacritic", "tmdb"}


def test_every_score_glyph_resolves_to_a_symbol(html):
    """Each provider's `symbol` id must exist, or the chip renders an empty box."""
    symbols = _symbol_viewboxes(html)
    for provider, meta in _score_providers(html).items():
        assert meta["symbol"] in symbols, (
            f"{provider} references <use href='#{meta['symbol']}'> but no such <symbol> exists"
        )


def test_glyph_width_matches_viewbox_aspect(html):
    """Height is a uniform 20px in CSS, so width must track the viewBox aspect ratio.

    Wordmarks (IMDb/TMDB, 64x32) are twice as wide as tall; the square icons (RT 24x24,
    Metacritic 40x40) are 1:1. A mismatch renders the glyph stretched or letterboxed.
    """
    symbols = _symbol_viewboxes(html)
    for provider, meta in _score_providers(html).items():
        vb_w, vb_h = symbols[meta["symbol"]]
        expected = round(20 * vb_w / vb_h)
        assert meta["width"] == expected, (
            f"{provider}: width={meta['width']} but viewBox {vb_w}x{vb_h} at 20px tall "
            f"needs width={expected}"
        )


def _mock_results_block(html: str) -> str:
    """Return the source of the MOCK_RESULTS array literal."""
    block = re.search(r"const MOCK_RESULTS = \[(.*?)\n\s*\];", html, re.S)
    assert block, "MOCK_RESULTS array literal not found in app.html"
    return block.group(1)


def _strip_line_comments(src: str) -> str:
    """Drop `//` comments so prose mentioning a function isn't mistaken for a call."""
    return "\n".join(line.split("//")[0] for line in src.splitlines())


def test_every_mock_fixture_has_inline_providers_and_scores(html):
    """Both arrays must be present on every fixture or __mock__ hits the network.

    renderStreamingProviders()/renderScores() only short-circuit when the field is an
    array; a fixture missing one silently falls through to a live /streaming (Watchmode)
    or /ratings (OMDb) fetch, which defeats the point of an offline mock.
    """
    block = _mock_results_block(html)
    titles = re.findall(r"^\s{8}title:", block, re.M)
    assert len(titles) == 5, f"expected 5 mock fixtures, found {len(titles)}"
    for field in ("providers", "scores"):
        found = re.findall(rf"^\s+{field}:\s*\[", block, re.M)
        assert len(found) == len(titles), (
            f"{len(found)} fixtures define `{field}` but there are {len(titles)} fixtures — "
            f"one without it falls through to a live fetch"
        )


def test_mock_fixtures_cover_every_streaming_render_branch(html):
    """The fixtures are the only way to exercise paintProviders' three branches locally."""
    block = _mock_results_block(html)
    types = set(re.findall(r'type:\s*"(\w+)"', block))
    assert {"sub", "free", "rent", "buy"} <= types, f"missing provider types: {types}"
    assert re.search(r"providers:\s*\[\]", block), (
        "no fixture with an empty providers list — the "
        '"Not available for streaming in <region>" branch is uncovered'
    )


def test_run_mock_search_makes_no_network_call(html):
    """runMockSearch() must not call the batch prefetchers — they POST to the backend."""
    fn = re.search(r"async function runMockSearch\(\) \{(.*?)\n    \}", html, re.S)
    assert fn, "runMockSearch() not found in app.html"
    body = _strip_line_comments(fn.group(1))
    for forbidden in ("batchPrefetchStreamingProviders", "batchPrefetchScores", "fetch("):
        assert forbidden not in body, (
            f"runMockSearch() calls {forbidden} — __mock__ must not touch the network"
        )


def test_metacritic_uses_official_monogram(html):
    """Pin the Metacritic mark to the real brand icon, not a font-rendered letter.

    Its stepped 'm' cannot be reproduced with a text glyph, so the symbol carries the
    official vector paths: a gold ring (#FFBD3F) over a dark field. A regression to a
    <text> element would look nothing like the brand.
    """
    symbol = re.search(r'<symbol id="icon-mc".*?</symbol>', html, re.S)
    assert symbol, "icon-mc symbol missing"
    markup = symbol.group(0)
    assert "#FFBD3F" in markup, "Metacritic gold ring colour missing"
    assert "<text" not in markup, "Metacritic mark regressed to a font glyph"
    assert markup.count("<path") >= 3, "expected the 3-path official monogram"
