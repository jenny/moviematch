"""Static checks on the inline SVG sprite in app.html.

The Reviews chips render each provider glyph as `<use href="#icon-x"/>` pointing at a
`<symbol>` defined in the sprite near the top of app.html. If the two drift apart the
glyph renders *invisibly* — no console error, no layout shift, just a missing logo —
so it is easy to ship broken. These tests parse app.html directly (no browser needed)
and assert the sprite and the SCORE_PROVIDERS config stay in sync.
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
