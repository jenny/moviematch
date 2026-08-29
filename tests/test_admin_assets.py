"""Static checks on the monthly-usage chart in admin.html (no browser needed).

The chart is inline HTML/CSS built by a template literal, so every failure mode here
is *silent*: a class the JS emits but the stylesheet never defines renders as an
unstyled inline span — zero height, no error, the chart simply isn't there. Likewise
a re-tinted bar still renders, just below the contrast floor it was validated at.
These parse admin.html directly and pin the parts that can't announce their own breakage.
"""

import re
from pathlib import Path

import pytest

ADMIN_HTML = Path(__file__).resolve().parent.parent / "admin.html"

# The validated palette, measured with tools/validate_palette.py against the panel's
# #1a1a1a surface. Neutral for past months; a blue/yellow/red severity ramp for the
# current one. Deliberately NOT green/yellow/red: green<->yellow measured 6.6 OKLab ΔE
# under deuteranopia, so "safe" and "getting close" were the same colour to a red-green
# colourblind reader; blue/yellow/red puts the worst pair at 24.4. Critical is #f4695f
# rather than a deeper red because darker reds collapse toward the neutral under
# protanopia (#e05c5c measured 5.0), hiding the current month exactly when it matters.
# Whole set: all >= 3:1 contrast, worst all-pairs 9.4 CVD / 19.0 normal.
USAGE_PALETTE = {
    "--usage-past": "#6b6b66",
    "--usage-safe": "#3987e5",
    "--usage-warning": "#fab219",
    "--usage-critical": "#f4695f",
}

# Severity cutoffs, as percentages of the monthly limit. These carried over from the
# budget meter this chart replaced.
SEVERITY_THRESHOLDS = (80, 50)


@pytest.fixture(scope="module")
def html() -> str:
    return ADMIN_HTML.read_text()


@pytest.fixture(scope="module")
def style_block(html: str) -> str:
    block = re.search(r"<style>(.*?)</style>", html, re.S)
    assert block, "no <style> block found in admin.html"
    return block.group(1)


@pytest.fixture(scope="module")
def history_fn(html: str) -> str:
    fn = re.search(r"function usageHistoryMarkup\(.*?\n    \}", html, re.S)
    assert fn, "usageHistoryMarkup() not found in admin.html"
    return fn.group(0)


@pytest.fixture(scope="module")
def severity_fn(html: str) -> str:
    fn = re.search(r"function severityClass\(.*?\n    \}", html, re.S)
    assert fn, "severityClass() not found in admin.html"
    return fn.group(0)


@pytest.fixture(scope="module")
def render_fn(html: str) -> str:
    fn = re.search(r"function renderWatchmode\(.*?\n    \}", html, re.S)
    assert fn, "renderWatchmode() not found in admin.html"
    return fn.group(0)


def test_every_emitted_usage_class_is_styled(history_fn: str, render_fn: str, style_block: str):
    """A class the markup emits but the CSS never defines renders as an unstyled
    inline span — the bar silently has no size and the chart vanishes. Both
    functions emit usage-* classes: the title lives in renderWatchmode (it heads
    the meter too), the rows in usageHistoryMarkup."""
    emitted = set(re.findall(r"usage-[a-z-]+", history_fn + render_fn))
    assert emitted, "no usage-* classes found — did the markup helper get renamed?"
    missing = [c for c in sorted(emitted) if f".{c}" not in style_block]
    assert not missing, f"emitted but unstyled: {missing}"


def test_header_carries_the_domain_label(history_fn: str, html: str):
    """With the budget meter gone, the header's "1,000 / month limit" is the only
    thing on screen naming the domain every bar is drawn against. Lose it and the
    bars become unitless lengths."""
    assert 'class="usage-history-scale"' in history_fn
    assert "/ month limit" in history_fn
    assert "budget-bar" not in html, "the replaced meter should be fully removed"


def test_empty_history_hides_the_header_too(history_fn: str, render_fn: str):
    """The head used to be emitted by renderWatchmode(), outside the empty guard, so
    an empty history stranded the title and scale label over blank space — the exact
    thing test_empty_history_renders_nothing claims to prevent. That test passed
    anyway because it only checked the guard inside the helper. Both must now live in
    the same function, with the guard first."""
    assert "usage-history-head" not in render_fn, (
        "the head must be emitted by usageHistoryMarkup, behind its empty guard"
    )
    guard = history_fn.index("if (!history.length) return")
    head = history_fn.index("usage-history-head")
    assert guard < head, "the empty guard must precede the header markup"


def test_rows_render_newest_first_without_mutating_the_input(history_fn: str):
    """Two ways this goes wrong quietly. Reversing in place mutates the caller's array,
    and the mock hands back the same fixture object on every state switch — so the order
    would flip each time you clicked between states. And reading currentMonth *after*
    the reversal would paint the severity colour and "so far" onto the oldest month."""
    assert "[...history].reverse()" in history_fn, "reverse a copy, not the input array"
    current = history_fn.index("const currentMonth")
    reverse = history_fn.index("[...history].reverse()")
    assert current < reverse, "currentMonth must be read from the chronological array"


def test_month_label_is_escaped(history_fn: str):
    """monthLabel() returns the key verbatim when it can't parse it, and keys come
    from a file on disk. watchmode.py validates them on load; this is layer two."""
    assert "escapeHtml(monthLabel(month))" in history_fn


@pytest.mark.parametrize("level", ["safe", "warning", "critical"])
def test_every_severity_level_is_styled(level: str, style_block: str, severity_fn: str):
    """severityClass() can emit any of the three. A level with no CSS rule leaves the
    current month rendering in the de-emphasis neutral — silently indistinguishable
    from a past month, and at exactly the moment ("critical") that matters most."""
    assert f'"{level}"' in severity_fn, f"severityClass() no longer emits {level}"
    assert f".usage-row.current.{level} .usage-fill" in style_block


def test_validated_palette_is_unchanged(style_block: str):
    """Guards the one property of these colors that isn't visible by looking at them."""
    for token, hex_value in USAGE_PALETTE.items():
        assert f"{token}: {hex_value}" in style_block, f"{token} drifted from {hex_value}"


def test_severity_is_not_green_and_yellow(style_block: str):
    """The specific regression this ramp exists to prevent: reverting "safe" to a
    green puts it 6.6 OKLab ΔE from the warning yellow under deuteranopia, i.e.
    indistinguishable for a red-green colourblind reader."""
    safe = USAGE_PALETTE["--usage-safe"].lstrip("#")
    r, g, b = (int(safe[i:i + 2], 16) for i in (0, 2, 4))
    assert b > g, "the safe step must stay blue-dominant, not green"


def test_severity_thresholds_match_the_replaced_meter(severity_fn: str):
    for threshold in SEVERITY_THRESHOLDS:
        assert f">= {threshold}" in severity_fn, f"{threshold}% severity cutoff missing"


def test_percent_column_is_emitted_and_styled(history_fn: str, style_block: str):
    assert 'class="usage-percent"' in history_fn
    assert ".usage-percent" in style_block


def test_small_nonzero_months_do_not_read_as_zero(html: str):
    """1 call against a 1,000 limit rounds to 0%. A month with traffic in it must
    never display as unused — the bar is already a 3px minimum for the same reason."""
    fn = re.search(r"function percentLabel\(.*?\n    \}", html, re.S)
    assert fn, "percentLabel() not found"
    assert '"<1%"' in fn.group(0)


def test_fill_keeps_min_width_floor(style_block: str):
    """One call against a 1,000 ceiling is 0.1% of the track. Without a floor the bar
    rounds to nothing and a low-usage month looks identical to an unused one."""
    fill = re.search(r"\.usage-fill \{(.*?)\}", style_block, re.S)
    assert fill, ".usage-fill rule not found"
    assert "min-width" in fill.group(1)


def test_fill_is_square_at_baseline_and_rounded_at_data_end(style_block: str):
    """Rounding both ends detaches the bar from the baseline it grows from."""
    fill = re.search(r"\.usage-fill \{(.*?)\}", style_block, re.S)
    assert "border-radius: 0 4px 4px 0" in fill.group(1)


def test_month_label_formats_in_utc(html: str):
    """The backend keys months in UTC. Parsing "2026-08-01" as a local date renders
    the *previous* month for anyone west of UTC, so the chart would silently
    disagree with the counter directly above it."""
    fn = re.search(r"function monthLabel\(.*?\n    \}", html, re.S)
    assert fn, "monthLabel() not found in admin.html"
    assert "Date.UTC(" in fn.group(0)
    assert 'timeZone: "UTC"' in fn.group(0)


def test_empty_history_renders_nothing(history_fn: str):
    """No months means no block at all, rather than a header over blank space."""
    assert "if (!history.length) return" in history_fn


def test_row_label_carries_the_year(history_fn: str, html: str):
    """12 retained months span two calendar years, so a bare month name can be
    ambiguous between the oldest and newest bars."""
    assert "monthLabel(month)" in history_fn
    fn = re.search(r"function monthLabel\(.*?\n    \}", html, re.S)
    assert fn, "monthLabel() not found"
    assert 'year: "numeric"' in fn.group(0)


def test_mock_short_circuits_before_any_fetch(html: str):
    """?mock exists so the page renders with no server and no auth. If the short
    circuit ever drifts below a fetch() the page starts hitting an endpoint that
    401s, and mock mode silently stops working outside a logged-in instance."""
    fn = re.search(r"async function load\(\).*?\n    \}", html, re.S)
    assert fn, "load() not found in admin.html"
    # Strip line comments first — the guard's own comment says the word "fetch()",
    # and matching prose instead of code would make this test meaningless.
    body = re.sub(r"//[^\n]*", "", fn.group(0))
    guard = body.index("MOCK_ACTIVE")
    first_fetch = body.index("fetch(")
    assert guard < first_fetch, "the mock guard must precede every fetch() in load()"


def test_mock_covers_every_severity_level(html: str, severity_fn: str):
    """A new severity level with no fixture can't be reviewed before it ships."""
    block = re.search(r"const MOCK_STATES = \{(.*?)\n    \};", html, re.S)
    assert block, "MOCK_STATES not found in admin.html"
    for level in re.findall(r'\? "(\w+)"|: "(\w+)";', severity_fn):
        name = level[0] or level[1]
        assert f"{name}:" in block.group(1), f"no mock fixture for severity level {name!r}"


def test_mock_covers_the_sparse_states(html: str):
    """The states real data can't produce on demand: a fresh deploy, and the
    single-month view Railway lands in right after the counter file is deleted."""
    block = re.search(r"const MOCK_STATES = \{(.*?)\n    \};", html, re.S)
    for name in ("empty", "sparse", "two", "capped"):
        assert f"{name}:" in block.group(1), f"missing {name!r} fixture"


def test_header_links_share_one_style(html: str, style_block: str):
    """"Sign out" and the mock toggle must look identical. They diverged once because
    Sign out carried an inline style attribute, which kept it out of every header
    link rule — so its hover never lightened while the toggle's did."""
    sign_out = re.search(r'<a href="/admin/logout"[^>]*>', html)
    assert sign_out, "Sign out link not found"
    assert 'class="header-link"' in sign_out.group(0), "Sign out must wear .header-link"
    assert "style=" not in sign_out.group(0), (
        "an inline style on Sign out overrides .header-link and re-splits the two"
    )

    toggle = re.search(r"function _renderMockToggle\(.*?\n    \}", html, re.S).group(0)
    assert '"header-link mock-toggle' in toggle, "toggle must wear .header-link too"

    assert ".header-link:hover" in style_block, "the shared hover rule is what keeps them in sync"


def test_mock_exit_state_overrides_its_own_hover(style_block: str):
    """`.mock-toggle.leaving` and `.header-link:hover` are both two-class selectors, so
    the later one wins. `.leaving` must sit below to stay amber while hovered — which
    is exactly why it needs its own :hover, or the amber would be the one header link
    that never reacts to the cursor."""
    leaving = style_block.index(".mock-toggle.leaving {")
    shared_hover = style_block.index(".header-link:hover")
    assert shared_hover < leaving, (
        ".mock-toggle.leaving must come after .header-link:hover or it loses on source order"
    )
    assert ".mock-toggle.leaving:hover" in style_block


def test_mock_state_lookup_ignores_prototype_keys(html: str):
    """MOCK_STATES is an object literal, so a truthiness check matched inherited keys:
    `?mock=constructor` resolved a function, state.stats was undefined, and the page
    rendered blank instead of falling back to "safe"."""
    fn = re.search(r"function runMockAdmin\(.*?\n    \}", html, re.S)
    assert fn, "runMockAdmin() not found"
    assert "Object.hasOwn(MOCK_STATES" in fn.group(0), (
        "state lookup must use Object.hasOwn, not a truthiness check"
    )


def test_mock_lives_in_two_named_fences(html: str, style_block: str):
    """Mock code ships in production, inert. It sits in two fences — behaviour in the
    script block, styles in the stylesheet — and removing the mock means cutting both.
    Each fence's comment points at the other so neither is missed."""
    start = html.index("── Mock admin (dev only) ──")
    end = html.index("── End mock ──")
    assert start < end
    behaviour = html[start:end]
    for token in ("MOCK_STATES", "runMockAdmin", "_renderMockSwitcher", "_renderMockToggle"):
        assert token in behaviour, f"{token} lives outside the mock script fence"

    assert "── Mock admin styles (dev only) ──" in style_block, "style fence missing"
    for token in (".mock-bar", ".mock-toggle.leaving"):
        assert token in style_block, f"{token} is not in the stylesheet"

    # Styles must NOT be injected from JS: that put them after the whole stylesheet
    # at runtime, making override order an artifact of script timing rather than a
    # visible decision (and it silently beat .header-link:hover).
    assert 'createElement("style")' not in behaviour, "mock styles must live in <style>"


def test_mock_toggle_renders_in_both_modes(html: str):
    """The toggle is the only way into mock mode from a live session, so its call
    must NOT sit behind the MOCK_ACTIVE guard — inside it, the entrance would only
    exist once you were already through the door."""
    start = html.index("── Mock admin (dev only) ──")
    end = html.index("── End mock ──")
    block = html[start:end]
    assert "\n    _renderMockToggle();" in block, "toggle must be called unconditionally"

    fn = re.search(r"function _renderMockToggle\(.*?\n    \}", html, re.S)
    assert fn, "_renderMockToggle() not found"
    body = fn.group(0)
    assert '"?mock"' in body, "no route into mock mode"
    # Anchored to the logout link, not an index: insertBefore(link, null) silently
    # appends, so a renamed logout href would slide the toggle past Sign out
    # rather than failing.
    assert """'a[href="/admin/logout"]'""" in body, "toggle must anchor to the logout link"
    # Leaving strips the query rather than hardcoding a path, so the round trip
    # works both at /admin.html and over file:// (where the mock is most useful).
    assert "location.pathname" in body, "exit must drop the query, not hardcode a path"


def test_values_use_tabular_figures(style_block: str):
    """The values are a right-aligned column of numbers, so digits must align."""
    value = re.search(r"\.usage-value \{(.*?)\}", style_block, re.S)
    assert value, ".usage-value rule not found"
    assert "tabular-nums" in value.group(1)
