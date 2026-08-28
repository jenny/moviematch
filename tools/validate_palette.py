"""Chart-palette validator: is a set of colors actually legible and distinguishable?

Python twin of the dataviz skill's `scripts/validate_palette.js`, ported because
this machine has no JS runtime. Same formulas, same thresholds — sRGB->linear,
OKLab, the Machado-Oliveira-Fernandes (2009) CVD transforms at severity 1.0, and
WCAG relative luminance. The upstream source anticipates a Python twin and asks
that the input-boundary and conversion definitions stay in lockstep with it.

Why it lives here: the admin panel's monthly-usage chart (admin.html) picks its bar
colors against a #1a1a1a surface, where "looks fine" is a bad guide — two of the
three neutrals tried during design read perfectly well by eye and failed the 3:1
contrast floor outright. Re-run this before re-tinting those bars; the current
values are pinned by tests/test_admin_assets.py.

The six checks, and which ones apply to what:
  - Lightness band / chroma floor apply to *categorical series slots*. A deliberate
    neutral (a de-emphasis gray) is expected to fail the chroma floor — that is what
    makes it neutral. Read those two as "is this a valid series hue", not as errors.
  - CVD separation, normal-vision floor and contrast-vs-surface apply to everything.

Usage (from project root, in venv):
    python tools/validate_palette.py "#3987e5,#6b6b66" --mode dark --surface "#1a1a1a"
    python tools/validate_palette.py "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
    python tools/validate_palette.py --self-check   # verify this port's math

Exit code 0 unless a check hard-FAILs (1 on FAIL). WARN-level results — CVD in the
6-8 floor band, contrast below 3:1 — exit 0 but oblige secondary encoding (visible
direct labels, a table view, or texture).
"""
import argparse
import math
import re
import sys

# ── thresholds (keep in lockstep with validate_palette.js) ────────────────────
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}  # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0   # OKLab ΔE x100, min(protan, deutan)
NORMAL_FLOOR = 15.0                # OKLab ΔE x100, unsimulated, hard gate
CONTRAST_MIN = 3.0                 # WCAG vs surface
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

# Machado, Oliveira & Fernandes (2009) CVD transforms at severity 1.0 (linear RGB).
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}

# Input boundary — mirrors the JS. Unguarded, a bad hex propagates NaN through every
# check and the run fails OPEN (reporting a pass it never verified). The whitespace
# set is the JS/Python intersection, which also covers NBSP picked up when pasting
# hex lists out of a rendered page.
_WS = "[ \t\n\v\f\r   -     　]+"
_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _strip_ws(v: str) -> str:
    return re.sub(f"^{_WS}|{_WS}$", "", v)


def split_colors(raw: str) -> list[str]:
    return [c for c in (_strip_ws(p) for p in (raw or "").split(",")) if c]


def is_hex(v: str) -> bool:
    return bool(_HEX_RE.match(v))


# ── color conversions ─────────────────────────────────────────────────────────
def _hex2srgb(h: str) -> list[float]:
    h = _strip_ws(h).lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(h: str) -> list[float]:
    return [_s2lin(c) for c in _hex2srgb(h)]


def _rel_lum(h: str) -> float:
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_lin(rgb: list[float]) -> list[float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]


def oklch(h: str) -> tuple[float, float]:
    L, a, b = _oklab_from_lin(_lin(h))
    return L, math.hypot(a, b)


def _simulate(h: str, kind: str) -> list[float]:
    r, g, b = _lin(h)
    M = MACHADO[kind]
    return [max(0.0, min(1.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    """Euclidean distance in OKLab, x100. kind=None is unsimulated (normal) vision."""
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


# ── checks ────────────────────────────────────────────────────────────────────
def validate(palette: list[str], mode: str = "light", surface: str | None = None,
             pairs: str = "adjacent") -> tuple[list[tuple[str, str, str]], bool]:
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    report: list[tuple[str, str, str]] = []
    ok = True

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    if offband:
        ok = False
    report.append(("Lightness band", "fail" if offband else "pass",
                   f"outside band: {offband}" if offband
                   else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    if lowc:
        ok = False
    report.append(("Chroma floor", "fail" if lowc else "pass",
                   f"below floor (reads gray): {lowc}" if lowc
                   else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]
    label = "all-pairs" if pairs == "all" else "adjacent"

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99.0)
    wd = worst[0] if worst else 99.0
    cvd_state = "pass" if wd >= CVD_TARGET else "floor" if wd >= CVD_FLOOR else "fail"
    if cvd_state == "fail":
        ok = False
    report.append(("CVD separation", cvd_state,
                   f"worst {label} {worst[3]}<->{worst[2]} ΔE {wd:.1f} ({worst[1]}) "
                   f"· tritan {tri:.1f}" if worst else "n/a"))

    nworst = None
    for i, j in pairlist:
        d = delta_e(palette[i], palette[j])
        if nworst is None or d < nworst[0]:
            nworst = (d, palette[i], palette[j])
    nd = nworst[0] if nworst else 99.0
    if nd < NORMAL_FLOOR:
        ok = False
    report.append(("Normal-vision floor", "pass" if nd >= NORMAL_FLOOR else "fail",
                   f"worst {label} {nworst[2]}<->{nworst[1]} ΔE {nd:.1f} (normal)"
                   + ("" if nd >= NORMAL_FLOOR else
                      f" — below {NORMAL_FLOOR:.0f}, hard to tell apart even with full color vision")
                   if nworst else "n/a"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette
           if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "relief" if low else "pass",
                   f"below {CONTRAST_MIN}:1 — relief required (visible labels or table view): {low}"
                   if low else f"all {len(palette)} >= {CONTRAST_MIN}:1"))

    return report, ok


# The reference palette's published results, used to prove this port's math matches
# upstream. If these drift, the port is wrong — not the palette.
_SELF_CHECK_PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500",
                       "#d55181", "#008300", "#9085e9", "#e66767"]
_SELF_CHECK_EXPECTED = {"cvd": 8.4, "normal": 19.3}


def self_check() -> bool:
    adj = list(zip(_SELF_CHECK_PALETTE, _SELF_CHECK_PALETTE[1:]))
    cvd = min(delta_e(a, b, k) for a, b in adj for k in ("protan", "deutan"))
    nor = min(delta_e(a, b) for a, b in adj)
    ok = (round(cvd, 1) == _SELF_CHECK_EXPECTED["cvd"]
          and round(nor, 1) == _SELF_CHECK_EXPECTED["normal"])
    print("Self-check — dark categorical palette, adjacent pairs")
    print(f"  worst CVD    {cvd:6.1f}   expected {_SELF_CHECK_EXPECTED['cvd']}")
    print(f"  worst normal {nor:6.1f}   expected {_SELF_CHECK_EXPECTED['normal']}")
    print("  " + ("OK — port matches the upstream validator"
                  if ok else "MISMATCH — the ported math has drifted"))
    return ok


GLYPH = {"pass": "PASS", "fail": "FAIL", "floor": "WARN", "relief": "WARN"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("palette", nargs="?", help='comma-separated hex, e.g. "#3987e5,#6b6b66"')
    ap.add_argument("--mode", choices=("light", "dark"), default="light")
    ap.add_argument("--surface", help="chart surface hex (defaults per mode)")
    ap.add_argument("--pairs", choices=("adjacent", "all"), default="adjacent",
                    help="adjacent for stacks/bars/lines; all for scatter/bubble/maps")
    ap.add_argument("--self-check", action="store_true",
                    help="verify this port reproduces the reference palette's numbers")
    args = ap.parse_args()

    if args.self_check:
        return 0 if self_check() else 1
    if not args.palette:
        ap.error("a palette is required (or use --self-check)")

    palette = split_colors(args.palette)
    surface = args.surface or DEFAULT_SURFACE[args.mode]
    bad = [c for c in [*palette, surface] if not is_hex(c)]
    if bad:
        print(f"Not 6-digit hex: {bad}", file=sys.stderr)
        return 1
    if not palette:
        print("Empty palette", file=sys.stderr)
        return 1

    report, ok = validate(palette, mode=args.mode, surface=surface, pairs=args.pairs)
    print(f"{len(palette)} colors · mode {args.mode} · surface {surface} · {args.pairs} pairs\n")
    for name, state, detail in report:
        print(f"  {GLYPH.get(state, state):5}  {name:22}  {detail}")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
