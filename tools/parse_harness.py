"""
tools/parse_harness.py — Regex parser test harness for MovieMatch query pre-parsing.

Runs a curated set of ~25 queries through parse_query() and prints a table of
extracted tokens. Use this to evaluate the regex approach before wiring it into
the main pipeline.

Usage (from project root, with venv active):
    python tools/parse_harness.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_parser import parse_query, ParsedQuery


# ---------------------------------------------------------------------------
# Test queries with ground-truth expectations for visual review
# ---------------------------------------------------------------------------

TEST_QUERIES: list[dict] = [
    # --- Person: director ---
    {
        "query": "Christopher Nolan movies",
        "expect": "persons=['Christopher Nolan'], dept=auto→directing",
    },
    {
        "query": "directed by Bong Joon-ho",
        "expect": "persons=['Bong Joon-ho'], dept=directing",
    },
    {
        "query": "films by Wes Anderson",
        "expect": "persons=['Wes Anderson'], dept=directing",
    },
    {
        "query": "by David Fincher movies",
        "expect": "persons=['David Fincher'], dept=directing",
    },
    # --- Person: actor ---
    {
        "query": "something with Tom Hanks",
        "expect": "persons=['Tom Hanks'], dept=cast",
    },
    {
        "query": "starring Meryl Streep",
        "expect": "persons=['Meryl Streep'], dept=cast",
    },
    {
        "query": "movies with Cate Blanchett in them",
        "expect": "persons=['Cate Blanchett'], dept=cast",
    },
    # --- Person: ambiguous / tricky ---
    {
        "query": "Spike Lee joint",
        "expect": "persons=['Spike Lee'], dept=auto",
    },
    {
        "query": "The Coen Brothers movies",
        "expect": "persons=['The Coen Brothers'], dept=auto",
    },
    {
        "query": "The Rock movies",
        "expect": "persons=['The Rock'] (false positive — film title or wrestler person)",
    },
    # --- Year filters ---
    {
        "query": "90s comedies",
        "expect": "year_min=1990, year_max=1999, required_genres=['Comedy']",
    },
    {
        "query": "movies from 2010",
        "expect": "year_min=2010",
    },
    {
        "query": "after 1985",
        "expect": "year_min=1986",
    },
    {
        "query": "before 2000",
        "expect": "year_max=1999",
    },
    {
        "query": "early 2000s thriller",
        "expect": "year_min=2000, year_max=2004, required_genres=['Thriller']",
    },
    {
        "query": "late 80s action",
        "expect": "year_min=1986, year_max=1989, required_genres=['Action']",
    },
    # --- Genre filters ---
    {
        "query": "no documentaries",
        "expect": "excluded_genres=['Documentary']",
    },
    {
        "query": "sci-fi with no horror",
        "expect": "required_genres=['Science Fiction'], excluded_genres=['Horror']",
    },
    {
        "query": "romantic comedies",
        "expect": "required_genres=['Comedy'] (romantic→Romance also possible)",
    },
    {
        "query": "action movies",
        "expect": "required_genres=['Action']",
    },
    # --- Certification filters ---
    {
        "query": "no R-rated movies",
        "expect": "excluded_certifications=['R']",
    },
    {
        "query": "family friendly animated films",
        "expect": "allowed_certifications=['G', 'PG'], required_genres=['Animation']",
    },
    {
        "query": "PG-13 only",
        "expect": "allowed_certifications=['PG-13']",
    },
    # --- Title references ---
    {
        "query": "something like Inception",
        "expect": "reference_titles=['Inception']",
    },
    {
        "query": "similar to The Godfather",
        "expect": "reference_titles=['The Godfather']",
    },
    {
        "query": 'in the style of Wes Anderson',
        "expect": "reference_titles=['Wes Anderson'] (ambiguous: name or title ref)",
    },
    # --- Combos ---
    {
        "query": "Christopher Nolan sci-fi films from the 2000s, no R-rated",
        "expect": "persons=['Christopher Nolan'], required_genres=['Science Fiction'], year_min=2000, year_max=2009, excluded_certifications=['R']",
    },
    {
        "query": "90s comedies with Tom Hanks",
        "expect": "year_min=1990, year_max=1999, required_genres=['Comedy'], persons=['Tom Hanks'], dept=cast",
    },
    # --- Tricky / negative cases ---
    {
        "query": "something fun and light",
        "expect": "all empty (no structured tokens)",
    },
    {
        "query": "great movies",
        "expect": "all empty",
    },
    {
        "query": "something like Inception but older",
        "expect": "reference_titles=['Inception'], relative_date_hints=['older']",
    },
    # --- Film slang trigger words (new) ---
    {
        "query": "a Tarantino pic",
        "expect": "persons=['Tarantino'] — single word MISS (name token requires 2+ words)",
    },
    {
        "query": "great Spielberg flicks",
        "expect": "persons=['Spielberg'] — single word MISS (name token requires 2+ words)",
    },
    # --- "in the style of" dual-purpose (new) ---
    {
        "query": "in the style of Wes Anderson",
        "expect": "reference_titles=['Wes Anderson'], persons=['Wes Anderson'], dept=auto",
    },
    {
        "query": "something like The Godfather",
        "expect": "reference_titles=['The Godfather'] only — 'something like' is title-only",
    },
    # --- Relative date hints (new) ---
    {
        "query": "a more recent thriller",
        "expect": "required_genres=['Thriller'], relative_date_hints=['newer']",
    },
    {
        "query": "classic noir films",
        "expect": "relative_date_hints=['older'] (no genre — 'noir' not in genre map)",
    },
    {
        "query": "something good from the last 10 years",
        "expect": "year_min=current_year-10 (hard filter)",
    },
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _render_parsed(p: ParsedQuery) -> list[str]:
    """Return a list of non-empty token lines for display."""
    lines = []
    if p.year_min is not None or p.year_max is not None:
        yr = f"{p.year_min or '?'}–{p.year_max or '?'}"
        lines.append(f"  year:      {CYAN}{yr}{RESET}")
    if p.required_genres:
        lines.append(f"  genres+:   {GREEN}{p.required_genres}{RESET}")
    if p.excluded_genres:
        lines.append(f"  genres-:   {YELLOW}{p.excluded_genres}{RESET}")
    if p.allowed_certifications:
        lines.append(f"  cert+:     {GREEN}{p.allowed_certifications}{RESET}")
    if p.excluded_certifications:
        lines.append(f"  cert-:     {YELLOW}{p.excluded_certifications}{RESET}")
    if p.person_names:
        lines.append(f"  persons:   {CYAN}{p.person_names}{RESET}  dept={p.person_department}")
    if p.reference_titles:
        lines.append(f"  ref_title: {CYAN}{p.reference_titles}{RESET}")
    if p.relative_date_hints:
        lines.append(f"  date_hint: {YELLOW}{p.relative_date_hints}{RESET}")
    if not lines:
        lines.append(f"  {DIM}(no tokens extracted){RESET}")
    return lines


def run_harness() -> None:
    print(f"\n{BOLD}=== MovieMatch Query Parser — Regex Harness ==={RESET}\n")
    print(f"Testing {len(TEST_QUERIES)} queries\n")
    print("─" * 70)

    empty_count = 0
    token_count = 0

    for i, entry in enumerate(TEST_QUERIES, 1):
        query = entry["query"]
        expect = entry["expect"]
        parsed = parse_query(query)

        has_any = (
            parsed.year_min is not None or parsed.year_max is not None
            or parsed.required_genres or parsed.excluded_genres
            or parsed.allowed_certifications or parsed.excluded_certifications
            or parsed.person_names or parsed.reference_titles
            or parsed.relative_date_hints
        )

        if has_any:
            token_count += 1
        else:
            empty_count += 1

        print(f"{BOLD}{i:2}. {query!r}{RESET}")
        print(f"    {DIM}expect: {expect}{RESET}")
        for line in _render_parsed(parsed):
            print(line)
        print()

    print("─" * 70)
    print(f"Queries with tokens extracted : {GREEN}{token_count}{RESET}")
    print(f"Queries with no tokens        : {DIM}{empty_count}{RESET}")
    print()
    print("Review the output above and look for:")
    print("  • FALSE POSITIVES — tokens extracted where none expected")
    print("  • FALSE NEGATIVES — expected tokens missing (marked MISS in expect)")
    print()


if __name__ == "__main__":
    run_harness()
