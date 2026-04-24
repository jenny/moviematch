"""
tools/haiku_parse_harness.py — Haiku-based structured extraction comparison harness.

Uses Claude Haiku with tool_use (forced structured output) to extract the same tokens
as the regex parser. Run this alongside parse_harness.py to compare:
  - Coverage: which queries does Haiku extract correctly vs. regex?
  - False positives/negatives: where does each approach fail?
  - Cost: actual token counts → $/1000 requests
  - Latency: real API call timing → does it fit in the embedding window (~200ms)?

Usage (from project root, with venv active):
    python tools/haiku_parse_harness.py

Requires ANTHROPIC_API_KEY to be set in .env.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, HAIKU_INPUT_PRICE, HAIKU_OUTPUT_PRICE, CLAUDE_FAST_MODEL

# Reuse the same query list from the regex harness
from tools.parse_harness import TEST_QUERIES


# ---------------------------------------------------------------------------
# Extraction tool schema (mirrors ParsedQuery fields)
# ---------------------------------------------------------------------------

EXTRACT_TOOL = {
    "name": "extract_query_tokens",
    "description": (
        "Extract structured search filters from a movie recommendation query. "
        "Only extract information that is EXPLICITLY stated in the query. "
        "Do NOT infer, assume, or hallucinate filters that aren't clearly present."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "year_min": {
                "type": ["integer", "null"],
                "description": "Minimum release year (inclusive). E.g. 1990 for '90s movies'.",
            },
            "year_max": {
                "type": ["integer", "null"],
                "description": "Maximum release year (inclusive). E.g. 1999 for '90s movies'.",
            },
            "required_genres": {
                "type": "array",
                "items": {"type": "string"},
                "description": "TMDB genre names the user wants (e.g. 'Comedy', 'Science Fiction'). Empty if none specified.",
            },
            "excluded_genres": {
                "type": "array",
                "items": {"type": "string"},
                "description": "TMDB genre names the user explicitly does NOT want (e.g. 'Documentary'). Empty if none excluded.",
            },
            "allowed_certifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "MPAA ratings the user explicitly wants (e.g. ['PG', 'PG-13']). Empty means no restriction.",
            },
            "excluded_certifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "MPAA ratings the user explicitly does NOT want (e.g. ['R']). Empty means none excluded.",
            },
            "persons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Person's full name"},
                        "role": {
                            "type": "string",
                            "enum": ["directing", "cast", "unknown"],
                            "description": "Whether they are referenced as a director or actor. Use 'unknown' if ambiguous.",
                        },
                    },
                    "required": ["name", "role"],
                },
                "description": "Named directors or actors mentioned in the query. Empty if no person is explicitly named.",
            },
            "reference_titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Movie titles referenced as style/similarity anchors (e.g. from 'something like Inception'). Empty if none.",
            },
        },
        "required": [
            "year_min", "year_max",
            "required_genres", "excluded_genres",
            "allowed_certifications", "excluded_certifications",
            "persons", "reference_titles",
        ],
    },
}

SYSTEM_PROMPT = (
    "You are a query parser for a movie recommendation engine. "
    "Your job is to extract structured search filters from natural language queries. "
    "Extract ONLY what is explicitly stated. Do not infer mood, theme, or unstated preferences."
)

USER_PROMPT_TEMPLATE = 'Extract search filters from this movie query:\n\n"{query}"'


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
RED    = "\033[31m"


def _render_result(result: dict) -> list[str]:
    lines = []
    yr_min = result.get("year_min")
    yr_max = result.get("year_max")
    if yr_min is not None or yr_max is not None:
        lines.append(f"  year:      {CYAN}{yr_min or '?'}–{yr_max or '?'}{RESET}")
    rg = result.get("required_genres", [])
    if rg:
        lines.append(f"  genres+:   {GREEN}{rg}{RESET}")
    eg = result.get("excluded_genres", [])
    if eg:
        lines.append(f"  genres-:   {YELLOW}{eg}{RESET}")
    ac = result.get("allowed_certifications", [])
    if ac:
        lines.append(f"  cert+:     {GREEN}{ac}{RESET}")
    ec = result.get("excluded_certifications", [])
    if ec:
        lines.append(f"  cert-:     {YELLOW}{ec}{RESET}")
    persons = result.get("persons", [])
    if persons:
        for p in persons:
            lines.append(f"  person:    {CYAN}{p['name']!r}{RESET}  role={p['role']}")
    ref = result.get("reference_titles", [])
    if ref:
        lines.append(f"  ref_title: {CYAN}{ref}{RESET}")
    if not lines:
        lines.append(f"  {DIM}(no tokens extracted){RESET}")
    return lines


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_haiku_harness() -> None:
    if not ANTHROPIC_API_KEY:
        print(f"{RED}Error: ANTHROPIC_API_KEY is not set. Check your .env file.{RESET}")
        sys.exit(1)

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"\n{BOLD}=== MovieMatch Query Parser — Haiku Extraction Harness ==={RESET}")
    print(f"Model: {CLAUDE_FAST_MODEL}\n")
    print(f"Testing {len(TEST_QUERIES)} queries\n")
    print("─" * 70)

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0
    latencies: list[float] = []
    errors = 0

    for i, entry in enumerate(TEST_QUERIES, 1):
        query = entry["query"]
        expect = entry["expect"]

        try:
            t0 = time.perf_counter()
            response = client.messages.create(
                model=CLAUDE_FAST_MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)}
                ],
                tools=[EXTRACT_TOOL],
                tool_choice={"type": "tool", "name": "extract_query_tokens"},
            )
            latency_ms = round((time.perf_counter() - t0) * 1000)
            latencies.append(latency_ms)
            total_latency_ms += latency_ms

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # Extract tool input from response
            result = {}
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_query_tokens":
                    result = block.input
                    break

            print(f"{BOLD}{i:2}. {query!r}{RESET}")
            print(f"    {DIM}expect: {expect}{RESET}")
            print(f"    {DIM}tokens: {input_tokens}in / {output_tokens}out  latency: {latency_ms}ms{RESET}")
            for line in _render_result(result):
                print(line)
            print()

        except Exception as e:
            errors += 1
            print(f"{BOLD}{i:2}. {query!r}{RESET}")
            print(f"    {RED}ERROR: {e}{RESET}")
            print()

    # Summary
    n = len(TEST_QUERIES) - errors
    print("─" * 70)
    print(f"\n{BOLD}Cost & Latency Summary ({n} successful calls){RESET}")
    print()

    cost_total = (
        total_input_tokens * HAIKU_INPUT_PRICE
        + total_output_tokens * HAIKU_OUTPUT_PRICE
    )
    cost_per_query = cost_total / n if n else 0
    cost_per_1k = cost_per_query * 1000

    avg_latency = total_latency_ms / n if n else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    p90_latency = sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0

    print(f"  Total input tokens  : {total_input_tokens:,}")
    print(f"  Total output tokens : {total_output_tokens:,}")
    print(f"  Cost per query      : ${cost_per_query:.6f}")
    print(f"  {BOLD}Cost per 1,000 reqs : ${cost_per_1k:.4f}{RESET}")
    print()
    print(f"  Avg latency         : {avg_latency:.0f}ms")
    print(f"  Min / Max latency   : {min_latency}ms / {max_latency}ms")
    print(f"  P90 latency         : {p90_latency}ms")
    print()

    # Assessment
    embedding_window_ms = 200  # approximate time available before vector search returns
    if p90_latency <= embedding_window_ms:
        print(f"  {GREEN}✓ P90 latency ({p90_latency}ms) fits within the ~{embedding_window_ms}ms embedding window.{RESET}")
        print(f"    → Haiku call can run in parallel with embedding at zero net overhead.")
    else:
        overhead = p90_latency - embedding_window_ms
        print(f"  {YELLOW}⚠ P90 latency ({p90_latency}ms) exceeds the ~{embedding_window_ms}ms embedding window by ~{overhead}ms.{RESET}")
        print(f"    → Haiku call would add ~{overhead}ms net latency for person queries.")

    print()
    print("Compare these results with tools/parse_harness.py output to decide:")
    print("  • Does Haiku catch more person names / edge cases than regex?")
    print("  • Are false positives lower?")
    print(f"  • Is the ${cost_per_1k:.4f}/1k cost acceptable?")
    print()


if __name__ == "__main__":
    run_haiku_harness()
