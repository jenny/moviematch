"""
query_parser.py — Rule-based query pre-parser for MovieMatch.

Extracts structured tokens from natural language queries before embedding and
vector search. No external API calls — pure regex/pattern matching.

Parsed tokens are used downstream to:
  - Hard-filter vector candidates before Claude sees them (year, genre, certification)
  - Pre-fetch person filmographies concurrently with embedding (avoids Claude tool rounds)
  - Inject explicit constraints into Claude's rerank prompt
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Genre keyword → TMDB canonical genre name
# ---------------------------------------------------------------------------
_GENRE_MAP: dict[str, str] = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "animated": "Animation",
    "comedy": "Comedy",
    "comedies": "Comedy",
    "comic": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "documentaries": "Documentary",
    "drama": "Drama",
    "dramas": "Drama",
    "fantasy": "Fantasy",
    "historical": "History",
    "history": "History",
    "horror": "Horror",
    "music": "Music",
    "musical": "Music",
    "mystery": "Mystery",
    "mysteries": "Mystery",
    "romance": "Romance",
    "romantic": "Romance",
    "romcom": "Comedy",
    "rom-com": "Comedy",
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "science fiction": "Science Fiction",
    "thriller": "Thriller",
    "thrillers": "Thriller",
    "war": "War",
    "western": "Western",
    "westerns": "Western",
}

# Build regex alternations — longest keywords first to prevent partial matches
_genre_keys_sorted = sorted(_GENRE_MAP.keys(), key=len, reverse=True)
_GENRE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _genre_keys_sorted) + r")\b",
    re.IGNORECASE,
)

# Negation context: "no X", "without X", "not X", "avoid X", "except X"
_NEGATION_BEFORE = re.compile(
    r"\b(no|without|not|avoid|except|excluding|exclude)\s+$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Year / decade patterns
# ---------------------------------------------------------------------------

# "early 90s", "late 1980s", "mid-2000s", "90s", "1980s"
# The century prefix (19/20) is optional to handle abbreviated forms like "80s", "90s".
_DECADE_PATTERN = re.compile(
    r"\b(?:(early|late|mid)[- ])?((?:19|20)?\d{2})s\b",
    re.IGNORECASE,
)
_YEAR_AFTER_PATTERN  = re.compile(r"\bafter\s+(\d{4})\b",  re.IGNORECASE)
_YEAR_BEFORE_PATTERN = re.compile(r"\bbefore\s+(\d{4})\b", re.IGNORECASE)
_YEAR_FROM_PATTERN   = re.compile(r"\bfrom\s+(\d{4})\b",   re.IGNORECASE)
_YEAR_IN_PATTERN     = re.compile(r"\bin\s+(\d{4})\b",     re.IGNORECASE)


# ---------------------------------------------------------------------------
# Certification patterns
# ---------------------------------------------------------------------------

# Longest tokens first so PG-13 is matched before PG, NC-17 before N
_CERT_TOKENS = ["NC-17", "PG-13", "PG", "G", "R"]
_CERT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _CERT_TOKENS) + r")\b",
    # Note: intentionally case-sensitive — "R" should not match "r" inside other words
)
_FAMILY_FRIENDLY_PATTERN = re.compile(
    r"\b("
    r"family[- ]friendly|family[- ]film|for kids|kid[- ]friendly|all ages"
    r"|kids?(?:'s?)?\s+(?:movies?|films?)"           # "kids movie", "kid's films"
    r"|(?:movies?|films?)\s+for\s+(?:kids?|children)" # "movies for kids/children"
    r"|children'?s?\s+(?:movies?|films?)"             # "children's movie"
    r"|for\s+children"                                # "for children"
    r")\b",
    re.IGNORECASE,
)

# "live action" / "live-action" — not a TMDB genre; signals exclusion of Animation.
# Negated ("not live action") inversely requires Animation.
_LIVE_ACTION_PATTERN = re.compile(r"\blive[- ]action\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Person name patterns
# ---------------------------------------------------------------------------

# A name token: 2+ capitalized words (possibly hyphenated/apostrophised).
# The optional "The " prefix handles "The Coen Brothers", "The Weeknd" etc.
_NAME_TOKEN = r"(?:The\s+)?[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)+"

# Each entry: (compiled pattern, inferred department)
# Patterns are ordered from most-specific to least-specific.
# department="auto" means: defer to TMDB's known_for_department at resolution time.
#
# IMPORTANT: Do NOT use re.IGNORECASE on patterns that include _NAME_TOKEN.
# The name token relies on capital letters to distinguish proper nouns from regular
# words. IGNORECASE would make it greedily consume lowercase words like "in" or "them".
_PERSON_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Explicit directing context
    (re.compile(r"\bdirected\s+by\s+(" + _NAME_TOKEN + r")"), "directing"),
    (re.compile(r"\b(?:films?|movies?|pictures?)\s+by\s+(" + _NAME_TOKEN + r")"), "directing"),
    (re.compile(r"\bby\s+(" + _NAME_TOKEN + r")\s+(?:movies?|films?|pictures?)"), "directing"),
    # Explicit cast context
    (re.compile(r"\bstarring\s+(" + _NAME_TOKEN + r")"), "cast"),
    (re.compile(r"\bfeaturing\s+(" + _NAME_TOKEN + r")"), "cast"),
    (re.compile(r"\b(?:movies?|films?|pictures?|comedies|thrillers|dramas)\s+(?:with|featuring)\s+(" + _NAME_TOKEN + r")"), "cast"),
    (re.compile(r"\bsomething\s+with\s+(" + _NAME_TOKEN + r")"), "cast"),
    (re.compile(r"\bwith\s+(" + _NAME_TOKEN + r")\s+in\s+(?:it|them)"), "cast"),
    # Generic "with [Name]" — broad cast pattern; fires only for multi-word proper nouns
    (re.compile(r"(?<!\bsimilar\s)(?<!\blike\s)\bwith\s+(" + _NAME_TOKEN + r")(?!\s+(?:no|without|a\b))"), "cast"),
    # Ambiguous: "[Name]'s movies" or "[Name] movies" — allows up to 2 words between name and trigger
    # to handle cases like "Christopher Nolan sci-fi films"
    (re.compile(r"\b(" + _NAME_TOKEN + r")(?:'s)?(?:\s+\S+){0,2}\s+(?:movies?|films?|pictures?|works?|catalogue|catalog|joints?|flicks?|pics?)"), "auto"),
]


# ---------------------------------------------------------------------------
# Title reference patterns ("something like X", "similar to X")
# ---------------------------------------------------------------------------

# Each entry: (compiled pattern, is_style_of)
# is_style_of=True means "in the style of X" — X is likely a director name,
# so the extracted title is also checked against _NAME_TOKEN and added to
# person_names if it matches. Other patterns are title-only.
_TITLE_REF_PATTERNS: list[tuple[re.Pattern, bool]] = [
    # "something like Inception" / "something like Inception but older"
    (re.compile(
        r"\bsomething\s+like\s+[\"']?([A-Z][A-Za-z0-9\s:!?'&,-]{1,50}?)[\"']?"
        r"(?=\s+but|\s+only|\s+with|\s*[,.]|$)",
        re.IGNORECASE,
    ), False),
    # "similar to The Godfather"
    (re.compile(
        r"\bsimilar\s+to\s+[\"']?([A-Z][A-Za-z0-9\s:!?'&,-]{1,50}?)[\"']?"
        r"(?=\s+but|\s+only|\s+with|\s*[,.]|$)",
        re.IGNORECASE,
    ), False),
    # "in the style of Wes Anderson" — X is treated as both a title ref and
    # a potential person name (director). TMDB miss degrades gracefully.
    (re.compile(
        r"\bin\s+the\s+style\s+of\s+[\"']?([A-Z][A-Za-z0-9\s:!?'&,-]{1,50}?)[\"']?"
        r"(?=\s*[,.]|$)",
        re.IGNORECASE,
    ), True),
    # Quoted titles after "like": like "The Dark Knight"
    (re.compile(r"\blike\s+[\"']([^\"']{1,60})[\"']", re.IGNORECASE), False),
]

# Matches a name-shaped string: 2+ capitalized words (with optional "The" prefix).
# Used to check whether a "style of" extraction looks like a person name.
_NAME_TOKEN_RE = re.compile(
    r"^(?:The\s+)?[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)+$"
)

# ---------------------------------------------------------------------------
# Relative date patterns
# ---------------------------------------------------------------------------

# Soft hints ("older", "recent") → injected into Claude prompt for guidance.
# Hard constraint ("last N years") → converted directly to year_min.
_RELATIVE_DATE_SOFT = [
    (re.compile(r"\b(older|classic|vintage|retro)\b", re.IGNORECASE), "older"),
    (re.compile(r"\b(newer|more\s+recent|recent|modern|contemporary)\b", re.IGNORECASE), "newer"),
]
_LAST_N_YEARS_PATTERN = re.compile(
    r"\b(?:from\s+the\s+)?last\s+(\d+)\s+years?\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """Structured tokens extracted from a natural language movie search query."""

    # Temporal filters
    year_min: int | None = None
    year_max: int | None = None

    # Genre filters (TMDB canonical names, e.g. "Science Fiction", "Documentary")
    required_genres: list[str] = field(default_factory=list)
    excluded_genres: list[str] = field(default_factory=list)

    # MPAA certification filters
    allowed_certifications: list[str] = field(default_factory=list)   # empty = no restriction
    excluded_certifications: list[str] = field(default_factory=list)  # always blocked
    # Soft guidance injected into Claude's prompt (not a hard filter)
    certification_caveats: list[str] = field(default_factory=list)

    # Person extraction (resolved to filmographies by resolve_persons())
    person_names: list[str] = field(default_factory=list)
    person_department: str = "auto"  # "directing", "cast", or "auto" (infer from TMDB)

    # Populated after TMDB lookup in resolve_persons()
    person_filmographies: list[dict] = field(default_factory=list)
    is_person_focused: bool = False

    # "Something like X" title references
    reference_titles: list[str] = field(default_factory=list)

    # Soft temporal signals: "older", "newer" — not hard filters, injected into
    # Claude's prompt as guidance. "last_N_years" is converted to year_min instead.
    relative_date_hints: list[str] = field(default_factory=list)

    def has_filters(self) -> bool:
        """True if any hard filter constraint or soft hint was extracted."""
        return bool(
            self.year_min is not None or self.year_max is not None
            or self.required_genres or self.excluded_genres
            or self.allowed_certifications or self.excluded_certifications
            or self.relative_date_hints
        )

    def has_persons(self) -> bool:
        """True if at least one person name was extracted."""
        return bool(self.person_names)


# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------

def parse_query(raw_query: str) -> ParsedQuery:
    """
    Extract structured tokens from a natural language movie query.
    Pure regex — no I/O, no external calls. Runs in <1ms.

    False negatives (missing tokens) degrade gracefully to Claude's existing
    tool-call path. False positives (wrong tokens) may trigger a harmless extra
    TMDB call or over-filter candidates — the harness reveals these cases.
    """
    parsed = ParsedQuery()
    q = raw_query  # preserve case for person name extraction

    # --- Year / decade ---
    for m in _DECADE_PATTERN.finditer(q):
        qualifier = (m.group(1) or "").lower()  # "early", "late", "mid", or ""
        raw = int(m.group(2))
        # Normalise abbreviated decades (e.g. "90" → 1990, "00" → 2000)
        if raw < 100:
            decade_start = (1900 + raw) if raw >= 30 else (2000 + raw)
        else:
            decade_start = (raw // 10) * 10  # e.g. 1990 → 1990, 2004 stays as 2000
        if qualifier == "early":
            parsed.year_min = decade_start
            parsed.year_max = decade_start + 4
        elif qualifier == "late":
            parsed.year_min = decade_start + 6
            parsed.year_max = decade_start + 9
        elif qualifier == "mid":
            parsed.year_min = decade_start + 3
            parsed.year_max = decade_start + 7
        else:
            parsed.year_min = decade_start
            parsed.year_max = decade_start + 9

    if m := _YEAR_AFTER_PATTERN.search(q):
        parsed.year_min = int(m.group(1)) + 1

    if m := _YEAR_BEFORE_PATTERN.search(q):
        parsed.year_max = int(m.group(1)) - 1

    # "from 2010" means "starting from 2010" — only apply if decade didn't already fire
    if (m := _YEAR_FROM_PATTERN.search(q)) and parsed.year_min is None:
        parsed.year_min = int(m.group(1))

    # "in 1994" → exact year, only if no other year constraint set
    if (m := _YEAR_IN_PATTERN.search(q)) and parsed.year_min is None and parsed.year_max is None:
        year = int(m.group(1))
        if 1900 <= year <= 2030:
            parsed.year_min = year
            parsed.year_max = year

    # --- Genres ---
    for m in _GENRE_PATTERN.finditer(q):
        genre_name = _GENRE_MAP[m.group(1).lower()]
        # Inspect the 40 chars preceding this match for a negation word
        before = q[max(0, m.start() - 40) : m.start()]
        if _NEGATION_BEFORE.search(before):
            if genre_name not in parsed.excluded_genres:
                parsed.excluded_genres.append(genre_name)
        else:
            if genre_name not in parsed.required_genres:
                parsed.required_genres.append(genre_name)

    # "live action" is not a TMDB genre — treat it as "exclude Animation".
    # Negated ("not live action") is treated as "require Animation".
    for m in _LIVE_ACTION_PATTERN.finditer(q):
        before = q[max(0, m.start() - 40) : m.start()]
        if _NEGATION_BEFORE.search(before):
            if "Animation" not in parsed.required_genres:
                parsed.required_genres.append("Animation")
        else:
            if "Animation" not in parsed.excluded_genres:
                parsed.excluded_genres.append("Animation")

    # --- Certifications ---
    if _FAMILY_FRIENDLY_PATTERN.search(q):
        # Hard-exclude NC-17; all other ratings remain as candidates for Claude.
        if "NC-17" not in parsed.excluded_certifications:
            parsed.excluded_certifications.append("NC-17")
        # Soft guidance: Claude prefers G/PG/PG-13 and only surfaces R as a last resort.
        parsed.certification_caveats.append(
            "Prefer G, PG, or PG-13 results. Only include an R-rated film if it is "
            "clearly the best match and no suitable alternative exists; if you do, "
            "note the rating explicitly in your explanation."
        )

    for m in _CERT_PATTERN.finditer(q):
        cert = m.group(1)
        before = q[max(0, m.start() - 40) : m.start()]
        if _NEGATION_BEFORE.search(before):
            if cert not in parsed.excluded_certifications:
                parsed.excluded_certifications.append(cert)
        else:
            if cert not in parsed.allowed_certifications:
                parsed.allowed_certifications.append(cert)

    # --- Person names ---
    # Track names seen across patterns to avoid duplicates
    seen_names: set[str] = set()
    for pattern, department in _PERSON_PATTERNS:
        for m in pattern.finditer(q):
            name = m.group(1).strip()
            # Reject single-word matches and genre/adjective words
            if len(name.split()) < 2:
                continue
            if name.lower() in _GENRE_MAP:
                continue
            if name not in seen_names:
                seen_names.add(name)
                parsed.person_names.append(name)
                # First explicit department (directing/cast) wins; "auto" is a fallback
                if parsed.person_department == "auto" and department != "auto":
                    parsed.person_department = department
                elif department != "auto":
                    parsed.person_department = department

    # --- Title references ---
    seen_titles: set[str] = set()
    for pattern, is_style_of in _TITLE_REF_PATTERNS:
        for m in pattern.finditer(q):
            title = m.group(1).strip().rstrip(".,")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            parsed.reference_titles.append(title)
            # "in the style of X" — if X looks like a person name, also treat
            # it as a person lookup. A TMDB miss degrades gracefully.
            if is_style_of and _NAME_TOKEN_RE.match(title) and title not in parsed.person_names:
                parsed.person_names.append(title)
                # Don't override an already-resolved department; "auto" lets TMDB decide.

    # --- Relative date hints ---
    # Hard constraint: "last N years" → concrete year_min
    if m := _LAST_N_YEARS_PATTERN.search(q):
        n = int(m.group(1))
        current_year = datetime.date.today().year
        if parsed.year_min is None:
            parsed.year_min = current_year - n

    # Soft hints: injected into Claude prompt, not a hard filter
    for pattern, hint in _RELATIVE_DATE_SOFT:
        if pattern.search(q) and hint not in parsed.relative_date_hints:
            parsed.relative_date_hints.append(hint)

    return parsed


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def apply_hard_filters(candidates: list[dict], parsed: ParsedQuery) -> list[dict]:
    """
    Remove candidates that violate hard constraints extracted from the query.

    Conservative: candidates with missing/empty year or certification are kept
    to avoid false drops on incomplete metadata (common for older ingested movies).
    """
    if not parsed.has_filters():
        return candidates

    filtered = []
    for c in candidates:
        title = c.get("title", "?")

        # Year filter
        year_str = c.get("year", "")
        if year_str:
            try:
                year = int(year_str)
                if parsed.year_min is not None and year < parsed.year_min:
                    logger.debug("Filtered %r: year %d < min %d", title, year, parsed.year_min)
                    continue
                if parsed.year_max is not None and year > parsed.year_max:
                    logger.debug("Filtered %r: year %d > max %d", title, year, parsed.year_max)
                    continue
            except ValueError:
                pass  # keep if year can't be parsed

        # Genre filters — conservatively keep candidates with no genre metadata,
        # since many older ingested movies may be missing genre tags.
        genres = set(c.get("genres", []))
        if genres:
            if parsed.excluded_genres and genres.intersection(parsed.excluded_genres):
                logger.debug("Filtered %r: excluded genre in %r", title, list(genres))
                continue
            if parsed.required_genres and not genres.intersection(parsed.required_genres):
                logger.debug("Filtered %r: no required genre found in %r", title, list(genres))
                continue

        # Certification filters
        cert = c.get("certification", "")
        if cert:
            if parsed.excluded_certifications and cert in parsed.excluded_certifications:
                logger.debug("Filtered %r: excluded certification %r", title, cert)
                continue
            if parsed.allowed_certifications and cert not in parsed.allowed_certifications:
                logger.debug("Filtered %r: cert %r not in allowlist %r", title, cert, parsed.allowed_certifications)
                continue

        filtered.append(c)

    removed = len(candidates) - len(filtered)
    if removed:
        logger.info("Hard filter removed %d of %d candidates", removed, len(candidates))
    return filtered


# ---------------------------------------------------------------------------
# Person resolution (intended for background thread execution)
# ---------------------------------------------------------------------------

def resolve_persons(parsed: ParsedQuery) -> None:
    """
    Resolve extracted person names to TMDB filmographies. Mutates parsed in place.

    Designed to run in a ThreadPoolExecutor concurrent with embedding+vector search,
    so the TMDB round-trip latency is absorbed instead of added.

    Uses TMDB's known_for_department to pick the correct credit type when the
    query context is ambiguous (person_department == "auto").
    """
    from tmdb import search_person, get_filmography  # imported here to avoid circular deps

    for name in parsed.person_names:
        try:
            results = search_person(name)
            if not results:
                logger.info("No TMDB results for person %r — skipping pre-fetch", name)
                continue

            person = results[0]
            person_id = person["id"]
            tmdb_dept = person.get("known_for_department", "")

            # Determine credit type: explicit query context takes precedence
            if parsed.person_department == "directing":
                department = "directing"
            elif parsed.person_department == "cast":
                department = "cast"
            elif tmdb_dept == "Directing":
                department = "directing"
            else:
                # Default to cast for actors/unknown; TMDB returns "Acting" for actors
                department = "cast"

            movies = get_filmography(person_id, department)
            titles = [m["title"] for m in movies if m.get("title")]

            parsed.person_filmographies.append({
                "name": person["name"],
                "person_id": person_id,
                "department": department,
                "titles": titles,
                # Raw movie dicts (with id, vote_average, vote_count) are stored so
                # main.py can trigger background ingestion after the future resolves,
                # mirroring the ingestion that happens when Claude calls get_filmography.
                "movies": movies,
            })
            parsed.is_person_focused = True
            logger.info(
                "Pre-fetched %d titles for %r (%s)",
                len(titles), person["name"], department,
            )

        except Exception as e:
            # Log and continue — caller falls back to Claude's tool-call path
            logger.warning("Person pre-fetch failed for %r: %s", name, e)
