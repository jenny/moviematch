import json
import logging
import threading

from anthropic import Anthropic, RateLimitError, InternalServerError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_FAST_MODEL, AGENT_MAX_TOOL_ROUNDS, FILMOGRAPHY_INGEST_LIMIT, FORCE_FAST_MODEL
from tmdb import search_person, get_filmography

logger = logging.getLogger(__name__)
_client = None
_client_lock = threading.Lock()


def get_client() -> Anthropic:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not ANTHROPIC_API_KEY:
                    raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")
                _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def prompt_claude(message: str) -> str:
    response = get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text


# Tool name constants — used in schema dicts AND all name comparisons so a typo
# in one place is caught rather than silently doing nothing at runtime.
_TOOL_SEARCH_PERSON = "search_person"
_TOOL_GET_FILMOGRAPHY = "get_filmography"
_TOOL_RETURN_RESULTS = "return_results"

SEARCH_PERSON_TOOL = {
    "name": _TOOL_SEARCH_PERSON,
    "description": (
        "Search TMDB for a person (director, actor, writer) by name. "
        "Returns their ID and known works. Use this when the user mentions a specific person by name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's name"},
        },
        "required": ["name"],
    },
}

GET_FILMOGRAPHY_TOOL = {
    "name": _TOOL_GET_FILMOGRAPHY,
    "description": (
        "Get movie credits for a person given their TMDB person ID. "
        "Use department='directing' for directors and department='cast' for actors."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "person_id": {"type": "integer", "description": "TMDB person ID"},
            "department": {
                "type": "string",
                "enum": ["directing", "cast"],
                "description": "Which credits to return",
            },
        },
        "required": ["person_id", "department"],
    },
}

RETURN_RESULTS_TOOL = {
    "name": _TOOL_RETURN_RESULTS,
    "description": "Return the final reranked and filtered movie recommendations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["title", "explanation"],
                },
            }
        },
        "required": ["results"],
    },
}

TOOLS = [SEARCH_PERSON_TOOL, GET_FILMOGRAPHY_TOOL, RETURN_RESULTS_TOOL]


def _sanitize(s: str) -> str:
    """Strip XML-significant characters from user-derived strings before prompt injection.
    Prevents prompt injection via crafted queries containing XML tags."""
    return s.replace("<", "").replace(">", "").replace("&", "and")


def _build_rerank_prompt(query: str, candidate_text: str, parsed=None) -> str:
    # Build optional constraint and filmography blocks from pre-parsed tokens.
    constraint_lines = []
    if parsed:
        if parsed.year_min is not None and parsed.year_max is not None:
            constraint_lines.append(f"- Year range: {parsed.year_min}–{parsed.year_max} only")
        elif parsed.year_min is not None:
            constraint_lines.append(f"- Released {parsed.year_min} or later")
        elif parsed.year_max is not None:
            constraint_lines.append(f"- Released {parsed.year_max} or earlier")
        if parsed.required_genres:
            constraint_lines.append(f"- Required genres: {', '.join(_sanitize(g) for g in parsed.required_genres)}")
        if parsed.excluded_genres:
            constraint_lines.append(f"- Exclude genres: {', '.join(_sanitize(g) for g in parsed.excluded_genres)}")
        if parsed.allowed_certifications:
            constraint_lines.append(f"- Certifications allowed: {', '.join(_sanitize(c) for c in parsed.allowed_certifications)}")
        if parsed.excluded_certifications:
            constraint_lines.append(f"- Exclude certifications: {', '.join(_sanitize(c) for c in parsed.excluded_certifications)}")
        for caveat in parsed.certification_caveats:
            constraint_lines.append(f"- {_sanitize(caveat)}")
        for hint in parsed.relative_date_hints:
            if hint == "older":
                constraint_lines.append("- Prefer older/classic films")
            elif hint == "newer":
                constraint_lines.append("- Prefer newer/recent films")

    constraints_block = ""
    if constraint_lines:
        constraints_block = "<constraints>\n" + "\n".join(constraint_lines) + "\n</constraints>\n\n"

    # Build reference films block when the user is looking for movies similar to
    # specific titles (e.g. "More movies like Inception"). This makes the anchor
    # explicit for Claude rather than leaving it buried in the query text.
    reference_films_block = ""
    if parsed and parsed.reference_titles:
        titles_str = ", ".join(_sanitize(t) for t in parsed.reference_titles)
        reference_lines = [f"The user wants movies similar to: {titles_str}"]
        # When a soft qualifier rode along ("like X but funnier"), the candidates were
        # retrieved from the reference's neighborhood; instruct Claude to reorder WITHIN
        # that neighborhood toward the qualifier rather than abandon the similarity.
        if parsed.has_soft_qualifier:
            reference_lines.append(
                f"Among these similar films, prefer ones that are: {_sanitize(parsed.residual_query)}"
            )
        reference_films_block = (
            "<reference_films>\n" + "\n".join(reference_lines) + "\n</reference_films>\n\n"
        )

    # Build filmography block for pre-resolved persons. Cap at 30 titles per person
    # to keep token cost bounded (prolific directors can have 50+ credits).
    filmography_block = ""
    suppress_instruction = ""
    if parsed and parsed.person_filmographies:
        lines = []
        resolved_names = []
        for pf in parsed.person_filmographies:
            name = _sanitize(pf["name"])
            dept = pf["department"]
            titles = [_sanitize(t) for t in pf["titles"][:30]]
            lines.append(f"{name} ({dept}): {', '.join(titles)}")
            resolved_names.append(name)
        filmography_block = "<filmography>\n" + "\n".join(lines) + "\n</filmography>\n\n"
        resolved_str = ", ".join(resolved_names)
        suppress_instruction = (
            f"The following people's filmographies have already been retrieved: {resolved_str}. "
            f"Do NOT call search_person or get_filmography for them. "
            f"For any other person mentioned in the query, you may still use search_person.\n\n"
        )

    # Determine tool-call instruction based on whether all persons were pre-resolved.
    all_resolved = (
        parsed is not None
        and bool(parsed.person_names)
        and len(parsed.person_filmographies) == len(parsed.person_names)
    )
    if all_resolved:
        tool_instruction = (
            "When you have enough information, use return_results with your final recommendations:\n"
            "1. Filter out candidates that don't match the query\n"
            "2. Rerank from most to least relevant, including any relevant films from the filmography above\n"
            "3. Write a brief explanation for each result"
        )
    else:
        tool_instruction = (
            "If the query mentions a specific director, actor, or filmmaker by name, use search_person "
            "to find them, then get_filmography to see their work — this ensures you can recommend their "
            "films even if they're not in the candidate list above.\n\n"
            "When you have enough information, use return_results with your final recommendations:\n"
            "1. Filter out candidates that don't match the query\n"
            "2. Rerank from most to least relevant, including any relevant films from a filmography lookup\n"
            "3. Write a brief explanation for each result"
        )

    return (
        f"You are a movie recommendation assistant. A user searched for:\n"
        f"<query>{_sanitize(query)}</query>\n\n"
        f"{constraints_block}"
        f"{reference_films_block}"
        f"{filmography_block}"
        f"{suppress_instruction}"
        f"Here are candidate movies retrieved by semantic search:\n\n"
        f"{candidate_text}\n\n"
        f"{tool_instruction}\n\n"
        f"Only include movies that are genuinely relevant."
    )


_ingesting_ids: set[int] = set()
_ingesting_lock = threading.Lock()


def _ingest_filmography_background(movies: list[dict]) -> None:
    from pipeline import ingest_single
    for movie in movies[:FILMOGRAPHY_INGEST_LIMIT]:
        movie_id = movie["id"]
        with _ingesting_lock:
            if movie_id in _ingesting_ids:
                continue
            _ingesting_ids.add(movie_id)
        try:
            ingest_single(movie_id, movie.get("vote_average", 0), movie.get("vote_count", 0))
        except Exception as e:
            logger.warning(f"Background ingestion failed for {movie.get('title', movie_id)}: {e}")
        finally:
            with _ingesting_lock:
                _ingesting_ids.discard(movie_id)


def _ingest_reference_background(references: list[dict]) -> None:
    """Ingest user-referenced titles ("movies like X") in the background.

    references: [{title, id}, ...] from parsed.reference_movie_ids. Ingestion is
    force=True (quality gate bypassed) — the user named the film explicitly, so even
    an obscure one should be anchorable next time. Films already ingested are a no-op
    inside ingest_single. De-duplicated against in-flight ingests via _ingesting_ids.
    """
    from pipeline import ingest_single
    for ref in references:
        movie_id = ref.get("id")
        if movie_id is None:
            continue
        with _ingesting_lock:
            if movie_id in _ingesting_ids:
                continue
            _ingesting_ids.add(movie_id)
        try:
            # vote_average/vote_count unknown here (we only resolved the ID); force
            # bypasses the gate, and ingest_single fetches full detail from TMDB.
            ingest_single(movie_id, 0, 0, force=True)
        except Exception as e:
            logger.warning(f"Background reference ingestion failed for {ref.get('title', movie_id)}: {e}")
        finally:
            with _ingesting_lock:
                _ingesting_ids.discard(movie_id)


def execute_tool(name: str, tool_input: dict) -> dict:
    args_summary = ", ".join(f"{k}={v}" for k, v in tool_input.items())
    logger.debug(f"Tool call: {name}({args_summary})")
    try:
        if name == _TOOL_SEARCH_PERSON:
            return {"results": search_person(tool_input["name"])}
        elif name == _TOOL_GET_FILMOGRAPHY:
            movies = get_filmography(tool_input["person_id"], tool_input.get("department", "directing"))
            threading.Thread(
                target=_ingest_filmography_background,
                args=(movies,),
                daemon=True,
            ).start()
            return {"movies": movies}
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.warning(f"Tool {name} failed: {e}")
        return {"error": str(e)}


def _is_transient_claude_error(exc: BaseException) -> bool:
    return isinstance(exc, (RateLimitError, InternalServerError, APIConnectionError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_transient_claude_error),
)
def _call_claude(model: str, messages: list, max_tokens: int = 1024, tools: list = TOOLS) -> object:
    return get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice={"type": "any"},
        messages=messages,
    )


def _format_model_log(models_used: list) -> str:
    return "→".join("haiku" if m == CLAUDE_FAST_MODEL else "opus" for m in models_used)


def _filter_results(results: list[dict], valid_titles: set[str]) -> list[dict]:
    """Drop fabricated titles and deduplicate, preserving order.

    Uses case-insensitive matching to handle minor casing differences between
    TMDB filmography titles and the strings stored in the vector DB.
    """
    valid_lower = {t.lower() for t in valid_titles}
    filtered = []
    seen: set[str] = set()
    rejected = []
    for r in results:
        title = r.get("title") or ""
        if title.lower() not in valid_lower:
            rejected.append(title)
        elif title not in seen:
            filtered.append(r)
            seen.add(title)
    if rejected:
        logger.warning(f"return_results validation: rejected fabricated title(s): {rejected}")
    return filtered


def _execute_non_terminal_tools(
    tool_uses: list,
    assistant_content: list,
    valid_titles: set[str],
    tools_called: list[str],
) -> list[dict]:
    """Execute a round of non-terminal tool calls and build the next two conversation messages.

    Mutates valid_titles (adds filmography titles discovered via get_filmography) and
    tools_called (appends each tool name) in-place so callers don't need to handle these.
    Returns [assistant_message, tool_results_message] ready to extend messages with.

    Shared between rerank() and rerank_stream() to keep the tool-dispatch logic in one place.
    """
    tools_called.extend(t.name for t in tool_uses)
    tool_results = []
    for t in tool_uses:
        result = execute_tool(t.name, t.input)
        if t.name == _TOOL_GET_FILMOGRAPHY:
            # Seed valid_titles with filmography discoveries so Claude can recommend
            # films that weren't in the original vector search results.
            for movie in result.get("movies", []):
                if movie.get("title"):
                    valid_titles.add(movie["title"])
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": t.id,
            "content": json.dumps(result),
        })
    return [
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": tool_results},
    ]


def rerank(query: str, candidates: list[dict], parsed=None) -> tuple[list[dict], dict]:
    candidate_text = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in candidates
    )
    prompt = _build_rerank_prompt(query, candidate_text, parsed=parsed)

    messages = [{"role": "user", "content": prompt}]
    total_input_tokens = 0
    total_output_tokens = 0
    haiku_input_tokens = 0
    haiku_output_tokens = 0
    opus_input_tokens = 0
    opus_output_tokens = 0
    tools_called = []
    models_used = []
    current_model = CLAUDE_FAST_MODEL
    exit_reason = "exhausted"
    valid_titles = {c["title"] for c in candidates}

    # Seed valid_titles with pre-fetched filmography titles so Claude can recommend
    # films discovered via pre-fetch that weren't in the vector search results.
    if parsed and parsed.person_filmographies:
        for pf in parsed.person_filmographies:
            valid_titles.update(pf["titles"])

    # If all requested persons were resolved, remove person-lookup tools entirely —
    # a hard guarantee that Claude won't trigger redundant TMDB calls.
    all_resolved = (
        parsed is not None
        and bool(parsed.person_names)
        and len(parsed.person_filmographies) == len(parsed.person_names)
    )
    active_tools = [RETURN_RESULTS_TOOL] if all_resolved else TOOLS

    for round_num in range(1, AGENT_MAX_TOOL_ROUNDS + 1):
        models_used.append(current_model)
        response = _call_claude(current_model, messages, max_tokens=1024 if current_model == CLAUDE_FAST_MODEL else 2048, tools=active_tools)

        in_toks = response.usage.input_tokens
        out_toks = response.usage.output_tokens
        total_input_tokens += in_toks
        total_output_tokens += out_toks
        # Round 1 always uses CLAUDE_FAST_MODEL (Haiku); subsequent rounds use
        # CLAUDE_MODEL (Opus) after non-terminal tools are invoked (see below).
        if current_model == CLAUDE_FAST_MODEL:
            haiku_input_tokens += in_toks
            haiku_output_tokens += out_toks
        else:
            opus_input_tokens += in_toks
            opus_output_tokens += out_toks

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            exit_reason = "no_tool_uses"
            break

        non_terminal = [t for t in tool_uses if t.name != _TOOL_RETURN_RESULTS]

        for tool_use in tool_uses:
            if tool_use.name == _TOOL_RETURN_RESULTS:
                # Only record tools that were actually executed (non-terminal tools
                # from prior rounds); non_terminal tools in this round are not executed
                # since we're returning immediately
                model_log = _format_model_log(models_used)
                usage = {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "haiku_input_tokens": haiku_input_tokens,
                    "haiku_output_tokens": haiku_output_tokens,
                    "opus_input_tokens": opus_input_tokens,
                    "opus_output_tokens": opus_output_tokens,
                    "rounds": round_num,
                    "tools_called": tools_called,
                }
                logger.info(
                    f"Claude usage: {total_input_tokens} input tokens, "
                    f"{total_output_tokens} output tokens, "
                    f"{round_num} round(s), "
                    f"models={model_log}, "
                    f"tools={tools_called or 'none'}"
                )
                raw = tool_use.input.get("results", [])
                results = _filter_results(raw, valid_titles)
                return results, usage

        # No return_results this round — execute non-terminal tools and continue.
        # _execute_non_terminal_tools mutates valid_titles and tools_called in-place.
        if not FORCE_FAST_MODEL:
            current_model = CLAUDE_MODEL
        messages.extend(
            _execute_non_terminal_tools(non_terminal, response.content, valid_titles, tools_called)
        )

    model_log = _format_model_log(models_used)
    usage = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "haiku_input_tokens": haiku_input_tokens,
        "haiku_output_tokens": haiku_output_tokens,
        "opus_input_tokens": opus_input_tokens,
        "opus_output_tokens": opus_output_tokens,
        "rounds": len(models_used),
        "tools_called": tools_called,
    }
    logger.warning(f"Claude agent loop ended ({exit_reason}) after {len(models_used)} rounds without return_results — returning no results. models={model_log}")
    return [], usage


def _extract_result_objects(partial_json: str) -> list[str]:
    """Extract complete {...} objects from the results array in partially-streamed JSON.

    Finds the 'results' key, locates its array, then extracts depth-1 objects
    within that array. This is robust to extra top-level keys being added to
    the return_results tool schema in the future.
    """
    results_key = partial_json.find('"results"')
    if results_key == -1:
        return []
    array_start = partial_json.find('[', results_key)
    if array_start == -1:
        return []

    objects = []
    depth = 0
    start = -1
    in_string = False
    i = array_start + 1  # start scanning inside the array
    while i < len(partial_json):
        ch = partial_json[i]
        if in_string:
            if ch == "\\":
                # Skip the backslash and the following character. This correctly
                # handles all single-char JSON escapes (\", \\, \/, \b, \f, \n,
                # \r, \t) and the \u in \uXXXX sequences — the 4 hex digits that
                # follow are 0-9/a-f/A-F, none of which are ", {, or }, so they
                # are safely processed as ordinary characters in subsequent loops.
                i += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
                if depth == 1:
                    start = i
            elif ch == "}":
                if depth == 1 and start != -1:
                    objects.append(partial_json[start:i + 1])
                    start = -1
                depth -= 1
            elif ch == "]" and depth == 0:
                break  # end of results array
        i += 1
    return objects


def rerank_stream(query: str, candidates: list[dict], parsed=None):
    """Streaming version of rerank(). Yields result dicts one by one as they
    complete in Claude's return_results tool call, then yields {"__usage": {...}}."""
    candidate_text = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in candidates
    )
    prompt = _build_rerank_prompt(query, candidate_text, parsed=parsed)

    messages = [{"role": "user", "content": prompt}]
    total_input_tokens = 0
    total_output_tokens = 0
    haiku_input_tokens = 0
    haiku_output_tokens = 0
    opus_input_tokens = 0
    opus_output_tokens = 0
    tools_called = []
    models_used = []
    current_model = CLAUDE_FAST_MODEL
    valid_titles = {c["title"] for c in candidates}

    # Seed valid_titles with pre-fetched filmography titles.
    if parsed and parsed.person_filmographies:
        for pf in parsed.person_filmographies:
            valid_titles.update(pf["titles"])

    # Hard-remove person-lookup tools when all persons were pre-resolved.
    all_resolved = (
        parsed is not None
        and bool(parsed.person_names)
        and len(parsed.person_filmographies) == len(parsed.person_names)
    )
    active_tools = [RETURN_RESULTS_TOOL] if all_resolved else TOOLS
    # Case-insensitive lookup for the inline streaming filter and the final safety-net filter.
    valid_lower = {t.lower() for t in valid_titles}

    for round_num in range(1, AGENT_MAX_TOOL_ROUNDS + 1):
        models_used.append(current_model)
        return_results_block_idx = None
        return_results_json = ""
        results_yielded = 0
        yielded_titles: set[str] = set()

        try:
            with get_client().messages.stream(
                model=current_model,
                max_tokens=1024 if current_model == CLAUDE_FAST_MODEL else 2048,
                tools=active_tools,
                tool_choice={"type": "any"},
                messages=messages,
            ) as stream:
                for event in stream:
                    if (event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "tool_use"
                            and event.content_block.name == _TOOL_RETURN_RESULTS):
                        return_results_block_idx = event.index
                    elif (event.type == "content_block_delta"
                            and hasattr(event, "delta")
                            and hasattr(event.delta, "partial_json")
                            and event.index == return_results_block_idx):
                        return_results_json += event.delta.partial_json
                        for obj_str in _extract_result_objects(return_results_json)[results_yielded:]:
                            try:
                                result = json.loads(obj_str)
                                if "title" in result and "explanation" in result:
                                    title = result["title"]
                                    if title.lower() not in valid_lower:
                                        logger.warning(
                                            f"rerank_stream: rejected fabricated title during streaming: {title!r}"
                                        )
                                    elif title not in yielded_titles:
                                        yield result
                                        yielded_titles.add(title)
                                results_yielded += 1
                            except json.JSONDecodeError:
                                pass
                final = stream.get_final_message()
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            logger.error(f"Claude streaming error: {e}")
            yield {"__usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "haiku_input_tokens": haiku_input_tokens,
                "haiku_output_tokens": haiku_output_tokens,
                "opus_input_tokens": opus_input_tokens,
                "opus_output_tokens": opus_output_tokens,
                "rounds": len(models_used),
                "tools_called": tools_called,
                "error": str(e),
            }}
            return

        in_toks = final.usage.input_tokens
        out_toks = final.usage.output_tokens
        total_input_tokens += in_toks
        total_output_tokens += out_toks
        if current_model == CLAUDE_FAST_MODEL:
            haiku_input_tokens += in_toks
            haiku_output_tokens += out_toks
        else:
            opus_input_tokens += in_toks
            opus_output_tokens += out_toks

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        return_results_call = next((t for t in tool_uses if t.name == _TOOL_RETURN_RESULTS), None)

        if return_results_call:
            # Safety net: yield any valid results the streaming parser missed
            for r in _filter_results(return_results_call.input.get("results", []), valid_titles):
                if r["title"] not in yielded_titles:
                    yield r
            model_log = _format_model_log(models_used)
            usage = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "haiku_input_tokens": haiku_input_tokens,
                "haiku_output_tokens": haiku_output_tokens,
                "opus_input_tokens": opus_input_tokens,
                "opus_output_tokens": opus_output_tokens,
                "rounds": round_num,
                "tools_called": tools_called,
            }
            logger.info(
                f"Claude usage (stream): {total_input_tokens} input tokens, "
                f"{total_output_tokens} output tokens, "
                f"{round_num} round(s), models={model_log}, "
                f"tools={tools_called or 'none'}"
            )
            yield {"__usage": usage}
            return

        non_terminal = [t for t in tool_uses if t.name != _TOOL_RETURN_RESULTS]
        # _execute_non_terminal_tools mutates valid_titles and tools_called in-place.
        if not FORCE_FAST_MODEL:
            current_model = CLAUDE_MODEL
        messages.extend(
            _execute_non_terminal_tools(non_terminal, final.content, valid_titles, tools_called)
        )
        # Rebuild valid_lower so the streaming filter in the next round reflects
        # any filmography titles added to valid_titles by get_filmography above.
        valid_lower = {t.lower() for t in valid_titles}

    model_log = _format_model_log(models_used)
    logger.warning(f"Claude agent loop exhausted after {len(models_used)} rounds without return_results — returning no results. models={model_log}")
    yield {"__usage": {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "haiku_input_tokens": haiku_input_tokens,
        "haiku_output_tokens": haiku_output_tokens,
        "opus_input_tokens": opus_input_tokens,
        "opus_output_tokens": opus_output_tokens,
        "rounds": len(models_used),
        "tools_called": tools_called,
    }}


if __name__ == "__main__":
    message = input("Enter a message for Claude: ")
    print(prompt_claude(message))
