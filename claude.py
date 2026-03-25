import json
import logging
import threading

from anthropic import Anthropic, RateLimitError, InternalServerError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_FAST_MODEL, AGENT_MAX_TOOL_ROUNDS
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


SEARCH_PERSON_TOOL = {
    "name": "search_person",
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
    "name": "get_filmography",
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
    "name": "return_results",
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


def _ingest_filmography_background(movies: list[dict]) -> None:
    from pipeline import ingest_single
    for movie in movies:
        try:
            ingest_single(movie["id"], movie.get("vote_average", 0), movie.get("vote_count", 0))
        except Exception as e:
            logger.warning(f"Background ingestion failed for {movie.get('title', movie['id'])}: {e}")


def execute_tool(name: str, tool_input: dict) -> dict:
    args_summary = ", ".join(f"{k}={v}" for k, v in tool_input.items())
    logger.debug(f"Tool call: {name}({args_summary})")
    try:
        if name == "search_person":
            return {"results": search_person(tool_input["name"])}
        elif name == "get_filmography":
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
def _call_claude(model: str, messages: list, max_tokens: int = 1024) -> object:
    return get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=messages,
    )


def _filter_results(results: list[dict], valid_titles: set[str]) -> list[dict]:
    """Drop any return_results entries whose title wasn't in the candidate or filmography set."""
    filtered = [r for r in results if r.get("title") in valid_titles]
    rejected = [r["title"] for r in results if r.get("title") not in valid_titles]
    if rejected:
        logger.warning(f"return_results validation: rejected fabricated title(s): {rejected}")
    return filtered


def rerank(query: str, candidates: list[dict]) -> tuple[list[dict], dict]:
    candidate_text = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in candidates
    )
    prompt = f"""You are a movie recommendation assistant. A user searched for:
<query>{query}</query>

Here are candidate movies retrieved by semantic search:

{candidate_text}

If the query mentions a specific director, actor, or filmmaker by name, use search_person \
to find them, then get_filmography to see their work — this ensures you can recommend their \
films even if they're not in the candidate list above.

When you have enough information, use return_results with your final recommendations:
1. Filter out candidates that don't match the query
2. Rerank from most to least relevant, including any relevant films from a filmography lookup
3. Write a brief explanation for each result

Only include movies that are genuinely relevant."""

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

    for round_num in range(1, AGENT_MAX_TOOL_ROUNDS + 1):
        models_used.append(current_model)
        response = _call_claude(current_model, messages, max_tokens=1024 if current_model == CLAUDE_FAST_MODEL else 2048)

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

        non_terminal = [t for t in tool_uses if t.name != "return_results"]

        for tool_use in tool_uses:
            if tool_use.name == "return_results":
                # Only record tools that were actually executed (non-terminal tools
                # from prior rounds); non_terminal tools in this round are not executed
                # since we're returning immediately
                model_log = "→".join(
                    "haiku" if m == CLAUDE_FAST_MODEL else "opus" for m in models_used
                )
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

        # No return_results this round — execute non-terminal tools and continue
        tools_called.extend(t.name for t in non_terminal)
        current_model = CLAUDE_MODEL

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for t in non_terminal:
            result = execute_tool(t.name, t.input)
            if t.name == "get_filmography":
                for movie in result.get("movies", []):
                    if movie.get("title"):
                        valid_titles.add(movie["title"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": t.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    model_log = "→".join("haiku" if m == CLAUDE_FAST_MODEL else "opus" for m in models_used)
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
    logger.info(f"Claude usage: agent loop ended ({exit_reason}) after {len(models_used)} rounds, models={model_log}")
    return [], usage


def _extract_result_objects(partial_json: str) -> list[str]:
    """Extract complete depth-2 {...} objects from partially-streamed JSON.
    These are the individual result objects inside {"results": [...]}."""
    objects = []
    depth = 0
    start = -1
    in_string = False
    i = 0
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
                if depth == 2:
                    start = i
            elif ch == "}":
                if depth == 2 and start != -1:
                    objects.append(partial_json[start:i + 1])
                    start = -1
                depth -= 1
        i += 1
    return objects


def rerank_stream(query: str, candidates: list[dict]):
    """Streaming version of rerank(). Yields result dicts one by one as they
    complete in Claude's return_results tool call, then yields {"__usage": {...}}."""
    candidate_text = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in candidates
    )
    prompt = f"""You are a movie recommendation assistant. A user searched for:
<query>{query}</query>

Here are candidate movies retrieved by semantic search:

{candidate_text}

If the query mentions a specific director, actor, or filmmaker by name, use search_person \
to find them, then get_filmography to see their work — this ensures you can recommend their \
films even if they're not in the candidate list above.

When you have enough information, use return_results with your final recommendations:
1. Filter out candidates that don't match the query
2. Rerank from most to least relevant, including any relevant films from a filmography lookup
3. Write a brief explanation for each result

Only include movies that are genuinely relevant."""

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

    for round_num in range(1, AGENT_MAX_TOOL_ROUNDS + 1):
        models_used.append(current_model)
        return_results_block_idx = None
        return_results_json = ""
        results_yielded = 0

        try:
            with get_client().messages.stream(
                model=current_model,
                max_tokens=1024 if current_model == CLAUDE_FAST_MODEL else 2048,
                tools=TOOLS,
                tool_choice={"type": "any"},
                messages=messages,
            ) as stream:
                for event in stream:
                    if (event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "tool_use"
                            and event.content_block.name == "return_results"):
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
                                    if result["title"] in valid_titles:
                                        yield result
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
        return_results_call = next((t for t in tool_uses if t.name == "return_results"), None)

        if return_results_call:
            # Safety net: yield any valid results the streaming parser missed
            for r in _filter_results(return_results_call.input.get("results", []), valid_titles)[results_yielded:]:
                yield r
            model_log = "→".join("haiku" if m == CLAUDE_FAST_MODEL else "opus" for m in models_used)
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

        non_terminal = [t for t in tool_uses if t.name != "return_results"]
        tools_called.extend(t.name for t in non_terminal)
        current_model = CLAUDE_MODEL

        messages.append({"role": "assistant", "content": final.content})
        tool_results = []
        for t in non_terminal:
            result = execute_tool(t.name, t.input)
            if t.name == "get_filmography":
                for movie in result.get("movies", []):
                    if movie.get("title"):
                        valid_titles.add(movie["title"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": t.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    model_log = "→".join("haiku" if m == CLAUDE_FAST_MODEL else "opus" for m in models_used)
    logger.info(f"Claude usage (stream): agent loop exhausted after {len(models_used)} rounds, models={model_log}")
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
