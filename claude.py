import json
import threading

from anthropic import Anthropic, RateLimitError, InternalServerError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_FAST_MODEL, AGENT_MAX_TOOL_ROUNDS
from tmdb import search_person, get_filmography

_client = None


def get_client() -> Anthropic:
    global _client
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
            print(f"Background ingestion failed for {movie.get('title', movie['id'])}: {e}")


def execute_tool(name: str, tool_input: dict) -> dict:
    args_summary = ", ".join(f"{k}={v}" for k, v in tool_input.items())
    print(f"Tool call: {name}({args_summary})")
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
        print(f"Tool {name} failed: {e}")
        return {"error": str(e)}


def _is_transient_claude_error(exc: BaseException) -> bool:
    return isinstance(exc, (RateLimitError, InternalServerError, APIConnectionError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_transient_claude_error),
)
def _call_claude(model: str, messages: list) -> object:
    return get_client().messages.create(
        model=model,
        max_tokens=2048,
        tools=TOOLS,
        tool_choice={"type": "any"},
        messages=messages,
    )


def rerank(query: str, candidates: list[dict]) -> tuple[list[dict], dict]:
    candidate_text = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in candidates
    )
    prompt = f"""You are a movie recommendation assistant. A user searched for: "{query}"

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
    tools_called = []
    models_used = []
    current_model = CLAUDE_FAST_MODEL
    exit_reason = "exhausted"

    for round_num in range(1, AGENT_MAX_TOOL_ROUNDS + 1):
        models_used.append(current_model)
        response = _call_claude(current_model, messages)

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

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
                    "rounds": round_num,
                    "tools_called": tools_called,
                }
                print(
                    f"Claude usage: {total_input_tokens} input tokens, "
                    f"{total_output_tokens} output tokens, "
                    f"{round_num} round(s), "
                    f"models={model_log}, "
                    f"tools={tools_called or 'none'}"
                )
                return tool_use.input.get("results", []), usage

        # No return_results this round — execute non-terminal tools and continue
        tools_called.extend(t.name for t in non_terminal)
        current_model = CLAUDE_MODEL

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": t.id,
                    "content": json.dumps(execute_tool(t.name, t.input)),
                }
                for t in non_terminal
            ],
        })

    model_log = "→".join("haiku" if m == CLAUDE_FAST_MODEL else "opus" for m in models_used)
    usage = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "rounds": len(models_used),
        "tools_called": tools_called,
    }
    print(f"Claude usage: agent loop ended ({exit_reason}) after {len(models_used)} rounds, models={model_log}")
    return [], usage


if __name__ == "__main__":
    message = input("Enter a message for Claude: ")
    print(prompt_claude(message))
