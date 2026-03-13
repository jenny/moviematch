from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

_client = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def prompt_claude(message: str) -> str:
    response = get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text


RERANK_TOOL = {
    "name": "return_results",
    "description": "Return the reranked and filtered movie results",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["title", "explanation"]
                }
            }
        },
        "required": ["results"]
    }
}


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    candidate_text = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in candidates
    )
    prompt = f"""You are a movie recommendation assistant. A user searched for: "{query}"

Here are candidate movies retrieved by semantic search:

{candidate_text}

Your task:
1. Filter out any movies that are not relevant to the query
2. Rerank the remaining movies from most to least relevant
3. For each movie you include, write a brief explanation of why it matches the query

Only include movies that are genuinely relevant. If a movie is a poor match, exclude it."""

    response = get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        tools=[RERANK_TOOL],
        tool_choice={"type": "tool", "name": "return_results"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].input.get("results", [])


if __name__ == "__main__":
    message = input("Enter a message for Claude: ")
    print(prompt_claude(message))
