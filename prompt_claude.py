import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
if not CLAUDE_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")

client = Anthropic(api_key=CLAUDE_KEY)

def prompt_claude(message):
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
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

def rerank(query, results):
    candidates = "\n\n".join(
        f"Title: {r['title']}\n{r['document']}"
        for r in results
    )

    prompt = f"""You are a movie recommendation assistant. A user searched for: "{query}"

Here are candidate movies retrieved by semantic search:

{candidates}

Your task:
1. Filter out any movies that are not relevant to the query
2. Rerank the remaining movies from most to least relevant
3. For each movie you include, write a brief explanation of why it matches the query

Only include movies that are genuinely relevant. If a movie is a poor match, exclude it."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=[RERANK_TOOL],
        tool_choice={"type": "tool", "name": "return_results"},
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].input.get("results", [])

if __name__ == "__main__":
    message = input("Enter a message for Claude: ")
    print(prompt_claude(message))
