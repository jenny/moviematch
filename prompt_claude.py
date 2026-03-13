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

if __name__ == "__main__":
    message = input("Enter a message for Claude: ")
    print(prompt_claude(message))
