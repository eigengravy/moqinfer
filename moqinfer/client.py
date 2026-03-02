"""MoQ inference client — CLI wrapper around AgentBackend."""

import asyncio
import sys

from moqinfer.backend import AgentBackend


async def generate(
    url: str = "https://localhost:4443/",
    prompt: str = "What is the meaning of life?",
    max_tokens: int = 512,
) -> str:
    """Send an inference request via MoQ relay and stream back the response."""
    backend = await AgentBackend.connect(url)
    try:
        full_text = ""
        async for frame in backend.generate_stream(prompt, max_tokens=max_tokens):
            if frame["type"] == "start":
                model = frame.get("model", "unknown")
                print(f"[model: {model}]", flush=True)
            elif frame["type"] == "token":
                text = frame["text"]
                full_text += text
                print(text, end="", flush=True)
            elif frame["type"] == "done":
                print(f"\n[done: {frame.get('finish_reason', '?')}]", flush=True)
        return full_text
    finally:
        await backend.close()


async def main():
    url = "https://localhost:4443/"
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the meaning of life?"

    print(f"Connecting to relay at {url}...")
    print(f"Prompt: {prompt!r}\n")

    text = await generate(url=url, prompt=prompt)
    print(f"\n--- {len(text)} chars ---")


if __name__ == "__main__":
    asyncio.run(main())
