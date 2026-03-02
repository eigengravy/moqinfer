"""Real LLM test: Multi-backend, multi-user, multi-turn tool calling on Metal GPU.

Connects to already-running moq-relay + vLLM server on port 4443.
Fires concurrent tool-calling sessions across multiple backends.
"""

import asyncio
import json
import time

from moqinfer.backend import AgentBackend

RELAY_URL = "https://localhost:4443/"

NUM_BACKENDS = 2
USERS_PER_BACKEND = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression and return the numeric result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like 2+3 or 10*5",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

# Each backend/user gets a unique math problem: (expression_str, expected_result)
PROBLEMS = [
    [("23 * 47", 1081), ("99 + 301", 400), ("144 / 12", 12)],
    [("256 / 8", 32), ("15 * 13", 195), ("1000 - 373", 627)],
]


def make_executor(backend_id, user_id):
    def executor(name, arguments):
        expr = arguments["expression"]
        result = eval(expr)
        print(f"  [B{backend_id}U{user_id}] tool {name}({expr}) -> {result}")
        return str(result)

    return executor


async def run_user(backend, backend_id, user_id, problem, expected):
    """One user session: ask a math question, model calls calculator, returns answer."""
    result = await backend.chat(
        messages=[
            {"role": "user", "content": f"What is {problem}? Use the calculator tool."}
        ],
        tools=TOOLS,
        tool_executor=make_executor(backend_id, user_id),
        max_tokens=256,
    )
    assert result.finish_reason == "stop", (
        f"B{backend_id}U{user_id}: expected stop, got {result.finish_reason!r}"
    )
    expected_str = str(int(expected)) if expected == int(expected) else str(expected)
    assert expected_str in result.text, (
        f"B{backend_id}U{user_id}: expected {expected_str} in response, "
        f"got {result.text!r}"
    )
    return result


async def run_backend(backend_id):
    """One backend: connect, fire concurrent users with tool calls."""
    backend = await AgentBackend.connect(RELAY_URL)
    print(f"[B{backend_id}] Connected")

    user_tasks = []
    for u in range(USERS_PER_BACKEND):
        problem, expected = PROBLEMS[backend_id][u]
        user_tasks.append(run_user(backend, backend_id, u, problem, expected))

    results = await asyncio.gather(*user_tasks)

    for u, r in enumerate(results):
        text_preview = r.text[:80].replace("\n", " ")
        print(
            f"[B{backend_id}U{u}] OK | {r.finish_reason} | "
            f"{len(r.messages)} msgs | {text_preview}"
        )

    await backend.close()
    return results


async def main():
    total = NUM_BACKENDS * USERS_PER_BACKEND
    print(
        f"=== {NUM_BACKENDS} backends x {USERS_PER_BACKEND} users = "
        f"{total} concurrent tool-calling sessions (Metal GPU) ===\n"
    )
    t0 = time.time()

    all_results = await asyncio.wait_for(
        asyncio.gather(*[run_backend(b) for b in range(NUM_BACKENDS)]),
        timeout=600.0,
    )

    total_done = sum(len(r) for r in all_results)
    elapsed = time.time() - t0
    print(
        f"\nAll {total_done} tool-calling sessions completed in {elapsed:.1f}s"
    )
    print(
        f"{NUM_BACKENDS} backends x {USERS_PER_BACKEND} users, "
        f"multi-turn tool calls, real Qwen3-4B on Metal GPU"
    )


if __name__ == "__main__":
    asyncio.run(main())
