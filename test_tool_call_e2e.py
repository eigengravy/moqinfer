"""End-to-end test: Tool calling through AgentBackend via relay.

Starts a moq-relay subprocess, connects a mock inference server that
simulates tool call responses, then uses AgentBackend to verify:
1. Single-turn tool call (no executor — returns tool_calls in result)
2. Multi-turn automatic tool loop (with executor)
3. Backward compat — generate() still works
"""

import asyncio
import json
import os
import signal
import subprocess
import sys

import moq_py
from moqinfer.backend import AgentBackend

RELAY_PORT = "14446"
RELAY_URL = f"https://localhost:{RELAY_PORT}/"
BROADCAST_NAME = "inference"
RELAY_BIN = os.path.join(
    os.path.dirname(__file__), "moq", "target", "release", "moq-relay"
)


async def mock_handle_inference(resp_track, req):
    """Handle one inference request — simulates tool calling behavior.

    - If tools are present and no tool results in messages yet, respond with a tool call.
    - If messages contain a tool result, respond with a text answer using the result.
    - If no tools (plain generate), echo prompt words as tokens.
    """
    request_id = req["request_id"]
    messages = req.get("messages")
    tools = req.get("tools")

    g = resp_track.append_group()
    g.write_frame(
        json.dumps(
            {"type": "start", "request_id": request_id, "model": "mock"}
        ).encode()
    )

    if messages and tools:
        # Check if there's already a tool result in messages
        has_tool_result = any(m["role"] == "tool" for m in messages)

        if not has_tool_result:
            # First turn: respond with a tool call
            tool_func = tools[0]["function"]
            tool_call_text = (
                f'<tool_call>{{"name": "{tool_func["name"]}", '
                f'"arguments": {{"city": "San Francisco"}}}}</tool_call>'
            )
            g.write_frame(
                json.dumps({"type": "token", "text": tool_call_text}).encode()
            )
            # Parse and send tool_calls frame
            g.write_frame(
                json.dumps(
                    {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "id": "call_mock001",
                                "type": "function",
                                "function": {
                                    "name": tool_func["name"],
                                    "arguments": json.dumps(
                                        {"city": "San Francisco"}
                                    ),
                                },
                            }
                        ],
                    }
                ).encode()
            )
            g.write_frame(
                json.dumps(
                    {"type": "done", "finish_reason": "tool_calls"}
                ).encode()
            )
        else:
            # Second turn: use tool result in response
            tool_result_msg = next(
                m for m in messages if m["role"] == "tool"
            )
            answer = f"The weather is {tool_result_msg['content']}."
            g.write_frame(
                json.dumps({"type": "token", "text": answer}).encode()
            )
            g.write_frame(
                json.dumps({"type": "done", "finish_reason": "stop"}).encode()
            )
    else:
        # Plain generate — echo prompt words
        prompt = req.get("prompt", "")
        for word in prompt.split():
            g.write_frame(
                json.dumps({"type": "token", "text": word + " "}).encode()
            )
            await asyncio.sleep(0.005)
        g.write_frame(
            json.dumps({"type": "done", "finish_reason": "stop"}).encode()
        )

    g.finish()


async def server_side(
    relay_url: str, server_ready: asyncio.Event, num_requests: int
):
    """Mock inference server that handles num_requests total requests."""
    publish_origin = moq_py.MoqOrigin.produce()
    consume_origin = moq_py.MoqOrigin.produce()

    config = moq_py.MoqClientConfig(tls_disable_verify=True)
    client = moq_py.MoqClient.create(config)
    client.with_publish(publish_origin.consume())
    client.with_consume(consume_origin)
    session = await client.connect(relay_url)
    print("[server] Connected to relay")

    inference_bc = publish_origin.create_broadcast(BROADCAST_NAME)
    inference_bc.dynamic()
    resp_track = inference_bc.create_track("data")

    server_ready.set()

    consumer = consume_origin.consume()
    tasks = []
    handled = 0
    while handled < num_requests:
        announce = await consumer.announced()
        if announce is None:
            break
        path, bc = announce
        if bc is None:
            continue
        if path == BROADCAST_NAME:
            continue
        data_track = bc.subscribe_track("data")
        group = await data_track.next_group()
        if group is None:
            continue
        frame_data = await group.read_frame()
        if frame_data is None:
            continue
        req = json.loads(frame_data)
        tasks.append(asyncio.create_task(mock_handle_inference(resp_track, req)))
        handled += 1

    if tasks:
        await asyncio.gather(*tasks)
    print(f"[server] Handled {len(tasks)} requests")

    await asyncio.sleep(1.0)
    del session


async def main():
    if not os.path.exists(RELAY_BIN):
        print(f"ERROR: moq-relay binary not found at {RELAY_BIN}")
        print("Build it with: cd moq && cargo build --release -p moq-relay")
        sys.exit(1)

    # Total requests: 1 (backward compat) + 1 (single-turn tool) + 2 (auto loop: turn1 + turn2)
    NUM_REQUESTS = 4

    print(f"Starting moq-relay on port {RELAY_PORT}...")
    relay_proc = subprocess.Popen(
        [
            RELAY_BIN,
            "--server-bind",
            f"[::]:{RELAY_PORT}",
            "--tls-generate",
            "localhost",
            "--auth-public",
            "",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "RUST_LOG": "info"},
    )
    await asyncio.sleep(1.0)
    if relay_proc.poll() is not None:
        stderr = relay_proc.stderr.read().decode()
        print(f"ERROR: moq-relay exited early:\n{stderr}")
        sys.exit(1)
    print("moq-relay started")

    try:
        server_ready = asyncio.Event()
        server_task = asyncio.create_task(
            server_side(RELAY_URL, server_ready, NUM_REQUESTS)
        )
        await asyncio.wait_for(server_ready.wait(), timeout=5.0)
        await asyncio.sleep(0.2)

        backend = await AgentBackend.connect(RELAY_URL)
        print("[test] AgentBackend connected")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        # ── Test 1: Backward compat — generate() still works ──
        print("\n[test] Test 1: Backward compat (generate)")
        result = await asyncio.wait_for(
            backend.generate("hello world test"), timeout=10.0
        )
        expected = "hello world test "
        assert result.text == expected, f"Expected {expected!r}, got {result.text!r}"
        assert result.finish_reason == "stop"
        print(f"  OK — {result.text!r}")

        # ── Test 2: Single-turn tool call (no executor) ──
        print("\n[test] Test 2: Single-turn tool call (no executor)")
        result = await asyncio.wait_for(
            backend.chat(
                messages=[
                    {"role": "user", "content": "What's the weather in SF?"}
                ],
                tools=tools,
                # No tool_executor — should return with tool_calls
            ),
            timeout=10.0,
        )
        assert result.finish_reason == "tool_calls", (
            f"Expected finish_reason='tool_calls', got {result.finish_reason!r}"
        )
        assert len(result.tool_calls) == 1, (
            f"Expected 1 tool call, got {len(result.tool_calls)}"
        )
        tc = result.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert args == {"city": "San Francisco"}
        print(f"  OK — tool_call: {tc['function']['name']}({args})")

        # ── Test 3: Multi-turn auto tool loop (with executor) ──
        print("\n[test] Test 3: Multi-turn auto tool loop (with executor)")

        async def mock_tool_executor(name, arguments):
            assert name == "get_weather"
            return "72°F and sunny"

        result = await asyncio.wait_for(
            backend.chat(
                messages=[
                    {"role": "user", "content": "What's the weather in SF?"}
                ],
                tools=tools,
                tool_executor=mock_tool_executor,
            ),
            timeout=10.0,
        )
        assert result.finish_reason == "stop", (
            f"Expected finish_reason='stop', got {result.finish_reason!r}"
        )
        assert "72°F and sunny" in result.text, (
            f"Expected tool result in response, got {result.text!r}"
        )
        # Verify conversation history: user -> assistant(tool_calls) -> tool(result)
        assert len(result.messages) == 3, (
            f"Expected 3 messages in history, got {len(result.messages)}"
        )
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"
        assert "tool_calls" in result.messages[1]
        assert result.messages[2]["role"] == "tool"
        assert result.messages[2]["content"] == "72°F and sunny"
        print(f"  OK — final response: {result.text!r}")
        print(f"  OK — conversation history: {len(result.messages)} messages")

        await backend.close()
        await server_task

        print(
            "\nTool calling e2e test passed! "
            "(backward compat + single-turn + multi-turn auto loop)"
        )

    finally:
        relay_proc.send_signal(signal.SIGTERM)
        try:
            relay_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            relay_proc.kill()
            relay_proc.wait()
        print("moq-relay stopped")


if __name__ == "__main__":
    asyncio.run(main())
