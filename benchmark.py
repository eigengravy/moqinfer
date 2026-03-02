#!/usr/bin/env python3
"""MoQ vs REST/SSE benchmark.

Runs the same agentic tool-calling workload through both transports
and compares latency, throughput, and tool-call round-trip times.

Usage:
    python benchmark.py              # Both transports (cold start each)
    python benchmark.py --moq-only   # MoQ only
    python benchmark.py --rest-only  # REST only
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time

from moqinfer.backend import AgentBackend
from moqinfer.metrics import BenchmarkResult, RequestMetrics, print_comparison, print_result
from moqinfer.rest_backend import RestBackend

# ── Configuration ───────────────────────────────────────────────────────

RELAY_PORT = 4443
RELAY_URL = f"https://localhost:{RELAY_PORT}/"
REST_PORT = 8000
REST_URL = f"http://localhost:{REST_PORT}"
RELAY_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "moq", "target", "release", "moq-relay"
)

# ── Deterministic sampling ─────────────────────────────────────────────

SAMPLING_PARAMS = {"temperature": 0, "seed": 42}

# ── Workload definition ────────────────────────────────────────────────

PROMPTS = [
    # Single tool: weather lookup (1 round each)
    "What's the current weather in San Francisco?",
    "Tell me the temperature in Tokyo right now.",
    "Is it raining in London today?",
    "What's the weather like in New York City?",
    "How's the weather in Paris right now?",
    # Chained tools: weather → temperature conversion (2 rounds each)
    "What's the temperature in Tokyo? Convert it to Fahrenheit.",
    "Get the weather in London and convert the temperature to Fahrenheit.",
    "What's the temperature in Berlin? I need it in Fahrenheit.",
    # Multi-city: should trigger parallel tool calls (1-2 rounds)
    "Compare the weather in San Francisco and Tokyo.",
    "What's the weather in both London and New York?",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "Convert a temperature value between Celsius and Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Temperature value to convert",
                    },
                    "from_unit": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "Source unit",
                    },
                    "to_unit": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "Target unit",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

WEATHER_DATA = {
    "san francisco": "72°F, sunny with light breeze",
    "tokyo": "18°C, partly cloudy",
    "london": "12°C, light rain",
    "new york": "65°F, clear skies",
    "paris": "15°C, overcast",
    "berlin": "10°C, windy and cloudy",
}


def tool_executor(name: str, arguments: dict) -> str:
    """Mock tool executor — returns canned weather/conversion data."""
    if name == "get_weather":
        city = arguments.get("city", "").lower()
        for key, value in WEATHER_DATA.items():
            if key in city:
                return value
        return "25°C, clear skies"
    if name == "convert_temperature":
        value = float(arguments.get("value", 0))
        from_unit = arguments.get("from_unit", "C")
        to_unit = arguments.get("to_unit", "F")
        if from_unit == "C" and to_unit == "F":
            return f"{value * 9 / 5 + 32:.1f}°F"
        if from_unit == "F" and to_unit == "C":
            return f"{(value - 32) * 5 / 9:.1f}°C"
        return f"{value}°{to_unit}"
    return f"Unknown tool: {name}"


# ── Process management ──────────────────────────────────────────────────


def start_process(cmd, env, label):
    """Start a subprocess and drain its merged stdout+stderr in a background thread.

    Returns (proc, lines, marker_events, lock).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    lines: list[str] = []
    marker_events: dict[str, threading.Event] = {}
    lock = threading.Lock()

    def drain():
        for raw_line in proc.stdout:
            text = raw_line.decode(errors="replace").rstrip()
            with lock:
                lines.append(text)
                for marker, event in marker_events.items():
                    if marker in text:
                        event.set()
            print(f"  [{label}] {text}", flush=True)

    threading.Thread(target=drain, daemon=True).start()
    return proc, lines, marker_events, lock


async def wait_for_marker(marker_events, lock, lines, marker, timeout=300):
    """Block until *marker* appears in the process output."""
    event = threading.Event()
    with lock:
        for line in lines:
            if marker in line:
                return True
        marker_events[marker] = event

    loop = asyncio.get_event_loop()
    found = await loop.run_in_executor(
        None, lambda: event.wait(timeout=timeout)
    )
    with lock:
        marker_events.pop(marker, None)
    if not found:
        raise TimeoutError(f"Timeout waiting for '{marker}' after {timeout}s")
    return True


def kill_procs(*procs):
    """SIGTERM → wait → SIGKILL for each process."""
    for proc in procs:
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in procs:
        if proc:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


# ── Instrumented chat ───────────────────────────────────────────────────


async def instrumented_chat(
    backend,
    messages,
    *,
    tools,
    tool_exec,
    max_tokens=512,
    max_rounds=10,
    sampling_params=None,
):
    """Run a multi-turn chat and return per-request timing metrics.

    Works with any backend that exposes ``chat_stream()`` with the standard
    frame protocol (start / token / tool_calls / done).
    """
    metrics = RequestMetrics(
        connect_ms=0,
        ttft_ms=0,
        completion_ms=0,
        tool_rounds=0,
        tool_rtt_ms=[],
        total_tokens=0,
    )

    messages = list(messages)
    t_start = time.perf_counter()
    first_token_seen = False

    for _round in range(max_rounds):
        t_turn_start = time.perf_counter()
        text = ""
        round_tool_calls: list[dict] = []
        finish_reason = ""
        turn_first_token = False

        async for frame in backend.chat_stream(
            messages, tools=tools, max_tokens=max_tokens,
            sampling_params=sampling_params,
        ):
            if frame["type"] == "token":
                metrics.total_tokens += 1
                text += frame["text"]
                if not first_token_seen:
                    metrics.ttft_ms = (time.perf_counter() - t_start) * 1000
                    first_token_seen = True
                if not turn_first_token and _round > 0:
                    # Tool round-trip: time from sending tool result → next first token
                    metrics.tool_rtt_ms.append(
                        (time.perf_counter() - t_turn_start) * 1000
                    )
                    turn_first_token = True
            elif frame["type"] == "tool_calls":
                round_tool_calls = frame.get("tool_calls", [])
            elif frame["type"] == "done":
                finish_reason = frame.get("finish_reason", "unknown")

        if finish_reason != "tool_calls" or not tool_exec:
            break

        metrics.tool_rounds += 1

        # Append assistant message with tool calls
        messages.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in round_tool_calls
                ],
            }
        )

        # Execute each tool call
        for tc in round_tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "")
            arguments = json.loads(raw_args) if raw_args else {}
            if asyncio.iscoroutinefunction(tool_exec):
                tool_result = await tool_exec(name, arguments)
            else:
                tool_result = tool_exec(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(tool_result),
                }
            )

    metrics.completion_ms = (time.perf_counter() - t_start) * 1000
    return metrics


# ── Benchmark runner ────────────────────────────────────────────────────


async def benchmark_transport(
    transport_name,
    backend_factory,
    *,
    num_backends=5,
    users_per_backend=5,
):
    """Create backends, run concurrent workload, collect metrics."""
    total_users = num_backends * users_per_backend
    # Cycle prompts to fill all user slots
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(total_users)]

    print(f"\n{'=' * 60}")
    print(f"  Benchmarking {transport_name.upper()}")
    print(
        f"  {num_backends} backend(s) x {users_per_backend} user(s) "
        f"= {total_users} concurrent request(s)"
    )
    print(f"  sampling: temperature={SAMPLING_PARAMS['temperature']}, "
          f"seed={SAMPLING_PARAMS['seed']}")
    print(f"{'=' * 60}")

    # Create backends
    print(f"  Creating {num_backends} backend(s)...")
    t_connect = time.perf_counter()
    backends = []
    for _ in range(num_backends):
        backend = await backend_factory()
        backends.append(backend)
    connect_ms = (time.perf_counter() - t_connect) * 1000
    print(f"  Connected in {connect_ms:.1f}ms")

    # Run concurrent requests
    print(f"  Running {total_users} concurrent requests...")
    t_wall = time.perf_counter()

    async def run_user(idx, prompt):
        backend = backends[idx % len(backends)]
        messages = [{"role": "user", "content": prompt}]
        return await instrumented_chat(
            backend,
            messages,
            tools=TOOLS,
            tool_exec=tool_executor,
            max_tokens=512,
            max_rounds=10,
            sampling_params=SAMPLING_PARAMS,
        )

    tasks = [run_user(i, p) for i, p in enumerate(prompts)]
    request_metrics = await asyncio.gather(*tasks)

    wall_time_ms = (time.perf_counter() - t_wall) * 1000

    # Set amortized connect time
    for m in request_metrics:
        m.connect_ms = connect_ms / num_backends

    # Cleanup
    for backend in backends:
        await backend.close()

    result = BenchmarkResult(
        transport=transport_name,
        num_backends=num_backends,
        users_per_backend=users_per_backend,
        requests=list(request_metrics),
        wall_time_ms=wall_time_ms,
    )
    print_result(result)
    return result


# ── Server lifecycle ────────────────────────────────────────────────────


async def run_moq_benchmark(num_backends=5, users_per_backend=5):
    """Start relay + MoQ inference server, benchmark, kill."""
    print(f"\n{'=' * 60}")
    print("  Phase: MoQ (cold start)")
    print(f"{'=' * 60}")

    # Start relay
    print("  Starting moq-relay...")
    relay_env = {**os.environ, "RUST_LOG": "info"}
    relay_proc, _, _, _ = start_process(
        [
            RELAY_BIN,
            "--server-bind", f"[::]:{RELAY_PORT}",
            "--tls-generate", "localhost",
            "--auth-public", "",
        ],
        relay_env,
        "relay",
    )
    await asyncio.sleep(2)
    if relay_proc.poll() is not None:
        raise RuntimeError("moq-relay exited early")
    print("  moq-relay started")

    # Start inference server
    print("  Starting MoQ inference server (loading model)...")
    server_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    server_proc, s_lines, s_markers, s_lock = start_process(
        [sys.executable, "-m", "moqinfer.server"],
        server_env,
        "moq-server",
    )
    await wait_for_marker(s_markers, s_lock, s_lines, "Connected to relay")
    print("  MoQ server ready")

    try:
        result = await benchmark_transport(
            "moq",
            lambda: AgentBackend.connect(RELAY_URL),
            num_backends=num_backends,
            users_per_backend=users_per_backend,
        )
    finally:
        print("  Stopping MoQ servers...")
        kill_procs(server_proc, relay_proc)
        print("  MoQ servers stopped")

    return result


async def run_rest_benchmark(num_backends=5, users_per_backend=5):
    """Start custom REST/SSE server (same engine as MoQ), benchmark, kill."""
    print(f"\n{'=' * 60}")
    print("  Phase: REST (cold start)")
    print(f"{'=' * 60}")

    print("  Starting REST/SSE inference server (loading model)...")
    server_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    server_proc, s_lines, s_markers, s_lock = start_process(
        [sys.executable, "-m", "moqinfer.rest_server"],
        server_env,
        "rest-server",
    )

    print("  Waiting for REST server...")
    await wait_for_marker(s_markers, s_lock, s_lines, "REST server ready")
    print("  REST server ready")

    try:
        result = await benchmark_transport(
            "rest",
            lambda: RestBackend.connect(REST_URL),
            num_backends=num_backends,
            users_per_backend=users_per_backend,
        )
    finally:
        print("  Stopping REST server...")
        kill_procs(server_proc)
        print("  REST server stopped")

    return result


# ── CLI entry point ─────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="MoQ vs REST benchmark — agentic tool-calling workload"
    )
    parser.add_argument("--moq-only", action="store_true", help="MoQ benchmark only")
    parser.add_argument("--rest-only", action="store_true", help="REST benchmark only")
    parser.add_argument(
        "--backends", type=int, default=5, help="Number of backend instances (default: 5)"
    )
    parser.add_argument(
        "--users", type=int, default=5, help="Concurrent users per backend (default: 5)"
    )
    args = parser.parse_args()

    # Validate
    if not args.rest_only and not os.path.exists(RELAY_BIN):
        print(f"ERROR: moq-relay not found at {RELAY_BIN}")
        print("Build with: cd moq && cargo build --release -p moq-relay")
        sys.exit(1)

    moq_result = None
    rest_result = None

    # Phase 1: MoQ
    if not args.rest_only:
        moq_result = await run_moq_benchmark(args.backends, args.users)

    # Phase 2: REST (cold start — model reloaded from scratch)
    if not args.moq_only:
        rest_result = await run_rest_benchmark(args.backends, args.users)

    # Phase 3: Comparison
    if moq_result and rest_result:
        print_comparison(moq_result, rest_result)
    elif moq_result:
        print("\n(Run without --moq-only to compare against REST)")
    elif rest_result:
        print("\n(Run without --rest-only to compare against MoQ)")


if __name__ == "__main__":
    asyncio.run(main())
