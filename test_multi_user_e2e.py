"""End-to-end test: Multiple concurrent users through AgentBackend via relay.

Starts a moq-relay subprocess, connects a mock inference server,
then uses AgentBackend to send multiple concurrent requests and
verify they all complete correctly over a single MoQ connection.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys

import moq_py
from moqinfer.backend import AgentBackend

RELAY_PORT = "14444"
RELAY_URL = f"https://localhost:{RELAY_PORT}/"
BROADCAST_NAME = "inference"
RELAY_BIN = os.path.join(os.path.dirname(__file__), "moq", "target", "release", "moq-relay")


async def mock_handle_inference(resp_track, req):
    """Handle one inference request — echoes prompt words as tokens."""
    request_id = req["request_id"]
    prompt = req.get("prompt", "")

    # Each response is a new group on the shared response track
    g = resp_track.append_group()
    g.write_frame(
        json.dumps({"type": "start", "request_id": request_id, "model": "mock"}).encode()
    )
    for word in prompt.split():
        g.write_frame(json.dumps({"type": "token", "text": word + " "}).encode())
        await asyncio.sleep(0.005)
    g.write_frame(
        json.dumps({"type": "done", "finish_reason": "stop"}).encode()
    )
    g.finish()


async def server_side(relay_url: str, server_ready: asyncio.Event, num_requests: int):
    """Mock inference server that handles num_requests requests then exits."""
    publish_origin = moq_py.MoqOrigin.produce()
    consume_origin = moq_py.MoqOrigin.produce()

    config = moq_py.MoqClientConfig(tls_disable_verify=True)
    client = moq_py.MoqClient.create(config)
    client.with_publish(publish_origin.consume())
    client.with_consume(consume_origin)
    session = await client.connect(relay_url)
    print("[server] Connected to relay")

    # Single broadcast, single track for all responses
    inference_bc = publish_origin.create_broadcast(BROADCAST_NAME)
    inference_bc.dynamic()
    resp_track = inference_bc.create_track("data")

    server_ready.set()

    # Process request broadcasts (one broadcast per request)
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
        # Each announced broadcast is a single request
        data_track = bc.subscribe_track("data")
        group = await data_track.next_group()
        if group is None:
            continue
        frame_data = await group.read_frame()
        if frame_data is None:
            continue
        req = json.loads(frame_data)
        print(f"[server] Request {req['request_id'][:8]}...")
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

    print(f"Starting moq-relay on port {RELAY_PORT}...")
    relay_proc = subprocess.Popen(
        [
            RELAY_BIN,
            "--server-bind", f"[::]:{RELAY_PORT}",
            "--tls-generate", "localhost",
            "--auth-public", "",
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
        NUM_USERS = 5
        prompts = [f"User {i} says hello world" for i in range(NUM_USERS)]

        # Start mock server
        server_ready = asyncio.Event()
        server_task = asyncio.create_task(
            server_side(RELAY_URL, server_ready, NUM_USERS)
        )
        await asyncio.wait_for(server_ready.wait(), timeout=5.0)
        await asyncio.sleep(0.2)

        # Connect ONE AgentBackend (single MoQ connection)
        backend = await AgentBackend.connect(RELAY_URL)
        print(f"[test] AgentBackend connected (single connection)")

        # Fire concurrent requests from multiple "users"
        print(f"[test] Sending {NUM_USERS} concurrent requests...")
        results = await asyncio.wait_for(
            asyncio.gather(*[backend.generate(prompt) for prompt in prompts]),
            timeout=15.0,
        )

        # Verify all results
        for i, (result, prompt) in enumerate(zip(results, prompts)):
            expected = " ".join(prompt.split()) + " "
            assert result.text == expected, (
                f"User {i}: expected {expected!r}, got {result.text!r}"
            )
            assert result.model == "mock"
            assert result.finish_reason == "stop"
            print(f"[test] User {i}: OK — {result.text!r}")

        await backend.close()
        await server_task

        print(f"\nMulti-user e2e test passed! ({NUM_USERS} concurrent users, 1 MoQ connection)")

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
