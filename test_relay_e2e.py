"""End-to-end test: MoQ client <-> relay <-> server over native MoQ transport.

Starts a moq-relay subprocess, then connects a mock inference server
and a client through the relay to verify the full fan-out architecture.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid

import moq_py

RELAY_PORT = "14443"
RELAY_URL = f"https://localhost:{RELAY_PORT}/"
BROADCAST_NAME = "inference"
RELAY_BIN = os.path.join(os.path.dirname(__file__), "moq", "target", "release", "moq-relay")


async def server_side(relay_url: str, server_ready: asyncio.Event):
    """Connect to relay as inference server, handle one mock request."""
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

    # Signal that server is ready
    server_ready.set()

    # Wait for request broadcast
    consumer = consume_origin.consume()
    req_bc = None
    while True:
        announce = await consumer.announced()
        if announce is None:
            break
        path, bc = announce
        if bc is None:
            continue
        if path == BROADCAST_NAME:
            continue  # skip own echo
        print(f"[server] Request broadcast: {path[:8]}...")
        req_bc = bc
        break

    # Subscribe to "data" track on the request broadcast
    data_track = req_bc.subscribe_track("data")
    group = await data_track.next_group()
    frame = await group.read_frame()
    request = json.loads(frame)
    rid = request["request_id"]
    prompt = request.get("prompt", "")
    print(f"[server] Request {rid[:8]}...: {prompt!r}")

    # Response is a new group on the shared response track
    g = resp_track.append_group()
    g.write_frame(
        json.dumps({"type": "start", "request_id": rid, "model": "mock"}).encode()
    )
    for word in "The meaning of life is MoQ relay transport.".split():
        g.write_frame(json.dumps({"type": "token", "text": word + " "}).encode())
        await asyncio.sleep(0.005)
    g.write_frame(
        json.dumps({"type": "done", "finish_reason": "stop"}).encode()
    )
    g.finish()
    print("[server] Response streamed")

    await asyncio.sleep(1.0)
    del session


async def client_side(relay_url: str) -> str:
    """Connect to relay, send request, read streaming tokens."""
    req_origin = moq_py.MoqOrigin.produce()
    resp_origin = moq_py.MoqOrigin.produce()

    config = moq_py.MoqClientConfig(tls_disable_verify=True)
    client = moq_py.MoqClient.create(config)
    client.with_publish(req_origin.consume())
    client.with_consume(resp_origin)
    session = await client.connect(relay_url)
    print("[client] Connected to relay")

    # Find server's inference broadcast
    resp_consumer = resp_origin.consume()
    server_bc = None
    while True:
        announce = await resp_consumer.announced()
        if announce is None:
            raise RuntimeError("No announcements")
        path, bc_announced = announce
        if bc_announced is None:
            continue
        if path == BROADCAST_NAME:
            server_bc = bc_announced
            break

    # Subscribe to server's data track for responses
    resp_track = server_bc.subscribe_track("data")

    # Create request broadcast (one broadcast per request)
    rid = str(uuid.uuid4())
    bc = req_origin.create_broadcast(rid)
    bc.dynamic()
    track = bc.create_track("data")
    g = track.append_group()
    g.write_frame(
        json.dumps(
            {"request_id": rid, "prompt": "What is the meaning of life?"}
        ).encode()
    )
    g.finish()
    print(f"[client] Sent request {rid[:8]}...")

    # Read response group
    resp_group = await resp_track.next_group()
    full_text = ""
    while True:
        frame = await resp_group.read_frame()
        if frame is None:
            break
        msg = json.loads(frame)
        if msg["type"] == "start":
            print(f"[client] << stream start (model={msg['model']})")
        elif msg["type"] == "token":
            full_text += msg["text"]
        elif msg["type"] == "done":
            print(f"[client] << stream done ({msg['finish_reason']})")

    print(f"[client] Full response: {full_text!r}")
    del session
    return full_text


async def main():
    # 1. Start moq-relay subprocess
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

    # Wait for relay to start accepting connections
    await asyncio.sleep(1.0)
    if relay_proc.poll() is not None:
        stderr = relay_proc.stderr.read().decode()
        print(f"ERROR: moq-relay exited early:\n{stderr}")
        sys.exit(1)
    print("moq-relay started")

    try:
        # 2. Run server and client through the relay
        server_ready = asyncio.Event()
        server_task = asyncio.create_task(server_side(RELAY_URL, server_ready))

        # Wait for server to connect and publish before starting client
        await asyncio.wait_for(server_ready.wait(), timeout=5.0)
        await asyncio.sleep(0.1)  # let relay propagate the broadcast announcement

        text = await asyncio.wait_for(client_side(RELAY_URL), timeout=10.0)
        await server_task

        # 3. Verify
        expected = "The meaning of life is MoQ relay transport. "
        assert text == expected, f"Expected {expected!r}, got {text!r}"
        print("\nEnd-to-end MoQ relay transport test passed!")

    finally:
        # Clean up relay
        relay_proc.send_signal(signal.SIGTERM)
        try:
            relay_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            relay_proc.kill()
            relay_proc.wait()
        print("moq-relay stopped")


if __name__ == "__main__":
    asyncio.run(main())
