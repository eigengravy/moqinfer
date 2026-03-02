"""End-to-end test: MoQ client <-> server over native MoQ transport.

Starts a MoQ server with mock inference, then connects a client
that sends a request and reads the streaming token response.
"""

import asyncio
import json
import uuid

import moq_py


async def server_side(server: moq_py.MoqServer):
    """Accept one connection and serve one mock inference request."""
    req = await server.accept()
    print("[server] Connection received")

    # Per-connection origins
    resp_origin = moq_py.MoqOrigin.produce()
    req_origin = moq_py.MoqOrigin.produce()

    req.with_publish(resp_origin.consume())
    req.with_consume(req_origin)
    session = await req.ok()
    print("[server] Handshake complete")

    # Response broadcast with a single "data" track for all responses
    resp_broadcast = resp_origin.create_broadcast("")
    resp_broadcast.dynamic()
    resp_track = resp_broadcast.create_track("data")

    # Wait for client's request broadcast
    req_consumer = req_origin.consume()
    announce = await req_consumer.announced()
    _path, req_bc = announce
    data_track = req_bc.subscribe_track("data")

    # Read inference request
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
    for word in "The meaning of life is MoQ transport.".split():
        g.write_frame(json.dumps({"type": "token", "text": word + " "}).encode())
        await asyncio.sleep(0.005)  # simulate generation latency
    g.write_frame(
        json.dumps({"type": "done", "finish_reason": "stop"}).encode()
    )
    g.finish()
    print("[server] Response streamed")

    await asyncio.sleep(0.2)
    del session


async def client_side(port: str) -> str:
    """Connect to the server, send a request, read streaming tokens."""
    await asyncio.sleep(0.05)  # let server start accepting

    # Client origins
    req_origin = moq_py.MoqOrigin.produce()
    resp_origin = moq_py.MoqOrigin.produce()

    config = moq_py.MoqClientConfig(tls_disable_verify=True)
    client = moq_py.MoqClient.create(config)
    client.with_publish(req_origin.consume())
    client.with_consume(resp_origin)

    session = await client.connect(f"https://localhost:{port}")
    print("[client] Connected")

    # Discover server's broadcast
    resp_consumer = resp_origin.consume()
    announce = await resp_consumer.announced()
    _path, server_bc = announce

    # Subscribe to server's data track for responses
    resp_track = server_bc.subscribe_track("data")

    # Send request (one broadcast per request)
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
    # Start server on random port
    config = moq_py.MoqServerConfig(bind_addr="[::]:0", tls_generate=["localhost"])
    server = await moq_py.MoqServer.create(config)
    addr = server.local_addr()
    port = addr.split(":")[-1]
    print(f"MoQ server on :{port}\n")

    server_task = asyncio.create_task(server_side(server))
    text = await asyncio.wait_for(client_side(port), timeout=10.0)
    await server_task

    expected = "The meaning of life is MoQ transport. "
    assert text == expected, f"Expected {expected!r}, got {text!r}"
    print("\nEnd-to-end MoQ transport test passed!")


if __name__ == "__main__":
    asyncio.run(main())
