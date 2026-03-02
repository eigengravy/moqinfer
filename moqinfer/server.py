"""MoQ inference server — serves vLLM inference via MoQ relay."""

import asyncio
import json
import uuid

import moq_py

from moqinfer.inference import create_engine, run_inference

RELAY_URL = "https://localhost:4443/"
BROADCAST_NAME = "inference"


async def main():
    # 1. Initialize vLLM engine (shared inference core)
    engine, tokenizer = create_engine()

    # 2. Connect to MoQ relay
    publish_origin = moq_py.MoqOrigin.produce()
    consume_origin = moq_py.MoqOrigin.produce()

    config = moq_py.MoqClientConfig(tls_disable_verify=True)
    client = moq_py.MoqClient.create(config)
    client.with_publish(publish_origin.consume())
    client.with_consume(consume_origin)
    session = await client.connect(RELAY_URL)
    print(f"Connected to relay at {RELAY_URL}")

    # 3. Publish inference broadcast with a single "data" track for all responses
    inference_bc = publish_origin.create_broadcast(BROADCAST_NAME)
    inference_bc.dynamic()
    resp_track = inference_bc.create_track("data")

    # 4. Watch for request broadcasts (one broadcast per request)
    consumer = consume_origin.consume()
    while True:
        announce = await consumer.announced()
        if announce is None:
            break
        path, req_bc = announce
        if req_bc is None:
            continue
        # Skip our own broadcast echo from the relay
        if path == BROADCAST_NAME:
            continue
        asyncio.create_task(
            handle_request_broadcast(engine, tokenizer, resp_track, req_bc)
        )


async def handle_request_broadcast(engine, tokenizer, resp_track, req_bc):
    """Handle a single request broadcast (one broadcast per request)."""
    try:
        data_track = req_bc.subscribe_track("data")
        group = await data_track.next_group()
        if group is None:
            return
        frame_data = await group.read_frame()
        if frame_data is None:
            return
        req = json.loads(frame_data)
        request_id = req.get("request_id", str(uuid.uuid4()))
        print(f"Request {request_id[:8]}...")
        await handle_inference(engine, tokenizer, resp_track, req)
    except Exception as e:
        print(f"Request handler error: {e}")


async def handle_inference(engine, tokenizer, resp_track, req: dict):
    """Run vLLM inference for a single request, streaming tokens over MoQ.

    Uses the shared inference core (run_inference) and writes each frame
    as a MoQ group frame on the shared response track.
    """
    request_id = req.get("request_id", str(uuid.uuid4()))
    try:
        group = resp_track.append_group()
        async for frame in run_inference(engine, tokenizer, req):
            group.write_frame(json.dumps(frame).encode())
        group.finish()
    except Exception as e:
        print(f"Inference error for {request_id}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
