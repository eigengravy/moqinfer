"""REST/SSE inference server — same engine as MoQ server, HTTP transport.

Uses the shared inference core (inference.py) so the ONLY difference
between this and the MoQ server is the transport framing:
  MoQ  → groups/frames over QUIC
  REST → SSE over HTTP

Usage:
    python -m moqinfer.rest_server
"""

import asyncio
import json

from aiohttp import web

from moqinfer.inference import create_engine, run_inference

PORT = 8000


async def handle_inference(request):
    """POST /v1/inference — SSE stream of JSON frames."""
    req = await request.json()
    engine = request.app["engine"]
    tokenizer = request.app["tokenizer"]

    resp = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    await resp.prepare(request)

    async for frame in run_inference(engine, tokenizer, req):
        await resp.write(f"data: {json.dumps(frame)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")

    return resp


async def handle_health(request):
    """GET /health — liveness probe."""
    return web.Response(text="ok")


async def main():
    engine, tokenizer = create_engine()

    app = web.Application()
    app["engine"] = engine
    app["tokenizer"] = tokenizer
    app.router.add_post("/v1/inference", handle_inference)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"REST server ready on port {PORT}", flush=True)

    # Run forever
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
