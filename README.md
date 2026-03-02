# moqinfer

Native [MoQ (Media over QUIC)](https://datatracker.ietf.org/group/moq/about/) transport for LLM inference, built on [vLLM](https://github.com/vllm-project/vllm). MoQ is the first-class transport layer — no REST proxy.

## Why MoQ for inference?

HTTP/SSE is request-response: one TCP connection per inference call, new TLS handshake, head-of-line blocking. MoQ runs over QUIC with a persistent connection, multiplexed streams, and native fan-out via a relay. This means:

- **Persistent connection** — single QUIC session handles all requests (no per-request TCP/TLS overhead)
- **Multiplexed streams** — concurrent requests don't block each other
- **Relay fan-out** — one inference server can broadcast to many clients through a content-agnostic relay
- **Lower tool-call RTT** — multi-turn agent loops avoid HTTP round-trip overhead per turn

## Architecture

```
moq-relay (Rust binary)                    <- content-agnostic fan-out relay
    ^                       ^
    |                       |
Inference Server          Client(s)
(Python process)          (Python)
|-- vLLM AsyncLLM         +-- AgentBackend (moq_py)
+-- moq_py (PyO3)

REST/SSE Server                            <- HTTP alternative (same engine)
(Python process)
|-- vLLM AsyncLLM
+-- aiohttp                +-- RestBackend (httpx)
```

The relay is a standalone MoQ relay (`moq/rs/moq-relay/`) that handles routing between publishers and subscribers. Both the inference server and clients connect to the relay as MoQ clients.

Both MoQ and REST servers share the **exact same inference core** (`moqinfer/inference.py`): same `AsyncLLM` engine, same `apply_chat_template()`, same `parse_tool_calls()` regex parser. The only variable is transport framing.

## Project Structure

```
moqinfer/
|-- moq/                           # git submodule (moq-dev/moq) -- DO NOT MODIFY
|-- vllm-metal/                    # git submodule (vLLM for Apple Silicon) -- DO NOT MODIFY
|-- .venv-vllm-metal/              # Python 3.12 venv
|
|-- moq-py/                        # PyO3/maturin Rust crate -> moq_py Python module
|   |-- Cargo.toml                 # Rust deps: pyo3, moq-lite, moq-native, tokio
|   |-- pyproject.toml             # maturin build config
|   +-- src/
|       |-- lib.rs                 # PyO3 module registration (all 11 classes)
|       |-- server.rs              # MoqServerConfig, MoqServer, MoqRequest, MoqSession
|       |-- client.rs              # MoqClientConfig, MoqClient
|       |-- origin.rs              # MoqOrigin, MoqOriginProducer, MoqOriginConsumer
|       |-- broadcast.rs           # MoqBroadcastProducer, MoqBroadcastDynamic, MoqBroadcastConsumer
|       |-- track.rs               # MoqTrackProducer, MoqTrackConsumer
|       |-- group.rs               # MoqGroupProducer, MoqGroupConsumer
|       +-- error.rs               # moq_lite::Error -> PyErr, enter_runtime() helper
|
|-- moqinfer/                      # Python package
|   |-- __init__.py                # Public API: AgentBackend, RestBackend, metrics
|   |-- inference.py               # Shared inference core (engine, tokenizer, parse_tool_calls)
|   |-- server.py                  # MoQ inference server (connects to relay)
|   |-- rest_server.py             # REST/SSE inference server (aiohttp, same engine)
|   |-- client.py                  # CLI client for interactive use
|   |-- backend.py                 # AgentBackend (MoQ transport) + ChatResult/GenerateResult
|   |-- rest_backend.py            # RestBackend (HTTP/SSE transport via httpx)
|   +-- metrics.py                 # BenchmarkResult, RequestMetrics, comparison tables
|
|-- benchmark.py                   # MoQ vs REST benchmark harness
|-- test_e2e.py                    # Direct connection test (no relay, mock inference)
|-- test_relay_e2e.py              # Relay e2e test (server <-> relay <-> client)
|-- scripts/
|   |-- run.sh                     # Start relay + inference server
|   +-- relay.sh                   # Start relay standalone
|-- pyproject.toml                 # Python project config
+-- CLAUDE.md                      # Claude Code instructions
```

## Setup

### Prerequisites

- macOS with Apple Silicon (M1+) for vllm-metal GPU acceleration
- Rust toolchain (for building moq-py and moq-relay)
- Python 3.12+

### Build

```bash
# 1. Activate venv
source .venv-vllm-metal/bin/activate

# 2. Build moq-py (installs moq_py into venv)
cd moq-py && maturin develop --release && cd ..

# 3. Build moq-relay
cd moq && cargo build --release -p moq-relay && cd ..
```

### Run

```bash
# Terminal 1: Start the relay
./scripts/relay.sh

# Terminal 2: Start the MoQ inference server
python -m moqinfer.server

# Terminal 3: Run the client
python -m moqinfer.client "What is the meaning of life?"

# Or use the convenience script (relay + server together):
./scripts/run.sh
```

### REST/SSE server (alternative transport)

```bash
# Start the REST server (same engine, HTTP transport)
python -m moqinfer.rest_server
```

## MoQ Protocol Design

All communication flows through the relay. Both server and clients connect as MoQ clients.

### Connection topology

```
Client                          Relay                         Server
  |                               |                              |
  |-- connect (QUIC) ----------->|<---------- connect (QUIC) ---|
  |                               |                              |
  |   publish: "client-{uuid}"   |   publish: "inference"       |
  |   consume: origin            |   consume: origin             |
  |                               |                              |
  |   subscribe: inference/data  --->  forward to server         |
  |                               |                              |
  |   broadcast: {request_id}    --->  announce to server        |
  |     track: data              |     server subscribes         |
  |     group: [request JSON]    --->  server reads request      |
  |                               |                              |
  |                               |  server writes to            |
  |   read: inference/data       <---  inference/data track      |
  |     group: [start, tokens.., done]                           |
```

### Request format (JSON frame on request broadcast's "data" track)

```json
{
  "request_id": "uuid",
  "messages": [{"role": "user", "content": "..."}],
  "tools": [{"type": "function", "function": {...}}],
  "sampling_params": {"max_tokens": 512, "temperature": 0, "seed": 42}
}
```

Simple prompt mode is also supported:
```json
{"request_id": "uuid", "prompt": "...", "sampling_params": {"max_tokens": 512}}
```

### Response format (JSON frames in a group on "inference" broadcast's "data" track)

```
Frame 0:   {"type": "start", "request_id": "...", "model": "..."}
Frame 1-N: {"type": "token", "text": "delta"}
Frame N+1: {"type": "tool_calls", "tool_calls": [...]}   (optional, if tools detected)
Frame N+2: {"type": "done", "finish_reason": "stop"|"tool_calls"}
```

The REST/SSE server uses the **exact same frame protocol** over SSE (`data: {json}\n\n` lines), so both backends (`AgentBackend` and `RestBackend`) speak the same language.

### Broadcast-per-request pattern

Each client request creates a new broadcast named by its `request_id`. All responses flow back on a shared "data" track on the server's "inference" broadcast, with groups demuxed by `request_id` in the start frame. This allows a single MoQ connection to multiplex many concurrent requests.

## Python API

### AgentBackend (MoQ transport)

```python
from moqinfer.backend import AgentBackend

# Connect to relay (single persistent QUIC connection)
backend = await AgentBackend.connect("https://localhost:4443/")

# Simple text generation
result = await backend.generate("What is life?")
print(result.text)

# Streaming
async for frame in backend.generate_stream("What is life?"):
    if frame["type"] == "token":
        print(frame["text"], end="")

# Chat with tool calling (automatic multi-turn loop)
result = await backend.chat(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
    tool_executor=my_tool_function,  # called when model invokes tools
    max_rounds=10,
)
print(result.text)
print(result.tool_calls)    # list of tool calls from final round
print(result.messages)      # full conversation history

await backend.close()
```

### RestBackend (HTTP/SSE transport)

Drop-in replacement — identical API, but each request is an HTTP/SSE call:

```python
from moqinfer.rest_backend import RestBackend

backend = await RestBackend.connect("http://localhost:8000")
result = await backend.chat(messages=[...], tools=[...], tool_executor=fn)
await backend.close()
```

### Low-level moq_py bindings

```python
import moq_py

# Origins (pub/sub routing)
origin = moq_py.MoqOrigin.produce()       # MoqOriginProducer
consumer = origin.consume()                 # MoqOriginConsumer
announce = await consumer.announced()       # (path, Optional[MoqBroadcastConsumer])

# Broadcasts (named content streams)
broadcast = origin.create_broadcast("name") # MoqBroadcastProducer
dynamic = broadcast.dynamic()               # MoqBroadcastDynamic (track request handler)
bc_consumer = broadcast.consume()           # MoqBroadcastConsumer

# Tracks (ordered sequence of groups within a broadcast)
track = broadcast.create_track("data")      # MoqTrackProducer
tc = bc_consumer.subscribe_track("data")    # MoqTrackConsumer

# Groups (ordered sequence of frames within a track)
group = track.append_group()                # MoqGroupProducer
group.write_frame(b"data")
group.finish()

group = await tc.next_group()               # MoqGroupConsumer
data = await group.read_frame()             # bytes or None

# Client connection
config = moq_py.MoqClientConfig(tls_disable_verify=True)
client = moq_py.MoqClient.create(config)
client.with_publish(origin.consume())       # what we publish
client.with_consume(another_origin)         # where we receive
session = await client.connect("https://localhost:4443")

# Server (direct connections, no relay)
config = moq_py.MoqServerConfig(bind_addr="[::]:4443", tls_generate=["localhost"])
server = await moq_py.MoqServer.create(config)
request = await server.accept()             # MoqRequest
request.with_publish(origin.consume())
request.with_consume(another_origin)
session = await request.ok()                # MoqSession
```

## moq-py Rust Internals

The `moq-py` crate (671 lines of Rust) bridges moq-lite/moq-native to Python via PyO3.

### Runtime model

- **Tokio runtime** runs on background threads, managed by `pyo3-async-runtimes`
- **`future_into_py()`** bridges Rust `async fn` to Python awaitables
- **`enter_runtime()`** provides tokio context for sync methods (needed because moq-lite internally spawns tokio tasks)
- **`blocking_lock()`** is safe on Python's main thread since it's not inside the tokio runtime

### Ownership patterns

| Python Class | Rust Wrapper | Why |
|---|---|---|
| MoqServer | `Arc<tokio::sync::Mutex>` | async accept() takes &mut |
| MoqRequest | `Option<Request>` | consumed by ok() |
| MoqSession | `Option<Session>` | consumed by close() |
| MoqOriginProducer | direct field (Clone) | all methods take &self |
| MoqOriginConsumer | `Arc<tokio::sync::Mutex>` | async announced() takes &mut |
| MoqBroadcastProducer | `Arc<std::sync::Mutex>` | sync &mut self methods |
| MoqBroadcastDynamic | `Arc<tokio::sync::Mutex>` | async requested_track() takes &mut |
| MoqBroadcastConsumer | direct field (Clone) | all methods take &self |
| MoqTrackProducer | `Arc<std::sync::Mutex>` | sync &mut self methods |
| MoqTrackConsumer | `Arc<tokio::sync::Mutex>` | async next_group() takes &mut |
| MoqGroupProducer | `Arc<std::sync::Mutex>` | sync &mut self methods |
| MoqGroupConsumer | `Arc<tokio::sync::Mutex>` | async read_frame() takes &mut |

The choice between `std::sync::Mutex` and `tokio::sync::Mutex` depends on whether the methods are sync (called from Python thread) or async (called inside tokio futures).

### Error handling

`moq-py/src/error.rs` maps moq-lite errors to Python exceptions:
- Transport/Closed/Cancel/Dropped -> `ConnectionError`
- NotFound/Version/InvalidRole/Duplicate -> `ValueError`
- Everything else -> `RuntimeError`

## Shared Inference Core

`moqinfer/inference.py` contains all inference logic shared between MoQ and REST servers:

- **`MODEL`** — model identifier (`Qwen/Qwen3-4B-Instruct-2507`)
- **`parse_tool_calls(text)`** — extracts `<tool_call>{JSON}</tool_call>` blocks (Qwen3 format), strips `<think>` reasoning blocks
- **`create_engine(seed=42)`** — creates vLLM `AsyncLLM` engine + tokenizer
- **`run_inference(engine, tokenizer, req)`** — async generator yielding the standard frame protocol (start/token/tool_calls/done)

Both servers are thin transport wrappers around `run_inference()`:
- `server.py`: writes each frame as a MoQ group frame
- `rest_server.py`: writes each frame as an SSE `data:` line

## Tool Calling

The model (Qwen3) emits tool calls in XML format:
```
<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>
```

`parse_tool_calls()` extracts these via regex and returns structured tool call objects. Both `AgentBackend.chat()` and `RestBackend.chat()` support automatic multi-turn tool loops via `tool_executor` callback — they send tool results back to the model and loop until the model stops calling tools or `max_rounds` is reached.

## Benchmark

The benchmark compares MoQ vs REST transport using identical inference (same engine, same seed, same parser):

```bash
python benchmark.py                    # Both transports
python benchmark.py --moq-only         # MoQ only
python benchmark.py --rest-only        # REST only
python benchmark.py --backends 1 --users 1   # No contention (isolate transport overhead)
```

### What it measures

- **TTFT** — time to first token
- **Completion time** — total request time including all tool rounds
- **Tool RTT** — round-trip time per tool-calling turn
- **Throughput** — tokens/second across all concurrent requests
- **Total tokens** — should be identical between transports (same engine + seed)

### Workload

10 prompts exercising tool calling (weather lookup + temperature conversion), cycled across `backends * users` concurrent requests. Deterministic sampling: `temperature=0, seed=42`.

## Tests

```bash
# Direct MoQ connection test (no relay, mock inference)
python test_e2e.py

# Relay e2e test (relay subprocess, mock inference, full fan-out path)
python test_relay_e2e.py
```

## Platform Notes

- **vllm-metal** reports `device_config=cpu` but uses MLX + Metal GPU underneath on Apple Silicon
- **TLS**: `tls_generate=["localhost"]` creates self-signed certs; clients must use `tls_disable_verify=True`
- **PYTHONUNBUFFERED=1**: always use when starting servers (stdout buffering hides readiness messages)
- **moq submodule**: pinned via git submodule at `moq/` — uses path dependencies to `moq/rs/moq-lite` and `moq/rs/moq-native`
