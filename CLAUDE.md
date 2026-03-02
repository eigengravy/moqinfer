# CLAUDE.md — moqinfer

## Project Overview

**moqinfer** is a native MoQ (Media over QUIC) endpoint for LLM inference using vLLM. MoQ is the first-class transport layer — no REST proxy.

### Architecture

```
moq-relay (Rust binary)                    ← content-agnostic fan-out relay
    ↑                       ↑
    |                       |
Inference Server          Client(s)
(Python process)          (Python)
├── vLLM AsyncLLM         └── moq_py
└── moq_py (PyO3)
```

The relay is a standalone MoQ relay (`moq/rs/moq-relay/`) that handles routing between publishers and subscribers. Both the inference server and clients connect to the relay as MoQ clients.

Tokio runtime runs in background threads (managed by pyo3-async-runtimes). Python asyncio drives the main loop. Rust async methods become Python awaitables via `future_into_py()`.

## Project Structure

```
moqinfer/
├── moq/                        # git submodule — DO NOT MODIFY
├── vllm-metal/                 # git submodule — DO NOT MODIFY
├── .venv-vllm-metal/           # Python 3.12 venv
├── moq-py/                     # PyO3/maturin Rust crate
│   ├── Cargo.toml              # Rust deps: pyo3, moq-lite, moq-native, tokio
│   ├── pyproject.toml          # maturin build config
│   └── src/
│       ├── lib.rs              # PyO3 module registration
│       ├── server.rs           # MoqServerConfig, MoqServer, MoqRequest, MoqSession
│       ├── origin.rs           # MoqOrigin, MoqOriginProducer, MoqOriginConsumer
│       ├── broadcast.rs        # MoqBroadcastProducer, MoqBroadcastDynamic, MoqBroadcastConsumer
│       ├── track.rs            # MoqTrackProducer, MoqTrackConsumer
│       ├── group.rs            # MoqGroupProducer, MoqGroupConsumer
│       └── error.rs            # moq_lite::Error → PyErr conversion
├── moqinfer/                   # Python package
│   ├── __init__.py
│   ├── server.py               # Inference server (connects to relay)
│   └── client.py               # Inference client (connects to relay)
├── scripts/
│   ├── run.sh                  # Start relay + inference server
│   └── relay.sh                # Start relay standalone
├── test_e2e.py                 # Direct connection test (no relay)
├── test_relay_e2e.py           # Relay e2e test (server ↔ relay ↔ client)
├── pyproject.toml              # Python project config
└── CLAUDE.md                   # This file
```

## Build & Run

```bash
# 1. Activate venv
source .venv-vllm-metal/bin/activate

# 2. Build moq-py (installs moq_py into venv)
cd moq-py && maturin develop --release && cd ..

# 3. Build moq-relay
cd moq && cargo build --release -p moq-relay && cd ..

# 4. Start the relay (terminal 1)
./scripts/relay.sh

# 5. Start the inference server (terminal 2)
python -m moqinfer.server

# 6. Run the client (terminal 3)
python -m moqinfer.client "Your prompt here"

# Or use the convenience script (starts relay + server together):
./scripts/run.sh
```

## Tests

```bash
# Direct connection test (no relay, mock inference)
python test_e2e.py

# Relay e2e test (relay subprocess, mock inference)
python test_relay_e2e.py
```

## MoQ Protocol Design

All communication flows through the relay. Both server and clients connect as MoQ clients.

**Relay setup:**
- `moq-relay --auth-public ""` — fully public, no JWT auth required
- All broadcasts route through relay's `primary` → `combined` origin
- Both sides see each other's broadcasts (and their own echo, which must be filtered)

**Server → Relay:**
- Publishes broadcast named `"inference"` with dynamic track handler
- Uses `TrackDispatcher` to match `response/{request_id}` track subscriptions to request handlers
- Watches for client broadcasts via `consume_origin.announced()`

**Client → Relay:**
- Publishes broadcast named `"client-{uuid}"` containing a `"request"` track
- Discovers server's `"inference"` broadcast via `resp_origin.announced()`
- Subscribes to `"response/{request_id}"` track on the server's broadcast

**Request format** (JSON frame on `"request"` track):
```json
{"request_id": "uuid", "prompt": "...", "sampling_params": {"max_tokens": 512}}
```

**Response format** (JSON frames on `"response/{request_id}"` track):
- Frame 0: `{"type": "start", "request_id": "...", "model": "..."}`
- Frame 1..N: `{"type": "token", "text": "..."}`
- Frame N+1: `{"type": "done", "finish_reason": "stop"}`

## Python API (moq_py module)

### Server
```python
config = moq_py.MoqServerConfig(bind_addr="[::]:4443", tls_generate=["localhost"])
server = await moq_py.MoqServer.create(config)
request = await server.accept()  # Returns MoqRequest or None
```

### Client
```python
config = moq_py.MoqClientConfig(tls_disable_verify=True)
client = moq_py.MoqClient.create(config)
client.with_publish(req_origin.consume())
client.with_consume(resp_origin)
session = await client.connect("https://localhost:4443")
```

### Origins
```python
origin = moq_py.MoqOrigin.produce()       # MoqOriginProducer
consumer = origin.consume()                 # MoqOriginConsumer
announce = await consumer.announced()       # (path, Optional[MoqBroadcastConsumer]) or None
```

### Broadcasts
```python
broadcast = origin.create_broadcast("")     # MoqBroadcastProducer
track = broadcast.create_track("name")      # MoqTrackProducer
dynamic = broadcast.dynamic()               # MoqBroadcastDynamic
requested = await dynamic.requested_track() # Optional[MoqTrackProducer]

bc = broadcast.consume()                    # MoqBroadcastConsumer
tc = bc.subscribe_track("name")             # MoqTrackConsumer
```

### Tracks & Groups
```python
group = track.append_group()                # MoqGroupProducer
group.write_frame(b"data")
group.finish()

group = await track_consumer.next_group()   # Optional[MoqGroupConsumer]
data = await group.read_frame()             # bytes or None
```

## Ownership Patterns (Rust side)

| Python Class | Rust Wrapper | Reason |
|---|---|---|
| MoqServer | `Arc<tokio::sync::Mutex<Server>>` | async accept() takes &mut |
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

## Key Implementation Notes

- **Tokio runtime**: pyo3-async-runtimes spawns a multi-thread tokio runtime on background threads. `future_into_py()` bridges Rust futures to Python awaitables.
- **blocking_lock()**: Used for sync Python methods on tokio::sync::Mutex-wrapped types. Safe because Python's main thread is not inside the tokio runtime.
- **TLS**: `tls_generate=["localhost"]` creates self-signed certs. Clients must disable cert verification or use the fingerprint from `server.tls_info()`.
- **moq submodule**: Pinned via git submodule. moq-py uses path dependencies to `../moq/rs/moq-lite` and `../moq/rs/moq-native`.
