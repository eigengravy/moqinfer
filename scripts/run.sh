#!/bin/bash
# Start the MoQ relay + inference server.
# Usage: ./scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv-vllm-metal/bin/activate

# Build moq-py if needed
if ! python -c "import moq_py" 2>/dev/null; then
    echo "Building moq-py..."
    cd moq-py && maturin develop --release && cd ..
fi

# Build moq-relay if needed
RELAY_BIN="moq/target/release/moq-relay"
if [ ! -f "$RELAY_BIN" ]; then
    echo "Building moq-relay..."
    (cd moq && cargo build --release -p moq-relay)
fi

# Start relay in background
PORT="${MOQ_PORT:-4443}"
echo "Starting moq-relay on port $PORT..."
"$RELAY_BIN" \
    --server-bind "[::]:$PORT" \
    --tls-generate localhost \
    --auth-public "" &
RELAY_PID=$!
sleep 1

# Clean up relay on exit
cleanup() {
    echo "Stopping relay (PID $RELAY_PID)..."
    kill "$RELAY_PID" 2>/dev/null || true
    wait "$RELAY_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting inference server..."
exec python -m moqinfer.server "$@"
