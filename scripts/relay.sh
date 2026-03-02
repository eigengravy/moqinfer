#!/bin/bash
# Start the MoQ relay server.
# Usage: ./scripts/relay.sh [port]
set -e

cd "$(dirname "$0")/.."

PORT="${1:-4443}"
RELAY_BIN="moq/target/release/moq-relay"

if [ ! -f "$RELAY_BIN" ]; then
    echo "Building moq-relay..."
    (cd moq && cargo build --release -p moq-relay)
fi

echo "Starting moq-relay on port $PORT..."
exec "$RELAY_BIN" \
    --server-bind "[::]:$PORT" \
    --tls-generate localhost \
    --auth-public ""
