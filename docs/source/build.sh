#!/usr/bin/env bash
# Build the MQC-mini Jupyter Book 2 docs with notebook execution.
#
# The MyST .md sources do not store outputs, so the build must execute them.
# mystmd's self-spawned Jupyter server does not start reliably with the
# ipykernel version in this environment, so we run our own Jupyter server and
# point mystmd at it via JUPYTER_BASE_URL / JUPYTER_TOKEN.
#
# Usage:
#   ./build.sh                 # build static HTML into _build/html (with --execute)
#   ./build.sh --html          # same (explicit)
#   ./build.sh serve [port]    # serve the built _build/html on one port (default 8080)
#   ./build.sh start --execute # live preview with hot reload + executed outputs
# Any extra args are passed through to `jupyter-book`.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${JB_JUPYTER_PORT:-8899}"
TOKEN="${JB_JUPYTER_TOKEN:-mqc3doctoken}"

# `serve` mode: just serve the already-built static site on a single port.
if [ "${1:-}" = "serve" ]; then
  SERVE_PORT="${2:-8080}"
  if [ ! -d _build/html ]; then
    echo "❌ _build/html not found — run ./build.sh first." >&2
    exit 1
  fi
  echo "🌐 Serving _build/html at http://localhost:${SERVE_PORT}  (Ctrl-C to stop)"
  exec python -m http.server "${SERVE_PORT}" --directory _build/html
fi

# Default action: static HTML build with execution.
if [ "$#" -eq 0 ]; then
  set -- build --html --execute
fi

cleanup() { [ -n "${JPID:-}" ] && kill "$JPID" 2>/dev/null || true; }
trap cleanup EXIT

echo "🚀 Starting Jupyter server on port ${PORT} for notebook execution..."
jupyter server --no-browser --port="${PORT}" \
  --IdentityProvider.token="${TOKEN}" \
  --ServerApp.disable_check_xsrf=True >/tmp/mqc3_doc_jserver.log 2>&1 &
JPID=$!

for _ in $(seq 1 30); do
  if curl -s "http://localhost:${PORT}/api/status?token=${TOKEN}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

export JUPYTER_BASE_URL="http://localhost:${PORT}"
export JUPYTER_TOKEN="${TOKEN}"

echo "📚 Running: jupyter-book $*"
jupyter-book "$@"
