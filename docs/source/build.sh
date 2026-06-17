#!/usr/bin/env bash
# Build the MQC-mini docs with notebook execution.
#
# The MyST .md sources store no outputs, so the build must execute them. We do
# NOT rely on mystmd's self-spawned Jupyter server: in some environments it
# spawns a server but cannot connect to it ("🪐 Jupyter server did not respond")
# and leaks the process. Instead we start our own Jupyter server and point
# mystmd at it via JUPYTER_BASE_URL / JUPYTER_TOKEN.
#
# Usage:
#   ./build.sh                       # static HTML build into _build/html, then exit
#   ./build.sh build --html --execute --ci --check-links --strict   # link-checked
#   ./build.sh start --execute       # live preview with executed outputs
# With no arguments the default action is used; otherwise all arguments are
# passed through to `jupyter-book`.  (`--ci` makes the build exit instead of
# serving a preview afterwards.)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${JB_JUPYTER_PORT:-8899}"
TOKEN="${JB_JUPYTER_TOKEN:-mqc3doctoken}"

[ "$#" -eq 0 ] && set -- build --html --execute --ci

jupyter server --no-browser --port="$PORT" \
  --IdentityProvider.token="$TOKEN" \
  --ServerApp.disable_check_xsrf=True >/tmp/mqc3_doc_jserver.log 2>&1 &
trap 'kill $! 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  curl -s "http://localhost:${PORT}/api/status?token=${TOKEN}" >/dev/null 2>&1 && break
  sleep 1
done

export JUPYTER_BASE_URL="http://localhost:${PORT}" JUPYTER_TOKEN="${TOKEN}"
jupyter-book "$@"
