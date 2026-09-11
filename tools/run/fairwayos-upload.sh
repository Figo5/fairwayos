#!/usr/bin/env bash
# Durable launcher for the FairwayOS upload service.
#
#   tools/run/fairwayos-upload.sh start|stop|status|ready [port]
#
# Survives the launching shell (setsid/nohup + pidfile). Binds 127.0.0.1 only.
# No global installs: each target runs in its own repo-local interpreter.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${2:-8765}"
ROOT="${FAIRWAYOS_UPLOAD_ROOT:-$REPO/.uploads}"
PID="$ROOT/service.pid"
LOG="$ROOT/service.log"
PY="$REPO/.venv/bin/python3"

mkdir -p "$ROOT"

case "${1:-status}" in
  start)
    if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
      echo "already running (pid $(cat "$PID")) on port $PORT"; exit 0
    fi
    cd "$REPO"
    nohup "$PY" -m ghostcaddie.upload.server --root "$ROOT" --port "$PORT" \
      >"$LOG" 2>&1 < /dev/null &
    echo $! > "$PID"
    sleep 2
    if kill -0 "$(cat "$PID")" 2>/dev/null; then
      echo "started pid $(cat "$PID") on http://127.0.0.1:$PORT  root=$ROOT"
      echo "readiness: curl -s http://127.0.0.1:$PORT/ready"
    else
      echo "FAILED to start; last log lines:"; tail -20 "$LOG"; exit 1
    fi
    ;;
  stop)
    [ -f "$PID" ] && kill "$(cat "$PID")" 2>/dev/null || true
    rm -f "$PID"; echo "stopped"
    ;;
  status)
    if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
      echo "running pid $(cat "$PID") port $PORT"
    else
      echo "not running"; exit 1
    fi
    ;;
  ready)
    curl -s "http://127.0.0.1:$PORT/ready"
    ;;
  *) echo "usage: $0 start|stop|status|ready [port]"; exit 2;;
esac
