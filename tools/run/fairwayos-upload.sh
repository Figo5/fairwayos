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
    # An existing pidfile is not proof the service is ours, healthy, or running
    # the code that is checked out now. Verify identity, then readiness, then
    # whether the live server still matches this working tree.
    if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
      RUNPID="$(cat "$PID")"
      if ! ps -p "$RUNPID" -o command= 2>/dev/null | grep -q "ghostcaddie.upload.server"; then
        echo "pid $RUNPID is NOT a fairwayos upload server (pidfile is stale); removing it"
        rm -f "$PID"
      elif ! curl -fsS "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then
        echo "pid $RUNPID is running but /ready does not answer on port $PORT."
        echo "Refusing to report it healthy. Run: $0 stop && $0 start $PORT"
        exit 1
      else
        # compare the LIVE server's runtime map against this working tree's.
        # One python process, no shell quote escaping (which silently broke an
        # earlier version of this check and made every server look stale).
        if ! (cd "$REPO" && "$PY" - "$PORT" <<'PYCHECK'
import json, sys, urllib.request
from ghostcaddie.upload.runtimes import RuntimeRegistry
port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5) as r:
    live = {k: v["interpreter"]
            for k, v in json.load(r)["dependency_readiness"].items()}
want = {k: v.interpreter for k, v in RuntimeRegistry.default().specs.items()}
if live != want:
    print("STALE: the running server does not match this working tree.")
    for k in sorted(set(live) | set(want)):
        if live.get(k) != want.get(k):
            print(f"  {k}: live={live.get(k, 'ABSENT')} worktree={want.get(k, 'ABSENT')}")
    sys.exit(1)
PYCHECK
        ); then
          echo "Restart it:  $0 stop && $0 start $PORT"
          exit 1
        fi
        echo "already running (pid $RUNPID) on port $PORT, /ready healthy, code matches worktree"
        exit 0
      fi
    fi
    cd "$REPO"
    nohup "$PY" -m ghostcaddie.upload.server --root "$ROOT" --port "$PORT" \
      >"$LOG" 2>&1 < /dev/null &
    echo $! > "$PID"
    for i in $(seq 1 40); do
      if curl -fsS "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then
        echo "started pid $(cat "$PID") on http://127.0.0.1:$PORT  root=$ROOT"
        echo "UI:        http://127.0.0.1:$PORT/"
        echo "readiness: curl -s http://127.0.0.1:$PORT/ready"
        exit 0
      fi
      kill -0 "$(cat "$PID")" 2>/dev/null || { echo "FAILED to start:"; tail -20 "$LOG"; exit 1; }
      sleep 0.25
    done
    echo "started but /ready did not answer in 10s; last log lines:"; tail -20 "$LOG"; exit 1
    ;;
  stop)
    if [ -f "$PID" ]; then
      P="$(cat "$PID")"
      kill "$P" 2>/dev/null || true
      for i in $(seq 1 20); do kill -0 "$P" 2>/dev/null || break; sleep 0.25; done
      kill -0 "$P" 2>/dev/null && kill -9 "$P" 2>/dev/null || true
    fi
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
