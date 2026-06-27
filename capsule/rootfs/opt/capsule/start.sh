#!/usr/bin/env bash
# Starts RESTful Doom in gRPC agent mode.
set -u

LOG() { echo "[agent-capsule] $*" >&2; }

READY_FLAG=/run/capsule.ready
rm -f "$READY_FLAG" 2>/dev/null || true

cat > /opt/hook.py <<'PY'
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FLAG = "/run/capsule.ready"

class H(BaseHTTPRequestHandler):
    def _resp(self):
        ready = os.path.exists(FLAG)
        body = b'{"status":"ok"}' if ready else b'{"status":"starting"}'
        self.send_response(200 if ready else 503)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._resp()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n:
            self.rfile.read(n)
        self._resp()

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("0.0.0.0", 9000), H).serve_forever()
PY
python3 /opt/hook.py &
LOG "ready hook listening on 0.0.0.0:9000"

APPDIR=/home/app/app
WAD="$(find "$APPDIR" -maxdepth 1 -type f \( -name '*.wad' -o -name '*.WAD' \) | sort | head -1)"
[ -n "$WAD" ] || { LOG "FATAL: no IWAD found in $APPDIR"; exit 1; }

export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
AGENT_PORT="${DOOM_AGENT_PORT:-50051}"
API_PORT="${DOOM_API_PORT:-6666}"
VIDEO_MODE="${DOOM_VIDEO_MODE:-headless}"
VIDEO_PORT="${DOOM_VIDEO_PORT:-6080}"
VIDEO_GEOMETRY="${DOOM_VIDEO_GEOMETRY:-640x480}"
XVFB_DISPLAY="${DOOM_XVFB_DISPLAY:-:99}"
DOOM_VIDEO_ARGS=()

if [ "$VIDEO_MODE" = "mjpeg" ]; then
    export DISPLAY="$XVFB_DISPLAY"
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
    Xvfb "$XVFB_DISPLAY" -screen 0 "${VIDEO_GEOMETRY}x24" -nolisten tcp >/var/log/xvfb.log 2>&1 &
    XVFB_PID=$!
    LOG "Xvfb started pid=$XVFB_PID display=$XVFB_DISPLAY geometry=$VIDEO_GEOMETRY"

    python3 /opt/capsule/video_stream.py \
        --host 0.0.0.0 \
        --port "$VIDEO_PORT" \
        --display "$XVFB_DISPLAY" \
        --geometry "$VIDEO_GEOMETRY" \
        >/var/log/doom-video-stream.log 2>&1 &
    STREAM_PID=$!
    LOG "video stream started pid=$STREAM_PID port=$VIDEO_PORT"

    DOOM_VIDEO_ARGS=(-window -geometry "$VIDEO_GEOMETRY" -nograbmouse -nomouse)
else
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
fi

runuser -u app -- \
    restful-doom -iwad "$WAD" -warp 1 1 -skill 3 -nosound -nomusic \
    "${DOOM_VIDEO_ARGS[@]}" \
    -agentport "$AGENT_PORT" -apiport "$API_PORT" > /var/log/restful-doom.log 2>&1 &
DOOM_PID=$!
LOG "restful-doom started pid=$DOOM_PID agent_port=$AGENT_PORT api_port=$API_PORT video_mode=$VIDEO_MODE"

port_up() {
    python3 - "$1" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
}

for _ in $(seq 1 120); do
    if ! kill -0 "$DOOM_PID" 2>/dev/null; then
        LOG "FATAL: restful-doom exited before ready"
        tail -200 /var/log/restful-doom.log >&2 || true
        exit 1
    fi
    if port_up "$AGENT_PORT"; then
        touch "$READY_FLAG"
        LOG "READY: gRPC port $AGENT_PORT is accepting connections"
        break
    fi
    sleep 1
done

if [ ! -f "$READY_FLAG" ]; then
    LOG "FATAL: gRPC port $AGENT_PORT did not open"
    tail -200 /var/log/restful-doom.log >&2 || true
    exit 1
fi

wait "$DOOM_PID"
