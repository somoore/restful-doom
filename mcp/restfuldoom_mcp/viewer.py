"""Live browser spectator for RESTful Doom protobuf state."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


FIXED_SCALE = 65536.0


def _repo_root() -> Path:
    override = os.environ.get("RESTFUL_DOOM_REPO")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
AGENT_ROOT = ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


class SharedState:
    """Thread-safe holder for the latest observed Doom state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "connected": False,
            "tick": None,
            "updated_at": None,
            "error": None,
            "player": None,
            "enemies": [],
            "objects": [],
            "path": [],
        }

    def update(self, state: dict[str, Any]) -> None:
        with self._lock:
            prior_path = list(self._state.get("path") or [])
            player = state.get("player") or {}
            if "x" in player and "y" in player:
                prior_path.append({"x": player["x"], "y": player["y"]})
                prior_path = prior_path[-180:]
            state["path"] = prior_path
            self._state = state

    def mark_error(self, error: str) -> None:
        with self._lock:
            current = dict(self._state)
            current["connected"] = False
            current["error"] = error
            current["updated_at"] = time.time()
            self._state = current

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))


async def observe_loop(endpoint: str, shared: SharedState) -> None:
    """Continuously observes Doom state and updates the viewer snapshot."""
    from restfuldoom_agent.client import BackoffConfig, DoomAgentClient

    while True:
        client = DoomAgentClient(endpoint)
        try:
            async for state in client.observe_reconnecting(
                backoff=BackoffConfig(initial_seconds=0.25, max_seconds=2.0, max_attempts=1000)
            ):
                shared.update(_state_to_view(state, endpoint))
        except Exception as error:  # noqa: BLE001 - viewer should keep reconnecting.
            shared.mark_error(str(error))
            await asyncio.sleep(1.0)
        finally:
            await client.close()


def _state_to_view(state: Any, endpoint: str) -> dict[str, Any]:
    player_obj = state.player.object
    player_position = _position(player_obj.position)
    enemies = [
        {
            "id": enemy.object.id,
            "x": enemy.object.position.x_fp / FIXED_SCALE,
            "y": enemy.object.position.y_fp / FIXED_SCALE,
            "health": enemy.object.health,
            "type_id": enemy.object.type_id,
            "line_of_sight": enemy.line_of_sight,
            "distance": enemy.object.distance_fp / FIXED_SCALE,
        }
        for enemy in state.enemies
    ]
    objects = [
        {
            "id": obj.id,
            "x": obj.position.x_fp / FIXED_SCALE,
            "y": obj.position.y_fp / FIXED_SCALE,
            "health": obj.health,
            "type_id": obj.type_id,
            "flags": obj.flags,
        }
        for obj in state.objects[:160]
    ]
    bounds = _bounds([player_position, *enemies, *objects])
    return {
        "connected": True,
        "endpoint": endpoint,
        "tick": state.tick,
        "updated_at": time.time(),
        "error": None,
        "level": {
            "episode": state.level.episode,
            "map": state.level.map,
            "time": state.level.level_time,
            "total_kills": state.level.total_kills,
            "total_items": state.level.total_items,
            "total_secrets": state.level.total_secrets,
        },
        "player": {
            **player_position,
            "angle": player_obj.angle_degrees,
            "health": state.player.health,
            "armor": state.player.armor,
            "kills": state.player.kills,
            "items": state.player.items,
            "secrets": state.player.secrets,
            "weapon": int(state.player.ready_weapon),
        },
        "enemies": enemies,
        "objects": objects,
        "bounds": bounds,
        "has_delta_state": state.has_delta_state,
    }


def _position(position: Any) -> dict[str, float]:
    return {
        "x": position.x_fp / FIXED_SCALE,
        "y": position.y_fp / FIXED_SCALE,
        "z": position.z_fp / FIXED_SCALE,
    }


def _bounds(points: list[dict[str, Any]]) -> dict[str, float]:
    xs = [float(point["x"]) for point in points if "x" in point]
    ys = [float(point["y"]) for point in points if "y" in point]
    if not xs or not ys:
        return {"min_x": -1024, "max_x": 1024, "min_y": -1024, "max_y": 1024}
    padding = 256.0
    return {
        "min_x": min(xs) - padding,
        "max_x": max(xs) + padding,
        "min_y": min(ys) - padding,
        "max_y": max(ys) + padding,
    }


def make_handler(shared: SharedState) -> type[BaseHTTPRequestHandler]:
    """Builds an HTTP handler bound to shared viewer state."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                self._send("text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            if self.path == "/state":
                self._send(
                    "application/json",
                    json.dumps(shared.snapshot(), separators=(",", ":")).encode("utf-8"),
                )
                return
            if self.path == "/health":
                self._send("application/json", b'{"status":"ok"}')
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, content_type: str, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RESTful Doom Agent Viewer</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #10110f;
    --panel: #191b17;
    --line: #30342c;
    --text: #f1f1e8;
    --muted: #a8ad9d;
    --player: #5ee36d;
    --enemy: #ef4f45;
    --enemy-los: #ffb347;
    --object: #54a9ff;
    --path: #d6d971;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  main {
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 10px 14px;
    border-bottom: 1px solid var(--line);
    background: #151713;
  }
  h1 {
    margin: 0;
    font-size: 15px;
    font-weight: 650;
  }
  .stats {
    display: flex;
    align-items: center;
    gap: 16px;
    color: var(--muted);
    white-space: nowrap;
  }
  .stats strong { color: var(--text); font-weight: 650; }
  .wrap {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    min-height: 0;
  }
  canvas {
    display: block;
    width: 100%;
    height: calc(100vh - 46px);
    background: #0d0f0c;
  }
  aside {
    border-left: 1px solid var(--line);
    background: var(--panel);
    padding: 14px;
    overflow: auto;
  }
  .section {
    padding: 0 0 14px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--line);
  }
  .section:last-child { border-bottom: 0; }
  .label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
  }
  .value {
    margin-top: 4px;
    font-size: 18px;
    font-weight: 650;
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 3px 0;
  }
  .ok { color: var(--player); }
  .bad { color: var(--enemy); }
  @media (max-width: 800px) {
    header { align-items: flex-start; flex-direction: column; }
    .stats { flex-wrap: wrap; white-space: normal; }
    .wrap { grid-template-columns: 1fr; }
    canvas { height: 68vh; }
    aside { border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<main>
  <header>
    <h1>RESTful Doom Agent Viewer</h1>
    <div class="stats">
      <span>Status <strong id="status">starting</strong></span>
      <span>Tick <strong id="tick">-</strong></span>
      <span>Health <strong id="health">-</strong></span>
      <span>Enemies <strong id="enemies">-</strong></span>
    </div>
  </header>
  <div class="wrap">
    <canvas id="map"></canvas>
    <aside>
      <div class="section">
        <div class="label">Endpoint</div>
        <div class="value" id="endpoint">-</div>
      </div>
      <div class="section">
        <div class="row"><span>Level</span><strong id="level">-</strong></div>
        <div class="row"><span>Weapon</span><strong id="weapon">-</strong></div>
        <div class="row"><span>Armor</span><strong id="armor">-</strong></div>
      </div>
      <div class="section">
        <div class="row"><span>Kills</span><strong id="kills">-</strong></div>
        <div class="row"><span>Items</span><strong id="items">-</strong></div>
        <div class="row"><span>Secrets</span><strong id="secrets">-</strong></div>
      </div>
      <div class="section">
        <div class="row"><span>Objects</span><strong id="objects">-</strong></div>
        <div class="row"><span>Delta</span><strong id="delta">-</strong></div>
        <div class="row"><span>Last update</span><strong id="updated">-</strong></div>
      </div>
    </aside>
  </div>
</main>
<script>
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
let latest = null;

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function setText(id, value) {
  document.getElementById(id).textContent = value ?? "-";
}

function worldToScreen(point, bounds) {
  const pad = 36;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  const sx = w / Math.max(1, bounds.max_x - bounds.min_x);
  const sy = h / Math.max(1, bounds.max_y - bounds.min_y);
  const s = Math.min(sx, sy);
  const ox = (canvas.width - (bounds.max_x - bounds.min_x) * s) / 2;
  const oy = (canvas.height - (bounds.max_y - bounds.min_y) * s) / 2;
  return {
    x: ox + (point.x - bounds.min_x) * s,
    y: oy + (bounds.max_y - point.y) * s,
    scale: s
  };
}

function drawGrid(bounds) {
  ctx.strokeStyle = "#232820";
  ctx.lineWidth = 1;
  const step = 256;
  const minX = Math.floor(bounds.min_x / step) * step;
  const maxX = Math.ceil(bounds.max_x / step) * step;
  const minY = Math.floor(bounds.min_y / step) * step;
  const maxY = Math.ceil(bounds.max_y / step) * step;
  for (let x = minX; x <= maxX; x += step) {
    const a = worldToScreen({x, y: minY}, bounds);
    const b = worldToScreen({x, y: maxY}, bounds);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  for (let y = minY; y <= maxY; y += step) {
    const a = worldToScreen({x: minX, y}, bounds);
    const b = worldToScreen({x: maxX, y}, bounds);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
}

function drawPoint(point, bounds, color, radius) {
  const p = worldToScreen(point, bounds);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawPlayer(player, bounds) {
  const p = worldToScreen(player, bounds);
  const angle = (player.angle || 0) * Math.PI / 180;
  const r = 12;
  ctx.fillStyle = "#5ee36d";
  ctx.beginPath();
  ctx.moveTo(p.x + Math.cos(angle) * r, p.y - Math.sin(angle) * r);
  ctx.lineTo(p.x + Math.cos(angle + 2.45) * r, p.y - Math.sin(angle + 2.45) * r);
  ctx.lineTo(p.x + Math.cos(angle - 2.45) * r, p.y - Math.sin(angle - 2.45) * r);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "#d8ffd8";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawPath(path, bounds) {
  if (!path || path.length < 2) return;
  ctx.strokeStyle = "#d6d971";
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach((point, index) => {
    const p = worldToScreen(point, bounds);
    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
}

function draw() {
  fitCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0d0f0c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!latest || !latest.player) return;

  const bounds = latest.bounds;
  drawGrid(bounds);
  drawPath(latest.path, bounds);
  latest.objects.forEach(obj => drawPoint(obj, bounds, "#54a9ff", obj.health > 0 ? 4 : 2));
  latest.enemies.forEach(enemy => {
    drawPoint(enemy, bounds, enemy.line_of_sight ? "#ffb347" : "#ef4f45", enemy.health > 0 ? 7 : 4);
  });
  drawPlayer(latest.player, bounds);
}

async function refresh() {
  try {
    const response = await fetch("/state", {cache: "no-store"});
    latest = await response.json();
    const connected = latest.connected;
    setText("status", connected ? "connected" : "waiting");
    document.getElementById("status").className = connected ? "ok" : "bad";
    setText("tick", latest.tick);
    setText("health", latest.player?.health);
    setText("enemies", latest.enemies?.length);
    setText("endpoint", latest.endpoint);
    setText("level", latest.level ? `E${latest.level.episode}M${latest.level.map}` : "-");
    setText("weapon", latest.player?.weapon);
    setText("armor", latest.player?.armor);
    setText("kills", latest.player?.kills);
    setText("items", latest.player?.items);
    setText("secrets", latest.player?.secrets);
    setText("objects", latest.objects?.length);
    setText("delta", latest.has_delta_state ? "yes" : "no");
    setText("updated", latest.updated_at ? new Date(latest.updated_at * 1000).toLocaleTimeString() : "-");
    draw();
  } catch (error) {
    setText("status", "disconnected");
    document.getElementById("status").className = "bad";
  }
}

window.addEventListener("resize", draw);
setInterval(refresh, 120);
refresh();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    shared = SharedState()
    observer = threading.Thread(
        target=lambda: asyncio.run(observe_loop(args.endpoint, shared)),
        daemon=True,
    )
    observer.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    print(f"RESTful Doom viewer listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
