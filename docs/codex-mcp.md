# Codex MCP

This repo includes a local MCP server that can build and run the protobuf +
gRPC Doom environment in Docker, expose real in-game footage in a browser, and
drive it with the Python agent.

Install it into the current Codex profile:

```bash
scripts/install-codex-mcp.sh
```

The installer creates `.venv-mcp`, installs the Python agent and MCP
dependencies, generates protobuf stubs, and writes this server block to
`~/.codex/config.toml`:

```toml
[mcp_servers.restful_doom]
command = "/path/to/restful-doom/.venv-mcp/bin/python"
args = ["-m", "restfuldoom_mcp.server"]
startup_timeout_sec = 60

[mcp_servers.restful_doom.env]
RESTFUL_DOOM_REPO = "/path/to/restful-doom"
PYTHONPATH = "/path/to/restful-doom/mcp:/path/to/restful-doom/agent"
```

Restart Codex or open a new thread after installation so the MCP tools are
discovered.

Available tools:

- `docker_build`: build `restful-doom-agent:local`
- `docker_start`: start the local Dockerized Doom gRPC server
- `docker_stop`: stop and remove the local container
- `docker_logs`: inspect recent container logs
- `status`: check Docker state, local gRPC reachability, and video reachability
- `video_status`: check whether the actual rendered video stream is reachable
- `viewer_start`: start a browser spectator at `http://127.0.0.1:8765`
- `viewer_stop`: stop the browser spectator
- `viewer_status`: check whether the browser spectator is running
- `observe_once`: read one protobuf `GameState`
- `run_rollout`: drive the game with the deterministic smoke policy

`docker_start` enables video by default. The Doom process runs with SDL's X11
backend inside `Xvfb`, and the container exposes the rendered framebuffer as an
MJPEG stream:

```text
http://127.0.0.1:6080
```

The `viewer_start` tool is separate. It observes the same protobuf `GameState`
feed used by agents and renders a top-down debugging map at
`http://127.0.0.1:8765`; it is not actual in-game footage.

Manual equivalent:

```bash
docker build -f capsule/Dockerfile -t restful-doom-agent:local .
docker run -d --name restful-doom-agent \
  -p 50051:50051 \
  -p 6666:6666 \
  -p 9000:9000 \
  -p 6080:6080 \
  -e DOOM_VIDEO_MODE=mjpeg \
  restful-doom-agent:local
open http://127.0.0.1:6080
PYTHONPATH=mcp:agent .venv-mcp/bin/python -m restfuldoom_mcp.viewer
PYTHONPATH=agent .venv-mcp/bin/python -m restfuldoom_agent.smoke_agent \
  --endpoint 127.0.0.1:50051 \
  --goal-preset navigation \
  --max-states 35 \
  --trajectory-jsonl trajectories/docker-mcp-smoke.jsonl
```
