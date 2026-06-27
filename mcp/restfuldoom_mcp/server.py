"""Codex MCP server for local RESTful Doom agent experiments."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# The MCP process may call Docker subprocesses after importing grpc. Disabling
# grpc fork handlers prevents noisy warnings around subprocess creation.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

from mcp.server.fastmcp import FastMCP


DEFAULT_IMAGE = "restful-doom-agent:local"
DEFAULT_CONTAINER = "restful-doom-agent"
DEFAULT_AGENT_PORT = 50051
DEFAULT_API_PORT = 6666
DEFAULT_READY_PORT = 9000
DEFAULT_VIDEO_PORT = 6080
DEFAULT_VIEWER_PORT = 8765


def _repo_root() -> Path:
    override = os.environ.get("RESTFUL_DOOM_REPO")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
AGENT_ROOT = ROOT / "agent"
MCP = FastMCP("restful-doom")
VIEWER_PID_FILE = Path("/tmp/restfuldoom-mcp-viewer.pid")
VIEWER_LOG_FILE = Path("/tmp/restfuldoom-mcp-viewer.log")


@MCP.tool()
def docker_build(image: str = DEFAULT_IMAGE) -> dict[str, Any]:
    """Build the local Docker image for gRPC RESTful Doom."""
    result = _run(
        ["docker", "build", "-f", "capsule/Dockerfile", "-t", image, "."],
        timeout_seconds=1800,
    )
    return {"image": image, **result}


@MCP.tool()
def docker_start(
    image: str = DEFAULT_IMAGE,
    container: str = DEFAULT_CONTAINER,
    agent_port: int = DEFAULT_AGENT_PORT,
    api_port: int = DEFAULT_API_PORT,
    ready_port: int = DEFAULT_READY_PORT,
    video: bool = True,
    video_port: int = DEFAULT_VIDEO_PORT,
    replace: bool = True,
) -> dict[str, Any]:
    """Start the local Dockerized Doom agent server and wait for gRPC readiness."""
    if replace:
        _run(["docker", "rm", "-f", container], check=False, timeout_seconds=30)

    docker_args = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "-p",
        f"{agent_port}:50051",
        "-p",
        f"{api_port}:6666",
        "-p",
        f"{ready_port}:9000",
        "-e",
        "DOOM_AGENT_PORT=50051",
        "-e",
        "DOOM_API_PORT=6666",
        "-e",
        f"DOOM_VIDEO_MODE={'mjpeg' if video else 'headless'}",
    ]
    if video:
        docker_args.extend(
            [
                "-p",
                f"{video_port}:6080",
                "-e",
                "DOOM_VIDEO_PORT=6080",
                "-e",
                "DOOM_VIDEO_GEOMETRY=640x480",
            ]
        )
    docker_args.append(image)

    result = _run(
        docker_args,
        timeout_seconds=60,
    )
    ready = _wait_for_tcp("127.0.0.1", agent_port, timeout_seconds=120)
    video_ready = _wait_for_tcp("127.0.0.1", video_port, timeout_seconds=15) if video else False
    logs = docker_logs(container=container, lines=80)
    return {
        "container": container,
        "image": image,
        "agent_endpoint": f"127.0.0.1:{agent_port}",
        "api_endpoint": f"127.0.0.1:{api_port}",
        "video_url": f"http://127.0.0.1:{video_port}" if video else None,
        "ready": ready,
        "video_ready": video_ready,
        "docker": result,
        "logs": logs,
    }


@MCP.tool()
def docker_stop(container: str = DEFAULT_CONTAINER) -> dict[str, Any]:
    """Stop and remove the local Dockerized Doom agent server."""
    return {"container": container, **_run(["docker", "rm", "-f", container], check=False)}


@MCP.tool()
def docker_logs(container: str = DEFAULT_CONTAINER, lines: int = 120) -> dict[str, Any]:
    """Return recent logs for the local Doom container."""
    return _run(["docker", "logs", "--tail", str(lines), container], check=False)


@MCP.tool()
def status(
    container: str = DEFAULT_CONTAINER,
    agent_port: int = DEFAULT_AGENT_PORT,
    video_port: int = DEFAULT_VIDEO_PORT,
) -> dict[str, Any]:
    """Report Docker container state and local gRPC port reachability."""
    ps = _run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^{container}$",
            "--format",
            "{{.Names}}\t{{.Status}}\t{{.Ports}}",
        ],
        check=False,
    )
    return {
        "container": container,
        "agent_endpoint": f"127.0.0.1:{agent_port}",
        "video_url": f"http://127.0.0.1:{video_port}",
        "tcp_ready": _tcp_connect("127.0.0.1", agent_port, timeout_seconds=1.0),
        "video_tcp_ready": _tcp_connect("127.0.0.1", video_port, timeout_seconds=1.0),
        "docker_ps": ps,
    }


@MCP.tool()
def video_status(
    host: str = "127.0.0.1",
    port: int = DEFAULT_VIDEO_PORT,
) -> dict[str, Any]:
    """Report whether the actual rendered Doom video stream is reachable."""
    return {
        "url": f"http://{host}:{port}",
        "frame_url": f"http://{host}:{port}/frame",
        "stream_url": f"http://{host}:{port}/stream",
        "http_ready": _tcp_connect(host, port, timeout_seconds=1.0),
    }


@MCP.tool()
def viewer_start(
    endpoint: str = f"127.0.0.1:{DEFAULT_AGENT_PORT}",
    host: str = "127.0.0.1",
    port: int = DEFAULT_VIEWER_PORT,
) -> dict[str, Any]:
    """Start a browser spectator for the live protobuf GameState stream."""
    running = _viewer_process()
    if running is not None:
        return {
            "started": False,
            "pid": running,
            "url": f"http://{host}:{port}",
            "log": str(VIEWER_LOG_FILE),
            "message": "viewer is already running",
        }

    env = dict(os.environ)
    env["RESTFUL_DOOM_REPO"] = str(ROOT)
    env["PYTHONPATH"] = f"{ROOT / 'mcp'}:{ROOT / 'agent'}"
    with VIEWER_LOG_FILE.open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "restfuldoom_mcp.viewer",
                "--endpoint",
                endpoint,
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )

    VIEWER_PID_FILE.write_text(str(process.pid), encoding="utf-8")
    ready = _wait_for_tcp(host, port, timeout_seconds=10)
    return {
        "started": True,
        "pid": process.pid,
        "ready": ready,
        "url": f"http://{host}:{port}",
        "endpoint": endpoint,
        "log": str(VIEWER_LOG_FILE),
    }


@MCP.tool()
def viewer_stop() -> dict[str, Any]:
    """Stop the browser spectator process."""
    pid = _viewer_process()
    if pid is None:
        VIEWER_PID_FILE.unlink(missing_ok=True)
        return {"stopped": False, "message": "viewer is not running"}

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            VIEWER_PID_FILE.unlink(missing_ok=True)
            return {"stopped": True, "pid": pid}
        time.sleep(0.1)

    os.kill(pid, signal.SIGKILL)
    VIEWER_PID_FILE.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid, "forced": True}


@MCP.tool()
def viewer_status(
    host: str = "127.0.0.1",
    port: int = DEFAULT_VIEWER_PORT,
) -> dict[str, Any]:
    """Report whether the browser spectator is running."""
    pid = _viewer_process()
    return {
        "running": pid is not None,
        "pid": pid,
        "url": f"http://{host}:{port}",
        "http_ready": _tcp_connect(host, port, timeout_seconds=1.0),
        "log": str(VIEWER_LOG_FILE),
    }


@MCP.tool()
async def observe_once(
    endpoint: str = f"127.0.0.1:{DEFAULT_AGENT_PORT}",
    include_delta_state: bool = True,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Read one protobuf GameState from the Doom gRPC observe stream."""
    _ensure_agent_path()
    from restfuldoom_agent.client import DoomAgentClient, summarize_state

    client = DoomAgentClient(endpoint)
    try:
        async with asyncio.timeout(timeout_seconds):
            async for state in client.observe(include_delta_state=include_delta_state):
                return summarize_state(state)
    finally:
        await client.close()
    raise RuntimeError("observe stream ended before a state was received")


@MCP.tool()
async def run_rollout(
    endpoint: str = f"127.0.0.1:{DEFAULT_AGENT_PORT}",
    max_states: int = 35,
    goal_preset: str = "navigation",
    trajectory_jsonl: str | None = None,
) -> dict[str, Any]:
    """Drive Doom with the deterministic agent and return rollout telemetry."""
    _ensure_agent_path()
    from restfuldoom_agent.client import DoomAgentClient, action_cycle
    from restfuldoom_agent.reward import RewardEngine, goal_preset as build_goal

    class CyclePolicy:
        def __init__(self) -> None:
            self._actions = iter(action_cycle())

        async def next_action(self, _state: Any) -> Any:
            try:
                return next(self._actions)
            except StopIteration:
                self._actions = iter(action_cycle())
                return next(self._actions)

    client = DoomAgentClient(endpoint)
    total_reward = 0.0
    states = 0
    last_state: dict[str, Any] | None = None
    last_reward: dict[str, Any] | None = None
    last_metadata: dict[str, Any] | None = None

    try:
        async for step in client.stream_rollout(
            CyclePolicy(),
            reward_engine=RewardEngine(build_goal(goal_preset)),
            max_states=max_states,
            trajectory_jsonl=trajectory_jsonl,
            rollout_metadata={
                "source": "codex-mcp",
                "endpoint_host": endpoint,
                "goal_preset": goal_preset,
                "max_states": max_states,
                "token_present": False,
            },
        ):
            states += 1
            total_reward += step.reward.reward
            last_state = step.state_summary
            last_reward = step.reward_summary
            last_metadata = step.metadata
    finally:
        await client.close()

    return {
        "states": states,
        "total_reward": total_reward,
        "last_seen_tick": client.last_seen_tick,
        "last_state": last_state,
        "last_reward": last_reward,
        "metadata": last_metadata,
        "trajectory_jsonl": trajectory_jsonl,
    }


def _ensure_agent_path() -> None:
    import sys

    path = str(AGENT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _run(
    argv: list[str],
    *,
    check: bool = True,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    result = {
        "command": argv,
        "returncode": completed.returncode,
        "stdout": _tail(completed.stdout),
        "stderr": _tail(completed.stderr),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if check and completed.returncode != 0:
        raise RuntimeError(result)
    return result


def _wait_for_tcp(host: str, port: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _tcp_connect(host, port, timeout_seconds=1.0):
            return True
        time.sleep(0.5)
    return False


def _tcp_connect(host: str, port: int, *, timeout_seconds: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _viewer_process() -> int | None:
    try:
        pid = int(VIEWER_PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    if _pid_running(pid):
        return pid
    VIEWER_PID_FILE.unlink(missing_ok=True)
    return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _tail(value: str, *, limit: int = 6000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


if __name__ == "__main__":
    MCP.run()
