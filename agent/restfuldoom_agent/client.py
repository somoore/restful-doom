"""Async Python client for the RESTful Doom gRPC bridge."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import grpc

DEFAULT_AGENT_PORT = 50051
AUTH_METADATA_KEY = "x-aws-proxy-auth"
PORT_METADATA_KEY = "x-aws-proxy-port"


def _generated_root() -> Path:
    override = os.environ.get("RESTFULDOOM_PROTO_STUBS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "generated"


def _load_generated() -> tuple[Any, Any]:
    generated = _generated_root()
    if str(generated) not in sys.path:
        sys.path.insert(0, str(generated))
    try:
        from restfuldoom.v1 import agent_pb2, agent_pb2_grpc
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Python protobuf stubs are missing. Run "
            "`python -m restfuldoom_agent.generate_stubs` from the agent env."
        ) from error
    return agent_pb2, agent_pb2_grpc


agent_pb2, agent_pb2_grpc = _load_generated()


@dataclass(frozen=True)
class BackoffConfig:
    """Reconnect timing for long-running rollouts."""

    initial_seconds: float = 0.25
    max_seconds: float = 5.0
    multiplier: float = 2.0
    max_attempts: int = 5

    def delay_for_attempt(self, attempt: int) -> float:
        """Returns the delay before `attempt`, where attempt starts at 1."""
        delay = self.initial_seconds * (self.multiplier ** max(0, attempt - 1))
        return min(delay, self.max_seconds)


@dataclass(frozen=True)
class ReconnectInfo:
    """Reports a reconnect attempt and the last observed Doom tick."""

    attempt: int
    delay_seconds: float
    last_seen_tick: int | None
    code: str
    details: str


@dataclass(frozen=True)
class RolloutStep:
    """One streamed observe-act-reward step."""

    index: int
    state: Any
    state_summary: dict[str, Any]
    reward: Any
    reward_summary: dict[str, Any]
    next_action: Any
    action_summary: dict[str, Any] | None
    last_seen_tick: int
    reconnect_attempts: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EpisodeReset:
    """Accepted episode reset details from the Doom bridge."""

    accepted: bool
    message: str
    skill: int
    episode: int
    map: int
    seed: int
    seed_applied: bool
    start_queued: bool


@dataclass(frozen=True)
class SnapshotCommand:
    """Accepted snapshot save/load details from the Doom bridge."""

    accepted: bool
    message: str
    slot: int
    save_queued: bool
    load_queued: bool


@dataclass(frozen=True)
class EpisodeStart:
    """Optional post-reset curriculum start."""

    x_fp: int | None = None
    y_fp: int | None = None
    angle_degrees: int = 0
    face_nearest_enemy: bool = False
    health: int | None = None
    armor: int | None = None
    ammo_bullets: int | None = None
    ammo_shells: int | None = None
    ammo_cells: int | None = None
    ammo_rockets: int | None = None


class DoomAgentStreamError(RuntimeError):
    """Raised when a stream ends after reconnect attempts are exhausted."""

    def __init__(self, message: str, *, last_seen_tick: int | None) -> None:
        super().__init__(message)
        self.last_seen_tick = last_seen_tick


def semantic_action(
    action: int,
    *,
    amount: int = 0,
    duration_tics: int = 1,
    tick: int = 0,
) -> Any:
    """Creates a high-level player action."""
    return agent_pb2.PlayerAction(
        tick=tick,
        action=action,
        amount=amount,
        duration_tics=duration_tics,
    )


def _episode_start_proto(start: EpisodeStart | None) -> Any | None:
    if start is None:
        return None

    position = None
    if start.x_fp is not None and start.y_fp is not None:
        position = agent_pb2.Vec3Fixed(
            x_fp=int(start.x_fp),
            y_fp=int(start.y_fp),
            z_fp=0,
        )

    apply_resources = any(
        value is not None
        for value in (
            start.health,
            start.armor,
            start.ammo_bullets,
            start.ammo_shells,
            start.ammo_cells,
            start.ammo_rockets,
        )
    )
    ammo = agent_pb2.Ammo(
        bullets=0 if start.ammo_bullets is None else int(start.ammo_bullets),
        shells=0 if start.ammo_shells is None else int(start.ammo_shells),
        cells=0 if start.ammo_cells is None else int(start.ammo_cells),
        rockets=0 if start.ammo_rockets is None else int(start.ammo_rockets),
    )
    return agent_pb2.EpisodeStart(
        position=position,
        angle_degrees=int(start.angle_degrees) % 360,
        face_nearest_enemy=bool(start.face_nearest_enemy),
        health=100 if start.health is None else int(start.health),
        armor=0 if start.armor is None else int(start.armor),
        ammo=ammo,
        apply_resources=apply_resources,
    )


class DoomAgentClient:
    """Maintains an async gRPC connection to Doom."""

    def __init__(
        self,
        endpoint: str = "127.0.0.1:50051",
        *,
        token: str | None = None,
        agent_port: int | None = None,
        tls: bool = False,
        authority: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.metadata = _client_metadata(token=token, agent_port=agent_port)
        options = _channel_options(authority)
        if tls:
            self.channel = grpc.aio.secure_channel(
                endpoint,
                grpc.ssl_channel_credentials(),
                options=options,
            )
        else:
            self.channel = grpc.aio.insecure_channel(endpoint, options=options)
        self.stub = agent_pb2_grpc.DoomAgentStub(self.channel)
        self.last_seen_tick: int | None = None

    async def close(self) -> None:
        """Closes the underlying channel."""
        await self.channel.close()

    async def observe(self, *, include_delta_state: bool = True) -> AsyncIterator[Any]:
        """Streams observations without sending actions."""
        request = agent_pb2.ObserveRequest(include_delta_state=include_delta_state)
        async for state in self.stub.Observe(request, metadata=self.metadata):
            self.last_seen_tick = state.tick
            yield state

    async def reset_episode(
        self,
        *,
        skill: int = 2,
        episode: int = 1,
        map: int = 1,
        seed: int = 0,
        run_id: str = "",
        start: EpisodeStart | None = None,
    ) -> EpisodeReset:
        """Queues a fast episode reset on Doom's simulation thread."""
        response = await self.stub.ResetEpisode(
            agent_pb2.ResetEpisodeRequest(
                skill=int(skill),
                episode=int(episode),
                map=int(map),
                seed=int(seed),
                run_id=run_id,
                start=_episode_start_proto(start),
            ),
            metadata=self.metadata,
        )
        return EpisodeReset(
            accepted=bool(response.accepted),
            message=str(response.message),
            skill=int(response.skill),
            episode=int(response.episode),
            map=int(response.map),
            seed=int(response.seed),
            seed_applied=bool(response.seed_applied),
            start_queued=bool(getattr(response, "start_queued", False)),
        )

    async def save_snapshot(
        self,
        *,
        slot: int,
        description: str = "",
        run_id: str = "",
    ) -> SnapshotCommand:
        """Queues a local savegame snapshot on Doom's simulation thread."""
        response = await self.stub.SaveSnapshot(
            agent_pb2.SaveSnapshotRequest(
                slot=int(slot),
                description=description,
                run_id=run_id,
            ),
            metadata=self.metadata,
        )
        return SnapshotCommand(
            accepted=bool(response.accepted),
            message=str(response.message),
            slot=int(response.slot),
            save_queued=bool(response.save_queued),
            load_queued=bool(response.load_queued),
        )

    async def load_snapshot(
        self,
        *,
        slot: int,
        run_id: str = "",
    ) -> SnapshotCommand:
        """Queues a local savegame restore on Doom's simulation thread."""
        response = await self.stub.LoadSnapshot(
            agent_pb2.LoadSnapshotRequest(
                slot=int(slot),
                run_id=run_id,
            ),
            metadata=self.metadata,
        )
        return SnapshotCommand(
            accepted=bool(response.accepted),
            message=str(response.message),
            slot=int(response.slot),
            save_queued=bool(response.save_queued),
            load_queued=bool(response.load_queued),
        )

    async def observe_reconnecting(
        self,
        *,
        include_delta_state: bool = True,
        backoff: BackoffConfig | None = None,
        on_reconnect: Any | None = None,
    ) -> AsyncIterator[Any]:
        """Streams observations and reconnects with backoff on gRPC errors."""
        backoff = backoff or BackoffConfig()
        attempt = 0

        while True:
            stream_ended = False
            try:
                async for state in self.observe(include_delta_state=include_delta_state):
                    attempt = 0
                    yield state
                stream_ended = True
            except grpc.aio.AioRpcError as error:
                attempt += 1
                if attempt > backoff.max_attempts:
                    raise DoomAgentStreamError(
                        "observe stream ended after reconnect attempts were exhausted",
                        last_seen_tick=self.last_seen_tick,
                    ) from error

                info = _reconnect_info(
                    attempt,
                    backoff,
                    self.last_seen_tick,
                    error.code().name,
                    error.details() or "",
                )
                await _maybe_call(on_reconnect, info)
                await asyncio.sleep(info.delay_seconds)

            if stream_ended:
                attempt += 1
                if attempt > backoff.max_attempts:
                    raise DoomAgentStreamError(
                        "observe stream ended after reconnect attempts were exhausted",
                        last_seen_tick=self.last_seen_tick,
                    )

                info = _reconnect_info(
                    attempt,
                    backoff,
                    self.last_seen_tick,
                    "EOF",
                    "observe stream ended without a gRPC error",
                )
                await _maybe_call(on_reconnect, info)
                await asyncio.sleep(info.delay_seconds)

    async def session(self, actions: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Runs a bidirectional observe-act stream."""
        async for state in self.stub.GameSession(actions, metadata=self.metadata):
            self.last_seen_tick = state.tick
            yield state

    async def stream_rollout(
        self,
        policy: Any,
        *,
        reward_engine: Any | None = None,
        max_states: int | None = 350,
        include_idle_action: bool = True,
        action_queue_size: int = 16,
        trajectory_jsonl: str | Path | None = None,
        reconnect: bool = True,
        backoff: BackoffConfig | None = None,
        on_reconnect: Any | None = None,
        rollout_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[RolloutStep]:
        """Streams a policy rollout without retaining states in memory."""
        from .reward import RewardEngine

        reward_engine = reward_engine or RewardEngine()
        backoff = backoff or BackoffConfig()
        rollout_metadata = dict(rollout_metadata or {})
        previous_state = None
        index = 0
        reconnect_attempts = 0
        reconnect_count = 0
        policy_errors = 0
        bedrock_fallback_count = 0
        trajectory = _open_trajectory(trajectory_jsonl)

        try:
            while max_states is None or index < max_states:
                action_queue: asyncio.Queue[Any | None] = asyncio.Queue(
                    maxsize=action_queue_size
                )
                stream_ended = False

                async def action_iter() -> AsyncIterator[Any]:
                    if include_idle_action:
                        yield agent_pb2.PlayerAction()
                    while True:
                        action = await action_queue.get()
                        if action is None:
                            break
                        yield action

                try:
                    async for state in self.session(action_iter()):
                        transition = reward_engine.score(previous_state, state)
                        policy_started = time.perf_counter()
                        action = await policy.next_action(state)
                        policy_latency_ms = (time.perf_counter() - policy_started) * 1000.0
                        if action is not None:
                            await action_queue.put(action)

                        state_summary = summarize_state(state)
                        reward_summary = summarize_reward(transition)
                        action_summary = summarize_action(action)
                        metadata = _rollout_step_metadata(
                            rollout_metadata=rollout_metadata,
                            reconnect_count=reconnect_count,
                            reconnect_attempts=reconnect_attempts,
                            policy=policy,
                            policy_errors=policy_errors,
                            bedrock_fallback_count=bedrock_fallback_count,
                            policy_latency_ms=policy_latency_ms,
                        )
                        policy_errors = metadata["policy_errors"]
                        bedrock_fallback_count = metadata["bedrock_fallback_count"]
                        step = RolloutStep(
                            index=index,
                            state=state,
                            state_summary=state_summary,
                            reward=transition,
                            reward_summary=reward_summary,
                            next_action=action,
                            action_summary=action_summary,
                            last_seen_tick=state.tick,
                            reconnect_attempts=reconnect_attempts,
                            metadata=metadata,
                        )

                        if trajectory is not None:
                            trajectory.write(
                                json.dumps(
                                    {
                                        "index": step.index,
                                        "state": state_summary,
                                        "reward": reward_summary,
                                        "next_action": action_summary,
                                        "last_seen_tick": step.last_seen_tick,
                                        "reconnect_attempts": reconnect_attempts,
                                        "metadata": metadata,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            trajectory.flush()

                        yield step

                        previous_state = state
                        index += 1
                        reconnect_attempts = 0
                        if max_states is not None and index >= max_states:
                            return
                    stream_ended = True
                except grpc.aio.AioRpcError as error:
                    if not reconnect:
                        raise DoomAgentStreamError(
                            "game session stream ended and reconnect is disabled",
                            last_seen_tick=self.last_seen_tick,
                        ) from error

                    reconnect_attempts += 1
                    reconnect_count += 1
                    if reconnect_attempts > backoff.max_attempts:
                        raise DoomAgentStreamError(
                            "game session stream ended after reconnect attempts were exhausted",
                            last_seen_tick=self.last_seen_tick,
                        ) from error

                    info = _reconnect_info(
                        reconnect_attempts,
                        backoff,
                        self.last_seen_tick,
                        error.code().name,
                        error.details() or "",
                    )
                    await _maybe_call(on_reconnect, info)
                    await asyncio.sleep(info.delay_seconds)
                finally:
                    _stop_action_iter(action_queue)

                if stream_ended:
                    if not reconnect:
                        raise DoomAgentStreamError(
                            "game session stream ended and reconnect is disabled",
                            last_seen_tick=self.last_seen_tick,
                        )

                    reconnect_attempts += 1
                    reconnect_count += 1
                    if reconnect_attempts > backoff.max_attempts:
                        raise DoomAgentStreamError(
                            "game session stream ended after reconnect attempts were exhausted",
                            last_seen_tick=self.last_seen_tick,
                        )

                    info = _reconnect_info(
                        reconnect_attempts,
                        backoff,
                        self.last_seen_tick,
                        "EOF",
                        "game session stream ended without a gRPC error",
                    )
                    await _maybe_call(on_reconnect, info)
                    await asyncio.sleep(info.delay_seconds)
        finally:
            if trajectory is not None:
                trajectory.close()

    async def run_policy(
        self,
        policy: Any,
        *,
        max_states: int = 350,
        include_idle_action: bool = True,
    ) -> list[Any]:
        """Runs a policy and returns states. Prefer stream_rollout for long runs."""
        states: list[Any] = []
        async for step in self.stream_rollout(
            policy,
            max_states=max_states,
            include_idle_action=include_idle_action,
        ):
            states.append(step.state)
        return states


def summarize_state(state: Any) -> dict[str, Any]:
    """Converts a protobuf state to a compact JSON-serializable summary."""
    player = state.player
    obj = player.object
    position = obj.position
    navigation = getattr(state, "navigation", None)
    combat = getattr(state, "combat", None)
    return {
        "tick": state.tick,
        "episode": state.level.episode,
        "map": state.level.map,
        "health": player.health,
        "armor": player.armor,
        "kills": player.kills,
        "items": player.items,
        "secrets": player.secrets,
        "weapon": player.ready_weapon,
        "position_fp": [position.x_fp, position.y_fp, position.z_fp],
        "enemy_count": len(state.enemies),
        "object_count": len(state.objects),
        "has_delta_state": state.has_delta_state,
        "navigation": {
            "forward_open": bool(getattr(navigation, "forward_open", True)),
            "back_open": bool(getattr(navigation, "back_open", True)),
            "left_open": bool(getattr(navigation, "left_open", True)),
            "right_open": bool(getattr(navigation, "right_open", True)),
            "use_line_ahead": bool(getattr(navigation, "use_line_ahead", False)),
            "front_blocking_line_special": int(
                getattr(navigation, "front_blocking_line_special", 0)
            ),
            "front_block_distance_fp": int(
                getattr(navigation, "front_block_distance_fp", 0)
            ),
            "probe_distance_fp": int(getattr(navigation, "probe_distance_fp", 0)),
            "direction_probes": [
                {
                    "angle_offset_degrees": int(
                        getattr(probe, "angle_offset_degrees", 0)
                    ),
                    "open": bool(getattr(probe, "open", False)),
                    "block_distance_fp": int(getattr(probe, "block_distance_fp", 0)),
                    "blocking_line_special": int(
                        getattr(probe, "blocking_line_special", 0)
                    ),
                    "use_line_ahead": bool(getattr(probe, "use_line_ahead", False)),
                }
                for probe in getattr(navigation, "direction_probes", [])
            ],
            "use_lines": [
                {
                    "line_id": int(getattr(line, "line_id", 0)),
                    "midpoint_fp": [
                        int(getattr(getattr(line, "midpoint", None), "x_fp", 0)),
                        int(getattr(getattr(line, "midpoint", None), "y_fp", 0)),
                        int(getattr(getattr(line, "midpoint", None), "z_fp", 0)),
                    ],
                    "nearest_point_fp": [
                        int(getattr(getattr(line, "nearest_point", None), "x_fp", 0)),
                        int(getattr(getattr(line, "nearest_point", None), "y_fp", 0)),
                        int(getattr(getattr(line, "nearest_point", None), "z_fp", 0)),
                    ],
                    "start_fp": [
                        int(getattr(getattr(line, "start", None), "x_fp", 0)),
                        int(getattr(getattr(line, "start", None), "y_fp", 0)),
                        int(getattr(getattr(line, "start", None), "z_fp", 0)),
                    ],
                    "end_fp": [
                        int(getattr(getattr(line, "end", None), "x_fp", 0)),
                        int(getattr(getattr(line, "end", None), "y_fp", 0)),
                        int(getattr(getattr(line, "end", None), "z_fp", 0)),
                    ],
                    "special": int(getattr(line, "special", 0)),
                    "tag": int(getattr(line, "tag", 0)),
                    "distance_fp": int(getattr(line, "distance_fp", 0)),
                    "nearest_distance_fp": int(
                        getattr(line, "nearest_distance_fp", 0)
                    ),
                }
                for line in getattr(navigation, "use_lines", [])
            ],
            "current_sector": {
                "sector_id": int(
                    getattr(getattr(navigation, "current_sector", None), "sector_id", 0)
                ),
                "special": int(
                    getattr(getattr(navigation, "current_sector", None), "special", 0)
                ),
                "floor_height_fp": int(
                    getattr(
                        getattr(navigation, "current_sector", None),
                        "floor_height_fp",
                        0,
                    )
                ),
                "ceiling_height_fp": int(
                    getattr(
                        getattr(navigation, "current_sector", None),
                        "ceiling_height_fp",
                        0,
                    )
                ),
                "light_level": int(
                    getattr(
                        getattr(navigation, "current_sector", None),
                        "light_level",
                        0,
                    )
                ),
                "damaging": bool(
                    getattr(getattr(navigation, "current_sector", None), "damaging", False)
                ),
                "damage_per_32_tics": int(
                    getattr(
                        getattr(navigation, "current_sector", None),
                        "damage_per_32_tics",
                        0,
                    )
                ),
                "exit_damage": bool(
                    getattr(
                        getattr(navigation, "current_sector", None),
                        "exit_damage",
                        False,
                    )
                ),
            },
            "route_waypoint": _summarize_route_waypoint(
                getattr(navigation, "route_waypoint", None)
            ),
        },
        "combat": {
            "has_shootable_target": bool(
                getattr(combat, "has_shootable_target", False)
            ),
            "target_id": int(getattr(combat, "target_id", 0)),
            "target_health": int(getattr(combat, "target_health", 0)),
            "target_distance_fp": int(getattr(combat, "target_distance_fp", 0)),
            "aim_slope_fp": int(getattr(combat, "aim_slope_fp", 0)),
            "range_fp": int(getattr(combat, "range_fp", 0)),
            "target_is_enemy": bool(getattr(combat, "target_is_enemy", False)),
        },
    }


def _summarize_route_waypoint(route: Any | None) -> dict[str, Any]:
    if route is None or getattr(route, "line", None) is None:
        return {}
    line = route.line
    return {
        "line": {
            "line_id": int(getattr(line, "line_id", 0)),
            "midpoint_fp": [
                int(getattr(getattr(line, "midpoint", None), "x_fp", 0)),
                int(getattr(getattr(line, "midpoint", None), "y_fp", 0)),
                int(getattr(getattr(line, "midpoint", None), "z_fp", 0)),
            ],
            "nearest_point_fp": [
                int(getattr(getattr(line, "nearest_point", None), "x_fp", 0)),
                int(getattr(getattr(line, "nearest_point", None), "y_fp", 0)),
                int(getattr(getattr(line, "nearest_point", None), "z_fp", 0)),
            ],
            "special": int(getattr(line, "special", 0)),
            "tag": int(getattr(line, "tag", 0)),
            "distance_fp": int(getattr(line, "distance_fp", 0)),
            "nearest_distance_fp": int(getattr(line, "nearest_distance_fp", 0)),
        },
        "priority": int(getattr(route, "priority", 0)),
        "exit": bool(getattr(route, "exit", False)),
        "walk_trigger": bool(getattr(route, "walk_trigger", False)),
    }


def summarize_action(action: Any | None) -> dict[str, Any] | None:
    """Converts a PlayerAction message into a compact JSON record."""
    if action is None:
        return None

    raw = action.raw
    mouse = action.mouse
    return {
        "tick": action.tick,
        "action": int(action.action),
        "amount": action.amount,
        "duration_tics": action.duration_tics,
        "raw": {
            "forward_move": raw.forward_move,
            "side_move": raw.side_move,
            "angle_turn": raw.angle_turn,
            "buttons": raw.buttons,
        },
        "keys": [{"key": int(key.key), "pressed": key.pressed} for key in action.keys],
        "mouse": {
            "turn": mouse.turn,
            "forward": mouse.forward,
            "buttons": mouse.buttons,
        },
    }


def summarize_reward(transition: Any) -> dict[str, Any]:
    """Converts a reward transition into a compact JSON record."""
    return {
        "reward": transition.reward,
        "kill_delta": transition.kill_delta,
        "damage_delta": getattr(transition, "damage_delta", 0),
        "enemy_distance_delta": getattr(transition, "enemy_distance_delta", 0.0),
        "item_delta": transition.item_delta,
        "secret_delta": transition.secret_delta,
        "health_delta": transition.health_delta,
        "progress_delta": transition.progress_delta,
        "done": transition.done,
    }


def action_cycle() -> Iterable[Any]:
    """Returns a deterministic smoke-test action cycle."""
    yield semantic_action(agent_pb2.ACTION_FORWARD, amount=25, duration_tics=4)
    yield semantic_action(agent_pb2.ACTION_TURN_RIGHT, amount=10, duration_tics=2)
    yield semantic_action(agent_pb2.ACTION_SHOOT, duration_tics=1)
    yield semantic_action(agent_pb2.ACTION_USE, duration_tics=1)


def _open_trajectory(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    trajectory_path = Path(path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    return trajectory_path.open("w", encoding="utf-8")


def _reconnect_info(
    attempt: int,
    backoff: BackoffConfig,
    last_seen_tick: int | None,
    code: str,
    details: str,
) -> ReconnectInfo:
    return ReconnectInfo(
        attempt=attempt,
        delay_seconds=backoff.delay_for_attempt(attempt),
        last_seen_tick=last_seen_tick,
        code=code,
        details=details,
    )


async def _maybe_call(callback: Any | None, argument: Any) -> None:
    if callback is None:
        return
    result = callback(argument)
    if inspect.isawaitable(result):
        await result


def _stop_action_iter(queue: asyncio.Queue[Any | None]) -> None:
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:
        pass


def _rollout_step_metadata(
    *,
    rollout_metadata: dict[str, Any],
    reconnect_count: int,
    reconnect_attempts: int,
    policy: Any,
    policy_errors: int,
    bedrock_fallback_count: int,
    policy_latency_ms: float,
) -> dict[str, Any]:
    last_policy_error = getattr(policy, "last_error", None)
    if last_policy_error:
        policy_errors += 1

    reported_policy_errors = _int_attr(policy, "error_count")
    if reported_policy_errors is not None:
        policy_errors = max(policy_errors, reported_policy_errors)

    reported_fallbacks = _int_attr(policy, "fallback_count")
    if reported_fallbacks is not None:
        bedrock_fallback_count = max(bedrock_fallback_count, reported_fallbacks)

    llm_latency_ms = _float_attr(policy, "last_llm_latency_ms")
    return {
        "rollout": rollout_metadata,
        "reconnect_count": reconnect_count,
        "reconnect_attempts": reconnect_attempts,
        "policy_errors": policy_errors,
        "bedrock_fallback_count": bedrock_fallback_count,
        "policy_decision": _dict_attr(policy, "last_decision"),
        "last_token_usage": _dict_attr(policy, "last_token_usage"),
        "total_token_usage": _dict_attr(policy, "total_token_usage"),
        "policy_latency_ms": round(policy_latency_ms, 3),
        "llm_latency_ms": round(llm_latency_ms, 3) if llm_latency_ms is not None else None,
        "last_policy_error": last_policy_error,
    }


def _int_attr(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_attr(obj: Any, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_attr(obj: Any, name: str) -> dict[str, Any]:
    value = getattr(obj, name, None)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _client_metadata(
    *,
    token: str | None,
    agent_port: int | None,
) -> tuple[tuple[str, str], ...]:
    metadata: list[tuple[str, str]] = []
    if token:
        metadata.append((AUTH_METADATA_KEY, token))
        metadata.append((PORT_METADATA_KEY, str(agent_port or DEFAULT_AGENT_PORT)))
    elif agent_port is not None:
        metadata.append((PORT_METADATA_KEY, str(agent_port)))
    return tuple(metadata)


def _channel_options(authority: str | None) -> tuple[tuple[str, str], ...] | None:
    if not authority:
        return None
    return (("grpc.default_authority", authority),)
