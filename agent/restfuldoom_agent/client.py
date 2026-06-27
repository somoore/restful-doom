"""Async Python client for the RESTful Doom gRPC bridge."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from dataclasses import dataclass
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import grpc


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


class DoomAgentClient:
    """Maintains an async gRPC connection to Doom."""

    def __init__(self, endpoint: str = "127.0.0.1:50051") -> None:
        self.endpoint = endpoint
        self.channel = grpc.aio.insecure_channel(endpoint)
        self.stub = agent_pb2_grpc.DoomAgentStub(self.channel)
        self.last_seen_tick: int | None = None

    async def close(self) -> None:
        """Closes the underlying channel."""
        await self.channel.close()

    async def observe(self, *, include_delta_state: bool = True) -> AsyncIterator[Any]:
        """Streams observations without sending actions."""
        request = agent_pb2.ObserveRequest(include_delta_state=include_delta_state)
        async for state in self.stub.Observe(request):
            self.last_seen_tick = state.tick
            yield state

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
        async for state in self.stub.GameSession(actions):
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
    ) -> AsyncIterator[RolloutStep]:
        """Streams a policy rollout without retaining states in memory."""
        from .reward import RewardEngine

        reward_engine = reward_engine or RewardEngine()
        backoff = backoff or BackoffConfig()
        previous_state = None
        index = 0
        reconnect_attempts = 0
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
                        action = await policy.next_action(state)
                        if action is not None:
                            await action_queue.put(action)

                        state_summary = summarize_state(state)
                        reward_summary = summarize_reward(transition)
                        action_summary = summarize_action(action)
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
