"""Async Python client for the RESTful Doom gRPC bridge."""

from __future__ import annotations

import asyncio
import os
import sys
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

    async def close(self) -> None:
        """Closes the underlying channel."""
        await self.channel.close()

    async def observe(self, *, include_delta_state: bool = True) -> AsyncIterator[Any]:
        """Streams observations without sending actions."""
        request = agent_pb2.ObserveRequest(include_delta_state=include_delta_state)
        async for state in self.stub.Observe(request):
            yield state

    async def session(self, actions: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Runs a bidirectional observe-act stream."""
        async for state in self.stub.GameSession(actions):
            yield state

    async def run_policy(
        self,
        policy: Any,
        *,
        max_states: int = 350,
        include_idle_action: bool = True,
    ) -> list[Any]:
        """Runs a policy against Doom and returns observed states."""
        action_queue: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=16)

        async def action_iter() -> AsyncIterator[Any]:
            if include_idle_action:
                yield agent_pb2.PlayerAction()
            while True:
                action = await action_queue.get()
                if action is None:
                    break
                yield action

        states: list[Any] = []
        try:
            async for state in self.session(action_iter()):
                states.append(state)
                action = await policy.next_action(state)
                if action is not None:
                    await action_queue.put(action)
                if len(states) >= max_states:
                    break
        finally:
            await action_queue.put(None)

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


def action_cycle() -> Iterable[Any]:
    """Returns a deterministic smoke-test action cycle."""
    yield semantic_action(agent_pb2.ACTION_FORWARD, amount=25, duration_tics=4)
    yield semantic_action(agent_pb2.ACTION_TURN_RIGHT, amount=10, duration_tics=2)
    yield semantic_action(agent_pb2.ACTION_SHOOT, duration_tics=1)
    yield semantic_action(agent_pb2.ACTION_USE, duration_tics=1)
