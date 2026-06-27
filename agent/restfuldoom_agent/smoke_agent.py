"""Runs a deterministic gRPC smoke agent against Doom."""

from __future__ import annotations

import argparse
import asyncio
import json
from itertools import cycle
from pathlib import Path
from typing import Any

from .client import DoomAgentClient, action_cycle, agent_pb2, summarize_state
from .reward import RewardEngine


class CyclePolicy:
    """Cycles through deterministic actions."""

    def __init__(self) -> None:
        self.actions = cycle(action_cycle())

    async def next_action(self, _state):
        """Returns the next deterministic action."""
        return next(self.actions)


def summarize_action(action: Any) -> dict[str, Any]:
    """Converts a PlayerAction message into a compact JSON record."""
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


async def run(endpoint: str, max_states: int, trajectory_jsonl: Path | None) -> None:
    """Runs the smoke agent."""
    client = DoomAgentClient(endpoint)
    policy = CyclePolicy()
    reward = RewardEngine()
    prior = None
    total_reward = 0.0
    states = 0
    last_summary = {"states": 0}
    action_queue: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=16)
    if trajectory_jsonl is not None:
        trajectory_jsonl.parent.mkdir(parents=True, exist_ok=True)
    trajectory = trajectory_jsonl.open("w", encoding="utf-8") if trajectory_jsonl else None

    async def action_iter():
        yield agent_pb2.PlayerAction()
        while True:
            action = await action_queue.get()
            if action is None:
                break
            yield action

    try:
        async for state in client.session(action_iter()):
            transition = reward.score(prior, state)
            total_reward += transition.reward
            action = await policy.next_action(state)
            last_summary = summarize_state(state)
            states += 1

            if trajectory is not None:
                trajectory.write(
                    json.dumps(
                        {
                            "state": last_summary,
                            "reward": summarize_reward(transition),
                            "next_action": summarize_action(action),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            if action is not None:
                await action_queue.put(action)
            prior = state

            if states >= max_states:
                break

        summary = last_summary
        summary["states"] = states
        summary["total_reward"] = total_reward
        print(json.dumps(summary, sort_keys=True))
    finally:
        await action_queue.put(None)
        if trajectory is not None:
            trajectory.close()
        await client.close()


def main() -> None:
    """Runs the command-line smoke agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--max-states", type=int, default=35)
    parser.add_argument("--trajectory-jsonl", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.endpoint, args.max_states, args.trajectory_jsonl))


if __name__ == "__main__":
    main()
