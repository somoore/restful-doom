"""Runs a deterministic gRPC smoke agent against Doom."""

from __future__ import annotations

import argparse
import asyncio
import json
from itertools import cycle
from pathlib import Path
import sys

from .client import BackoffConfig, DoomAgentClient, action_cycle
from .reward import GOAL_PRESETS, RewardEngine, goal_preset


class CyclePolicy:
    """Cycles through deterministic actions."""

    def __init__(self) -> None:
        self.actions = cycle(action_cycle())

    async def next_action(self, _state):
        """Returns the next deterministic action."""
        return next(self.actions)


async def run(
    endpoint: str,
    max_states: int,
    trajectory_jsonl: Path | None,
    goal_name: str,
    target_x_fp: int | None,
    target_y_fp: int | None,
    reconnect: bool,
    backoff: BackoffConfig,
) -> None:
    """Runs the smoke agent."""
    client = DoomAgentClient(endpoint)
    policy = CyclePolicy()
    goal = (
        None
        if goal_name == "custom"
        else goal_preset(goal_name, target_x_fp=target_x_fp, target_y_fp=target_y_fp)
    )
    reward = RewardEngine(goal)
    total_reward = 0.0
    states = 0
    last_summary = {"states": 0}
    reconnect_events = 0

    async def report_reconnect(info):
        nonlocal reconnect_events
        reconnect_events += 1
        print(
            json.dumps(
                {
                    "event": "reconnect",
                    "attempt": info.attempt,
                    "delay_seconds": info.delay_seconds,
                    "last_seen_tick": info.last_seen_tick,
                    "code": info.code,
                    "details": info.details,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )

    try:
        async for step in client.stream_rollout(
            policy,
            reward_engine=reward,
            max_states=max_states,
            trajectory_jsonl=trajectory_jsonl,
            reconnect=reconnect,
            backoff=backoff,
            on_reconnect=report_reconnect,
        ):
            total_reward += step.reward.reward
            last_summary = step.state_summary
            states += 1

        summary = last_summary
        summary["states"] = states
        summary["total_reward"] = total_reward
        summary["goal"] = goal.name if goal is not None else "custom"
        summary["last_seen_tick"] = client.last_seen_tick
        summary["reconnect_events"] = reconnect_events
        print(json.dumps(summary, sort_keys=True))
    finally:
        await client.close()


def main() -> None:
    """Runs the command-line smoke agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--max-states", type=int, default=35)
    parser.add_argument("--trajectory-jsonl", type=Path)
    parser.add_argument(
        "--goal-preset",
        choices=["custom", *sorted(GOAL_PRESETS)],
        default="custom",
    )
    parser.add_argument("--target-x-fp", type=int)
    parser.add_argument("--target-y-fp", type=int)
    parser.add_argument("--no-reconnect", action="store_true")
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--backoff-initial", type=float, default=0.25)
    parser.add_argument("--backoff-max", type=float, default=5.0)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.endpoint,
            args.max_states,
            args.trajectory_jsonl,
            args.goal_preset,
            args.target_x_fp,
            args.target_y_fp,
            not args.no_reconnect,
            BackoffConfig(
                initial_seconds=args.backoff_initial,
                max_seconds=args.backoff_max,
                max_attempts=args.max_reconnects,
            ),
        )
    )


if __name__ == "__main__":
    main()
