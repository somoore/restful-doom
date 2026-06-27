"""Runs a deterministic gRPC smoke agent against Doom."""

from __future__ import annotations

import argparse
import asyncio
import json
from itertools import cycle
from pathlib import Path
import sys
import uuid

from .client import DoomAgentClient, DoomAgentStreamError, action_cycle
from .reward import GOAL_PRESETS, RewardEngine
from .rollout_config import RolloutConfig


class CyclePolicy:
    """Cycles through deterministic actions."""

    def __init__(self) -> None:
        self.actions = cycle(action_cycle())

    async def next_action(self, _state):
        """Returns the next deterministic action."""
        return next(self.actions)


async def run(config: RolloutConfig) -> None:
    """Runs the smoke agent."""
    if config.run_id is None:
        config = config.with_overrides(run_id=f"rollout-{uuid.uuid4().hex[:12]}")
    client = DoomAgentClient(
        config.endpoint,
        token=config.token,
        agent_port=config.agent_port,
        tls=config.use_tls(),
        authority=config.authority,
    )
    goal = config.goal()
    policy = build_policy(config)
    reward = RewardEngine(goal)
    total_reward = 0.0
    states = 0
    last_summary = {"states": 0}
    reconnect_events = 0
    last_metadata: dict[str, object] = {}

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
            max_states=config.max_states,
            trajectory_jsonl=config.trajectory_jsonl,
            reconnect=config.reconnect,
            backoff=config.backoff_config(),
            on_reconnect=report_reconnect,
            rollout_metadata=config.to_metadata(),
        ):
            total_reward += step.reward.reward
            last_summary = step.state_summary
            last_metadata = step.metadata
            states += 1

        summary = last_summary
        summary["states"] = states
        summary["total_reward"] = total_reward
        summary["goal"] = goal.name if goal is not None else "custom"
        summary["last_seen_tick"] = client.last_seen_tick
        summary["reconnect_events"] = reconnect_events
        summary["metadata"] = last_metadata
        print(json.dumps(summary, sort_keys=True))
    finally:
        await client.close()


def build_policy(config: RolloutConfig):
    """Creates the configured policy."""
    if config.policy == "bedrock":
        from .bedrock_policy import BedrockPolicy

        return BedrockPolicy(
            model_id=config.bedrock_model_id,
            timeout_seconds=config.bedrock_timeout,
            max_tokens=config.bedrock_max_tokens,
            mission=config.mission,
            goal_metadata=config.to_metadata(),
        )
    return CyclePolicy()


def main() -> None:
    """Runs the command-line smoke agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON rollout config file")
    parser.add_argument("--endpoint")
    parser.add_argument("--token", help="Hellbox/Shrink X-aws-proxy-auth token")
    parser.add_argument("--agent-port", type=int)
    parser.add_argument("--tls", dest="tls", action="store_true", default=None)
    parser.add_argument("--no-tls", dest="tls", action="store_false")
    parser.add_argument("--authority", help="Override gRPC TLS authority/SNI")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--trajectory-jsonl", type=Path)
    parser.add_argument(
        "--goal-preset",
        choices=["custom", *sorted(GOAL_PRESETS)],
    )
    parser.add_argument("--mission", help="Human-readable rollout objective")
    parser.add_argument("--target-x-fp", type=int)
    parser.add_argument("--target-y-fp", type=int)
    parser.add_argument("--reconnect", dest="reconnect", action="store_true", default=None)
    parser.add_argument("--no-reconnect", dest="reconnect", action="store_false")
    parser.add_argument("--max-reconnects", type=int)
    parser.add_argument("--backoff-initial", type=float)
    parser.add_argument("--backoff-max", type=float)
    parser.add_argument("--policy", choices=["cycle", "bedrock"])
    parser.add_argument("--bedrock-model-id")
    parser.add_argument("--bedrock-timeout", type=float)
    parser.add_argument("--bedrock-max-tokens", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--capsule-name")
    parser.add_argument("--capsule-id")
    parser.add_argument("--auth-lease-id")
    args = parser.parse_args()

    try:
        config = (
            RolloutConfig.from_json_file(args.config)
            if args.config is not None
            else RolloutConfig()
        )
        config = config.with_overrides(
            endpoint=args.endpoint,
            token=args.token,
            agent_port=args.agent_port,
            tls=args.tls,
            authority=args.authority,
            max_states=args.max_states,
            trajectory_jsonl=args.trajectory_jsonl,
            goal_preset=args.goal_preset,
            mission=args.mission,
            target_x_fp=args.target_x_fp,
            target_y_fp=args.target_y_fp,
            reconnect=args.reconnect,
            max_reconnects=args.max_reconnects,
            backoff_initial=args.backoff_initial,
            backoff_max=args.backoff_max,
            policy=args.policy,
            bedrock_model_id=args.bedrock_model_id,
            bedrock_timeout=args.bedrock_timeout,
            bedrock_max_tokens=args.bedrock_max_tokens,
            run_id=args.run_id,
            capsule_name=args.capsule_name,
            capsule_id=args.capsule_id,
            auth_lease_id=args.auth_lease_id,
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        asyncio.run(run(config))
    except DoomAgentStreamError as error:
        print(
            json.dumps(
                {
                    "event": "stream_error",
                    "error": str(error),
                    "last_seen_tick": error.last_seen_tick,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
