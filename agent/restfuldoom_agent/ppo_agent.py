"""CLI for PPO training over high-level Doom skills."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from .brain import AgentMemory
from .env import ACTION_SCHEMA, OBSERVATION_SCHEMA, DoomAgentEnv, DoomEnvConfig
from .ppo import PPOConfig, PPOTrainer, require_torch


async def train(args: argparse.Namespace) -> dict[str, object]:
    """Runs PPO collection and update batches."""
    require_torch()
    env_config = DoomEnvConfig(
        endpoint=args.endpoint,
        token=args.token,
        agent_port=args.agent_port,
        tls=args.tls,
        authority=args.authority,
        skill=args.skill,
        episode=args.episode,
        map=args.map,
        seed=args.seed,
        run_id=args.run_id,
        goal_preset=args.goal_preset,
        target_x_fp=args.target_x_fp,
        target_y_fp=args.target_y_fp,
        max_steps=args.max_steps,
        level_complete_bonus=args.level_complete_bonus,
        kill_goal_bonus=args.kill_goal_bonus,
        required_kills=args.required_kills,
        memory_path=args.memory_path,
    )
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
    )
    env = DoomAgentEnv(env_config)
    memory = AgentMemory.load(args.memory_path) if args.memory_path is not None else None
    trainer = PPOTrainer(
        obs_dim=len(OBSERVATION_SCHEMA["feature_names"]),
        action_dim=len(ACTION_SCHEMA["actions"]),
        config=ppo_config,
        device=args.device,
    )
    summaries = []
    try:
        for update_index in range(args.updates):
            buffer = await trainer.collect_rollout(
                env,
                steps=args.rollout_steps,
                seed=args.seed + update_index,
            )
            buffer_path = args.buffer_dir / f"{args.run_id}-buffer-{update_index:04d}.jsonl"
            buffer.save_jsonl(buffer_path)
            rollout_summary = _summarize_buffer(buffer)
            metrics = trainer.update(buffer)
            checkpoint_path = args.checkpoint_dir / f"{args.run_id}-ppo-{update_index:04d}.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                reward_config={
                    "goal_preset": args.goal_preset,
                    "target_x_fp": args.target_x_fp,
                    "target_y_fp": args.target_y_fp,
                    "level_complete_bonus": args.level_complete_bonus,
                    "kill_goal_bonus": args.kill_goal_bonus,
                    "required_kills": args.required_kills,
                },
                extra={
                    "buffer_path": str(buffer_path),
                    "endpoint": args.endpoint,
                    "episode": args.episode,
                    "map": args.map,
                    "skill": args.skill,
                    "rollout_summary": rollout_summary,
                },
            )
            if memory is not None:
                _record_ppo_checkpoint(
                    memory,
                    checkpoint_path,
                    goal_preset=args.goal_preset,
                    reward_config={
                        "goal_preset": args.goal_preset,
                        "target_x_fp": args.target_x_fp,
                        "target_y_fp": args.target_y_fp,
                        "level_complete_bonus": args.level_complete_bonus,
                        "kill_goal_bonus": args.kill_goal_bonus,
                        "required_kills": args.required_kills,
                    },
                    metrics=metrics,
                    rollout_summary=rollout_summary,
                    update_index=update_index,
                    buffer_path=buffer_path,
                )
            summaries.append(
                {
                    "update": update_index,
                    "records": len(buffer),
                    "buffer_path": str(buffer_path),
                    "checkpoint_path": str(checkpoint_path),
                    "metrics": metrics,
                    "rollout_summary": rollout_summary,
                }
            )
    finally:
        await env.close()
    return {
        "schema": "restfuldoom.ppo_training_run.v1",
        "run_id": args.run_id,
        "updates": summaries,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
    }


def _record_ppo_checkpoint(
    memory: AgentMemory,
    checkpoint_path: Path,
    *,
    goal_preset: str,
    reward_config: dict[str, object],
    metrics: dict[str, float],
    rollout_summary: dict[str, object],
    update_index: int,
    buffer_path: Path,
) -> None:
    """Records the latest PPO checkpoint for export/resume."""
    record = {
        "schema": "restfuldoom.ppo_policy.v1",
        "checkpoint_path": str(checkpoint_path),
        "goal_preset": goal_preset,
        "reward_config": reward_config,
        "metrics": metrics,
        "rollout_summary": rollout_summary,
        "update_index": update_index,
        "buffer_path": str(buffer_path),
        "eval_history": [],
    }
    memory.data["ppo_policy"] = record
    checkpoints = memory.data.setdefault("ppo_checkpoints", [])
    checkpoints.append(
        {
            "checkpoint_path": str(checkpoint_path),
            "update_index": update_index,
            "buffer_path": str(buffer_path),
            "rollout_summary": rollout_summary,
        }
    )
    memory.data["updated_at"] = _iso_now()
    memory.save()


def _summarize_buffer(buffer: object) -> dict[str, object]:
    records = getattr(buffer, "records", [])
    skills = Counter(
        record.info.get("skill", "unknown")
        for record in records
        if isinstance(record.info, dict)
    )
    transitions = [
        record.info.get("transition", {})
        for record in records
        if isinstance(record.info, dict)
    ]
    states = [
        record.info.get("state", {})
        for record in records
        if isinstance(record.info, dict)
    ]
    return {
        "records": len(records),
        "total_reward": round(sum(float(record.reward) for record in records), 4),
        "positive_reward_steps": sum(1 for record in records if record.reward > 0),
        "negative_reward_steps": sum(1 for record in records if record.reward < 0),
        "done_count": sum(1 for record in records if record.done),
        "damage_delta": sum(int(transition.get("damage_delta", 0)) for transition in transitions),
        "enemy_distance_delta": round(
            sum(float(transition.get("enemy_distance_delta", 0.0)) for transition in transitions),
            4,
        ),
        "max_kills": max((int(state.get("kills", 0)) for state in states), default=0),
        "min_health": min((int(state.get("health", 0)) for state in states), default=0),
        "skill_counts": dict(sorted(skills.items())),
    }


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    """Entrypoint for PPO training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--token")
    parser.add_argument("--agent-port", type=int, default=50051)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--authority")
    parser.add_argument("--skill", type=int, default=2)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--map", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-id", default="ppo-local")
    parser.add_argument("--goal-preset", default="combat")
    parser.add_argument("--target-x-fp", type=int)
    parser.add_argument("--target-y-fp", type=int)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--required-kills", type=int, default=1)
    parser.add_argument("--level-complete-bonus", type=float, default=100.0)
    parser.add_argument("--kill-goal-bonus", type=float, default=10.0)
    parser.add_argument("--memory-path", type=Path, default=Path("agent_memory/e1m1.json"))
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("agent_models/ppo"))
    parser.add_argument("--buffer-dir", type=Path, default=Path("trajectories/ppo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(train(args)), sort_keys=True))


if __name__ == "__main__":
    main()
