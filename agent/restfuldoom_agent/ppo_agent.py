"""CLI for PPO training over high-level Doom skills."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .brain import AgentMemory
from .curriculum import build_curriculum, curriculum_names, stage_for_update
from .env import ACTION_SCHEMA, OBSERVATION_SCHEMA, DoomAgentEnv, DoomEnvConfig
from .ppo import PPOConfig, PPOTrainer, require_torch
from .ppo_eval import (
    decide_promotion,
    evaluate_checkpoint,
    evaluate_heuristic_policy,
    evaluate_random_policy,
)
from .schemas import map_expert_skill_to_ppo_action, pad_observation_features
from .skill_policy import features_from_record


async def train(args: argparse.Namespace) -> dict[str, object]:
    """Runs PPO collection and update batches."""
    require_torch()
    reset_start = _resolve_reset_start(args)
    curriculum = build_curriculum(
        name=args.curriculum,
        manual_reset_start=reset_start,
        mode=args.curriculum_mode,
        start_index=args.curriculum_start_index,
        seed=args.seed,
    )
    env_config = _env_config_for_start(
        args,
        stage_for_update(curriculum, 0).get("reset_start", {}),
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
    memory = AgentMemory.load(args.memory_path) if args.memory_path is not None else None
    resume_checkpoint = _resolve_resume_checkpoint(args, memory)
    env = DoomAgentEnv(env_config)
    if resume_checkpoint is not None:
        trainer = PPOTrainer.load_checkpoint(
            resume_checkpoint,
            device=args.device,
            target_obs_dim=len(OBSERVATION_SCHEMA["feature_names"]),
            target_action_dim=len(ACTION_SCHEMA["actions"]),
        )
    else:
        trainer = PPOTrainer(
            obs_dim=len(OBSERVATION_SCHEMA["feature_names"]),
            action_dim=len(ACTION_SCHEMA["actions"]),
            config=ppo_config,
            device=args.device,
        )
    behavior_clone_summary = None
    if args.bc_trajectory:
        samples, behavior_clone_summary = _load_behavior_clone_samples(args)
        behavior_clone_summary.update(
            trainer.pretrain_actor(
                samples,
                epochs=args.bc_epochs,
                minibatch_size=args.bc_batch_size,
                learning_rate=args.bc_learning_rate,
            )
        )
    summaries = []
    best_checkpoint: dict[str, object] | None = None
    try:
        for update_index in range(args.updates):
            curriculum_stage = stage_for_update(curriculum, update_index)
            env.config = replace(
                _env_config_for_start(args, curriculum_stage.get("reset_start", {})),
                run_id=f"{args.run_id}-{curriculum_stage['name']}",
            )
            buffer = await trainer.collect_rollout(
                env,
                steps=args.rollout_steps,
                seed=args.seed + update_index,
            )
            _annotate_buffer_curriculum(buffer, curriculum, curriculum_stage)
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
                    "first_visible_bonus": args.first_visible_bonus,
                    "first_shootable_bonus": args.first_shootable_bonus,
                    "visible_contact_progress_reward": args.visible_contact_progress_reward,
                    "terminate_on_first_visible": args.terminate_on_first_visible,
                    "terminate_on_first_shootable": args.terminate_on_first_shootable,
                },
                extra={
                    "buffer_path": str(buffer_path),
                    "endpoint": args.endpoint,
                    "episode": args.episode,
                    "map": args.map,
                    "skill": args.skill,
                    "rollout_summary": rollout_summary,
                    "behavior_clone": behavior_clone_summary or {},
                    "reset_start": curriculum_stage.get("reset_start", {}),
                    "curriculum": curriculum,
                    "curriculum_stage": curriculum_stage,
                    "resume_checkpoint": str(resume_checkpoint)
                    if resume_checkpoint is not None
                    else None,
                    "resume_migration": trainer.resume_migration,
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
                        "first_visible_bonus": args.first_visible_bonus,
                        "first_shootable_bonus": args.first_shootable_bonus,
                        "visible_contact_progress_reward": args.visible_contact_progress_reward,
                        "terminate_on_first_visible": args.terminate_on_first_visible,
                        "terminate_on_first_shootable": args.terminate_on_first_shootable,
                    },
                    metrics=metrics,
                    rollout_summary=rollout_summary,
                    update_index=update_index,
                    buffer_path=buffer_path,
                    curriculum=curriculum,
                    curriculum_stage=curriculum_stage,
                )
            summary_record = {
                "update": update_index,
                "records": len(buffer),
                "buffer_path": str(buffer_path),
                "checkpoint_path": str(checkpoint_path),
                "metrics": metrics,
                "rollout_summary": rollout_summary,
                "curriculum_stage": curriculum_stage,
            }
            summaries.append(summary_record)
            score = float(rollout_summary.get("checkpoint_selection_score", 0.0))
            if best_checkpoint is None or score > float(best_checkpoint["score"]):
                best_checkpoint = {
                    "update": update_index,
                    "score": score,
                    "checkpoint_path": str(checkpoint_path),
                    "buffer_path": str(buffer_path),
                    "rollout_summary": rollout_summary,
                    "curriculum_stage": curriculum_stage,
                }
    finally:
        await env.close()
    resume_source = "new"
    if args.resume_best_checkpoint:
        resume_source = "memory_best"
    elif resume_checkpoint is not None:
        resume_source = "explicit"
    return {
        "schema": "restfuldoom.ppo_training_run.v1",
        "run_id": args.run_id,
        "updates": summaries,
        "behavior_clone": behavior_clone_summary or {},
        "reset_start": reset_start,
        "curriculum": curriculum,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_checkpoint_source": resume_source,
        "best_checkpoint": best_checkpoint or {},
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
    }


async def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Evaluates a PPO checkpoint against a baseline."""
    reset_start = _resolve_reset_start(args)
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
        run_id=f"{args.run_id}-eval",
        goal_preset=args.goal_preset,
        target_x_fp=args.target_x_fp,
        target_y_fp=args.target_y_fp,
        max_steps=args.eval_max_steps,
        level_complete_bonus=args.level_complete_bonus,
        kill_goal_bonus=args.kill_goal_bonus,
        required_kills=args.required_kills,
        memory_path=args.memory_path,
        reset_timeout_seconds=args.reset_timeout_seconds,
        reset_attempts=args.reset_attempts,
        reset_start_x_fp=reset_start.get("x_fp"),
        reset_start_y_fp=reset_start.get("y_fp"),
        reset_start_angle_degrees=int(reset_start.get("angle_degrees", 0)),
        reset_start_face_nearest_enemy=bool(reset_start.get("face_nearest_enemy", False)),
        reset_start_health=reset_start.get("health"),
        reset_start_armor=reset_start.get("armor"),
        reset_start_ammo_bullets=reset_start.get("ammo_bullets"),
        reset_warmup_steps=args.reset_warmup_steps,
        reset_warmup_max_tics=args.reset_warmup_max_tics,
        reset_warmup_until_visible=args.reset_warmup_until_visible,
        reset_warmup_until_shootable=args.reset_warmup_until_shootable,
        first_visible_bonus=args.first_visible_bonus,
        first_shootable_bonus=args.first_shootable_bonus,
        visible_contact_progress_reward=args.visible_contact_progress_reward,
        terminate_on_first_visible=args.terminate_on_first_visible,
        terminate_on_first_shootable=args.terminate_on_first_shootable,
    )
    candidate = await evaluate_checkpoint(
        str(args.eval_checkpoint),
        env_config,
        episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        seed=args.seed,
        device=args.device,
        deterministic=not args.eval_sample,
    )
    if args.eval_baseline == "random":
        baseline = await evaluate_random_policy(
            env_config,
            episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
            seed=args.seed,
        )
    else:
        baseline = await evaluate_heuristic_policy(
            env_config,
            episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
            seed=args.seed,
        )
    decision = decide_promotion(
        candidate=candidate,
        baseline=baseline,
        min_completion_delta=args.promotion_min_completion_delta,
        min_kill_delta=args.promotion_min_kill_delta,
        min_reward_delta=args.promotion_min_reward_delta,
        min_completion_rate=args.promotion_min_completion_rate,
        min_mean_kills=args.promotion_min_mean_kills,
    )
    _record_eval_history(args, candidate.to_dict(), baseline.to_dict(), decision)
    return {
        "schema": "restfuldoom.ppo_eval.v1",
        "checkpoint_path": str(args.eval_checkpoint),
        "candidate": candidate.to_dict(),
        "baseline": baseline.to_dict(),
        "promotion": {
            "promote": decision.promote,
            "reasons": decision.reasons,
        },
    }


def _env_config_for_start(
    args: argparse.Namespace,
    reset_start: object,
) -> DoomEnvConfig:
    start = reset_start if isinstance(reset_start, dict) else {}
    return DoomEnvConfig(
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
        reset_timeout_seconds=args.reset_timeout_seconds,
        reset_attempts=args.reset_attempts,
        reset_start_x_fp=start.get("x_fp"),
        reset_start_y_fp=start.get("y_fp"),
        reset_start_angle_degrees=int(start.get("angle_degrees", 0)),
        reset_start_face_nearest_enemy=bool(start.get("face_nearest_enemy", False)),
        reset_start_health=start.get("health"),
        reset_start_armor=start.get("armor"),
        reset_start_ammo_bullets=start.get("ammo_bullets"),
        reset_warmup_steps=args.reset_warmup_steps,
        reset_warmup_max_tics=args.reset_warmup_max_tics,
        reset_warmup_until_visible=args.reset_warmup_until_visible,
        reset_warmup_until_shootable=args.reset_warmup_until_shootable,
        first_visible_bonus=args.first_visible_bonus,
        first_shootable_bonus=args.first_shootable_bonus,
        visible_contact_progress_reward=args.visible_contact_progress_reward,
        terminate_on_first_visible=args.terminate_on_first_visible,
        terminate_on_first_shootable=args.terminate_on_first_shootable,
    )


def _annotate_buffer_curriculum(
    buffer: object,
    curriculum: dict[str, object],
    curriculum_stage: dict[str, object],
) -> None:
    for record in getattr(buffer, "records", []):
        if isinstance(getattr(record, "info", None), dict):
            record.info["curriculum"] = {
                "schema": curriculum.get("schema"),
                "name": curriculum.get("name"),
                "mode": curriculum.get("mode"),
            }
            record.info["curriculum_stage"] = dict(curriculum_stage)


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
    curriculum: dict[str, object] | None = None,
    curriculum_stage: dict[str, object] | None = None,
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
        "curriculum": curriculum or {},
        "curriculum_stage": curriculum_stage or {},
        "eval_history": [],
    }
    memory.data["ppo_policy"] = record
    score = float(rollout_summary.get("checkpoint_selection_score", 0.0))
    previous_best = memory.data.get("ppo_best_checkpoint")
    if not isinstance(previous_best, dict) or score > float(
        previous_best.get("checkpoint_selection_score", -1e12)
    ):
        memory.data["ppo_best_checkpoint"] = {
            "schema": "restfuldoom.ppo_best_checkpoint.v1",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_selection_score": score,
            "goal_preset": goal_preset,
            "update_index": update_index,
            "buffer_path": str(buffer_path),
            "rollout_summary": rollout_summary,
            "curriculum_stage": curriculum_stage or {},
            "updated_at": _iso_now(),
        }
    checkpoints = memory.data.setdefault("ppo_checkpoints", [])
    checkpoints.append(
        {
            "checkpoint_path": str(checkpoint_path),
            "update_index": update_index,
            "buffer_path": str(buffer_path),
            "rollout_summary": rollout_summary,
            "curriculum_stage": curriculum_stage or {},
        }
    )
    memory.data["updated_at"] = _iso_now()
    memory.save()


def _resolve_resume_checkpoint(
    args: argparse.Namespace,
    memory: AgentMemory | None,
) -> Path | None:
    """Returns the checkpoint path to resume from, validating memory-backed resumes."""
    if args.resume_checkpoint is not None and args.resume_best_checkpoint:
        raise ValueError("--resume-checkpoint cannot be combined with --resume-best-checkpoint")
    if args.resume_checkpoint is not None:
        path = Path(args.resume_checkpoint)
        if not path.exists():
            raise ValueError(f"--resume-checkpoint does not exist: {path}")
        return path
    if not args.resume_best_checkpoint:
        return None
    if memory is None:
        raise ValueError("--resume-best-checkpoint requires --memory-path")
    best = memory.data.get("ppo_best_checkpoint")
    if not isinstance(best, dict):
        raise ValueError("memory does not contain ppo_best_checkpoint")
    checkpoint = best.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("memory ppo_best_checkpoint does not include checkpoint_path")
    path = Path(checkpoint)
    if not path.exists():
        raise ValueError(f"memory ppo_best_checkpoint file does not exist: {path}")
    return path


def _load_behavior_clone_samples(
    args: argparse.Namespace,
) -> tuple[list[tuple[list[float], int]], dict[str, object]]:
    samples: list[tuple[list[float], int]] = []
    label_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    skipped = 0
    max_samples = max(1, int(args.bc_max_samples))
    for path in args.bc_trajectory:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(samples) >= max_samples:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                decision = record.get("metadata", {}).get("policy_decision", {})
                if not isinstance(decision, dict):
                    skipped += 1
                    continue
                skill = decision.get("skill")
                if not isinstance(skill, str):
                    skipped += 1
                    continue
                label_counts[skill] += 1
                action = map_expert_skill_to_ppo_action(skill)
                if action is None:
                    skipped += 1
                    continue
                samples.append((pad_observation_features(features_from_record(record)), action))
                mapped_counts[ACTION_SCHEMA["actions"][action]] += 1
        if len(samples) >= max_samples:
            break
    if not samples:
        raise ValueError("no usable behavior-cloning samples found")
    return samples, {
        "schema": "restfuldoom.ppo_behavior_clone.v1",
        "trajectory_paths": [str(path) for path in args.bc_trajectory],
        "samples": len(samples),
        "skipped": skipped,
        "expert_skill_counts": dict(sorted(label_counts.items())),
        "ppo_skill_counts": dict(sorted(mapped_counts.items())),
    }


def _resolve_reset_start(args: argparse.Namespace) -> dict[str, object]:
    start: dict[str, object] = {}
    if args.reset_start_trajectory is not None:
        start.update(
            _reset_start_from_trajectory(
                args.reset_start_trajectory,
                index=args.reset_start_index,
            )
        )

    if args.reset_start_x_fp is not None:
        start["x_fp"] = int(args.reset_start_x_fp)
    if args.reset_start_y_fp is not None:
        start["y_fp"] = int(args.reset_start_y_fp)
    if args.reset_start_angle_degrees is not None:
        start["angle_degrees"] = int(args.reset_start_angle_degrees) % 360
    if args.reset_start_face_nearest_enemy:
        start["face_nearest_enemy"] = True
    if args.reset_start_health is not None:
        start["health"] = int(args.reset_start_health)
    if args.reset_start_armor is not None:
        start["armor"] = int(args.reset_start_armor)
    if args.reset_start_ammo_bullets is not None:
        start["ammo_bullets"] = int(args.reset_start_ammo_bullets)
    return start


def _reset_start_from_trajectory(path: Path, *, index: int) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index != index:
                continue
            record = json.loads(line)
            state = record.get("state", {}) if isinstance(record.get("state"), dict) else {}
            position = state.get("position_fp")
            if not isinstance(position, list) or len(position) < 2:
                raise ValueError(
                    f"trajectory row {index} in {path} does not include state.position_fp"
                )
            return {
                "x_fp": int(position[0]),
                "y_fp": int(position[1]),
                "health": int(state.get("health", 100)),
                "armor": int(state.get("armor", 0)),
                "ammo_bullets": int(state.get("ammo_bullets", 50)),
            }
    raise ValueError(f"trajectory row {index} not found in {path}")


def _record_eval_history(
    args: argparse.Namespace,
    candidate: dict[str, object],
    baseline: dict[str, object],
    decision: object,
) -> None:
    if args.memory_path is None:
        return
    memory = AgentMemory.load(args.memory_path)
    record = {
        "schema": "restfuldoom.ppo_eval.v1",
        "checkpoint_path": str(args.eval_checkpoint),
        "baseline": args.eval_baseline,
        "candidate": candidate,
        "baseline_result": baseline,
        "promotion": {
            "promote": bool(getattr(decision, "promote", False)),
            "reasons": list(getattr(decision, "reasons", [])),
        },
        "evaluated_at": _iso_now(),
    }
    history = memory.data.setdefault("ppo_eval_history", [])
    history.append(record)
    policy = memory.data.get("ppo_policy")
    if isinstance(policy, dict) and policy.get("checkpoint_path") == str(args.eval_checkpoint):
        policy["eval_history"] = history[-10:]
    memory.data["updated_at"] = _iso_now()
    memory.save()


def _summarize_buffer(buffer: object) -> dict[str, object]:
    records = getattr(buffer, "records", [])
    skills = Counter(
        record.info.get("skill", "unknown")
        for record in records
        if isinstance(record.info, dict)
    )
    curriculum_stages = Counter(
        str(record.info.get("curriculum_stage", {}).get("name", "unknown"))
        for record in records
        if isinstance(record.info, dict) and record.info.get("curriculum_stage")
    )
    transitions = [
        record.info.get("transition", {})
        for record in records
        if isinstance(record.info, dict)
    ]
    route_outcomes = [
        record.info.get("route_outcome", {})
        for record in records
        if isinstance(record.info, dict) and isinstance(record.info.get("route_outcome", {}), dict)
    ]
    contact_contexts = [
        context
        for record in records
        if (context := _learning_trace_contact_context(record))
    ]
    states = [
        record.info.get("state", {})
        for record in records
        if isinstance(record.info, dict)
    ]
    action_mask_sizes = [
        sum(1 for allowed in getattr(record, "action_mask", []) if allowed)
        for record in records
    ]
    invalid_action_steps = sum(
        1
        for record in records
        if getattr(record, "action_mask", [])
        and (
            record.action >= len(record.action_mask)
            or not bool(record.action_mask[record.action])
        )
    )
    warmups = [
        record.info.get("reset_warmup", {})
        for record in records
        if isinstance(record.info, dict) and record.info.get("reset_warmup")
    ]
    unique_warmups: dict[object, dict[str, object]] = {}
    for index, warmup in enumerate(warmups):
        key = warmup.get("episode_index", index)
        unique_warmups.setdefault(key, warmup)
    warmups = list(unique_warmups.values())
    warmup_reasons = Counter(
        str(warmup.get("stop_reason", "unknown"))
        for warmup in warmups
        if warmup.get("enabled")
    )
    summary = {
        "records": len(records),
        "total_reward": round(sum(float(record.reward) for record in records), 4),
        "action_reward": round(
            sum(float(record.info.get("action_reward", 0.0)) for record in records),
            4,
        ),
        "route_action_reward": round(
            sum(
                float(record.info.get("route_action_reward", 0.0))
                for record in records
                if isinstance(record.info, dict)
            ),
            4,
        ),
        "shootable_target_steps": sum(
            1 for record in records if record.info.get("had_shootable_target")
        ),
        "fire_on_shootable_steps": sum(
            1
            for record in records
            if record.info.get("had_shootable_target")
            and record.info.get("skill") == "fire"
        ),
        "missed_shootable_fire_steps": sum(
            1
            for record in records
            if record.info.get("had_shootable_target")
            and record.info.get("skill") != "fire"
        ),
        "visible_enemy_steps": sum(
            1 for record in records if record.info.get("had_visible_enemy")
        ),
        "first_visible_contacts": sum(
            1 for record in records if record.info.get("first_visible_contact")
        ),
        "first_shootable_contacts": sum(
            1 for record in records if record.info.get("first_shootable_contact")
        ),
        "contact_reward": round(
            sum(
                float(record.info.get("contact_reward", 0.0))
                for record in records
                if isinstance(record.info, dict)
            ),
            4,
        ),
        "visible_contact_distance_delta": round(
            sum(
                float(record.info.get("visible_contact_distance_delta", 0.0))
                for record in records
                if isinstance(record.info, dict)
            ),
            4,
        ),
        "visible_contact_progress_reward": round(
            sum(
                float(record.info.get("visible_contact_progress_reward", 0.0))
                for record in records
                if isinstance(record.info, dict)
            ),
            4,
        ),
        "route_attempt_steps": sum(1 for outcome in route_outcomes if outcome.get("attempted")),
        "route_reached_steps": sum(1 for outcome in route_outcomes if outcome.get("reached")),
        "route_failed_steps": sum(1 for outcome in route_outcomes if outcome.get("failed")),
        "route_progress_units": round(
            sum(float(outcome.get("progress_units", 0.0)) for outcome in route_outcomes),
            4,
        ),
        "contact_context_active_steps": sum(
            1 for context in contact_contexts if context.get("recent_contact_active")
        ),
        "contact_use_line_active_steps": sum(
            1 for context in contact_contexts if context.get("contact_use_line_active")
        ),
        "contact_use_line_close_steps": sum(
            1 for context in contact_contexts if context.get("contact_use_line_close")
        ),
        "contact_use_line_followthrough_steps": sum(
            1
            for context in contact_contexts
            if context.get("contact_use_line_followthrough_active")
        ),
        "mean_contact_use_line_distance_norm": _mean_contact_context_value(
            contact_contexts,
            "contact_use_line_distance_norm",
            active_key="contact_use_line_active",
        ),
        "mean_contact_use_line_age_norm": _mean_contact_context_value(
            contact_contexts,
            "contact_use_line_age_norm",
            active_key="contact_use_line_active",
        ),
        "mean_action_mask_size": round(
            sum(action_mask_sizes) / max(1, len(action_mask_sizes)),
            4,
        ),
        "invalid_action_steps": invalid_action_steps,
        "reset_warmup_tics": sum(int(warmup.get("tics", 0)) for warmup in warmups),
        "reset_warmup_steps": sum(int(warmup.get("steps", 0)) for warmup in warmups),
        "reset_warmup_stop_reasons": dict(sorted(warmup_reasons.items())),
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
        "curriculum_stage_counts": dict(sorted(curriculum_stages.items())),
    }
    summary["checkpoint_selection_score"] = _checkpoint_selection_score(summary)
    return summary


def _learning_trace_contact_context(record: object) -> dict[str, object]:
    """Returns the compact contact feature group from a rollout record."""
    info = getattr(record, "info", {})
    if not isinstance(info, dict):
        return {}
    trace = info.get("learning_trace", {})
    if not isinstance(trace, dict):
        return {}
    observation = trace.get("observation", {})
    if not isinstance(observation, dict):
        return {}
    groups = observation.get("groups", {})
    if not isinstance(groups, dict):
        return {}
    contact = groups.get("contact", {})
    return contact if isinstance(contact, dict) else {}


def _mean_contact_context_value(
    contexts: list[dict[str, object]],
    key: str,
    *,
    active_key: str,
) -> float:
    values = [
        float(context.get(key, 0.0))
        for context in contexts
        if context.get(active_key)
    ]
    return round(sum(values) / max(1, len(values)), 4)


def _checkpoint_selection_score(summary: dict[str, object]) -> float:
    """Scores a rollout for resume selection without changing promotion rules."""
    total_reward = float(summary.get("total_reward", 0.0))
    max_kills = int(summary.get("max_kills", 0))
    damage_delta = int(summary.get("damage_delta", 0))
    first_shootable_contacts = int(summary.get("first_shootable_contacts", 0))
    fire_on_shootable = int(summary.get("fire_on_shootable_steps", 0))
    missed_shootable_fire = int(summary.get("missed_shootable_fire_steps", 0))
    route_failed = int(summary.get("route_failed_steps", 0))
    return round(
        total_reward
        + max_kills * 75.0
        + damage_delta * 2.0
        + first_shootable_contacts * 25.0
        + fire_on_shootable * 0.5
        - missed_shootable_fire
        - route_failed * 0.1,
        4,
    )


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
    parser.add_argument("--reset-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--reset-attempts", type=int, default=2)
    parser.add_argument("--reset-start-x-fp", type=int)
    parser.add_argument("--reset-start-y-fp", type=int)
    parser.add_argument("--reset-start-angle-degrees", type=int)
    parser.add_argument("--reset-start-face-nearest-enemy", action="store_true")
    parser.add_argument("--reset-start-health", type=int)
    parser.add_argument("--reset-start-armor", type=int)
    parser.add_argument("--reset-start-ammo-bullets", type=int)
    parser.add_argument("--reset-start-trajectory", type=Path)
    parser.add_argument("--reset-start-index", type=int, default=0)
    parser.add_argument(
        "--curriculum",
        choices=["none", *curriculum_names()],
        default="none",
        help="Named reset-start curriculum for PPO training.",
    )
    parser.add_argument(
        "--curriculum-mode",
        choices=["fixed", "round_robin", "progressive", "random"],
        default="round_robin",
        help="How training updates select curriculum stages.",
    )
    parser.add_argument("--curriculum-start-index", type=int, default=0)
    parser.add_argument("--reset-warmup-steps", type=int, default=0)
    parser.add_argument("--reset-warmup-max-tics", type=int, default=0)
    parser.add_argument("--reset-warmup-until-visible", action="store_true")
    parser.add_argument("--reset-warmup-until-shootable", action="store_true")
    parser.add_argument("--first-visible-bonus", type=float, default=0.0)
    parser.add_argument("--first-shootable-bonus", type=float, default=0.0)
    parser.add_argument("--visible-contact-progress-reward", type=float, default=0.0)
    parser.add_argument("--terminate-on-first-visible", action="store_true")
    parser.add_argument("--terminate-on-first-shootable", action="store_true")
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
    parser.add_argument("--bc-trajectory", type=Path, action="append", default=[])
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-learning-rate", type=float)
    parser.add_argument("--bc-max-samples", type=int, default=20000)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--resume-best-checkpoint",
        action="store_true",
        help="Resume from ppo_best_checkpoint in --memory-path.",
    )
    parser.add_argument("--eval-checkpoint", type=Path)
    parser.add_argument("--eval-baseline", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=256)
    parser.add_argument("--eval-sample", action="store_true")
    parser.add_argument("--promotion-min-completion-delta", type=float, default=0.0)
    parser.add_argument("--promotion-min-kill-delta", type=float, default=0.0)
    parser.add_argument("--promotion-min-reward-delta", type=float, default=0.0)
    parser.add_argument("--promotion-min-completion-rate", type=float, default=1.0)
    parser.add_argument("--promotion-min-mean-kills", type=float, default=1.0)
    args = parser.parse_args()
    if args.eval_checkpoint:
        print(json.dumps(asyncio.run(evaluate(args)), sort_keys=True))
    else:
        print(json.dumps(asyncio.run(train(args)), sort_keys=True))


if __name__ == "__main__":
    main()
