"""CLI for PPO training over high-level Doom skills."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shlex
import subprocess
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
from .snapshot_curriculum import load_snapshot_curriculum
from .skill_policy import features_from_record


async def train(args: argparse.Namespace) -> dict[str, object]:
    """Runs PPO collection and update batches."""
    require_torch()
    reset_start = _resolve_reset_start(args)
    curriculum = _build_training_curriculum(args, reset_start)
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
            if _rollout_stage_mixing_enabled(args, curriculum):
                curriculum_stage = _mixed_curriculum_stage_descriptor(args, curriculum)
                buffer = await _collect_mixed_curriculum_rollout(
                    trainer,
                    env,
                    args,
                    curriculum=curriculum,
                    update_index=update_index,
                )
            else:
                curriculum_stage = stage_for_update(curriculum, update_index)
                buffer = await _collect_curriculum_stage_rollout(
                    trainer,
                    env,
                    args,
                    curriculum=curriculum,
                    curriculum_stage=curriculum_stage,
                    update_index=update_index,
                )
            buffer_path = args.buffer_dir / f"{args.run_id}-buffer-{update_index:04d}.jsonl"
            buffer.save_jsonl(buffer_path)
            rollout_summary = _summarize_buffer(buffer)
            metrics = trainer.update(buffer)
            checkpoint_path = args.checkpoint_dir / f"{args.run_id}-ppo-{update_index:04d}.pt"
            checkpoint_extra = {
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
            }
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
                extra=checkpoint_extra,
            )
            checkpoint_eval = None
            if args.checkpoint_eval_curriculum:
                checkpoint_eval = await _evaluate_checkpoint_curriculum(
                    checkpoint_path,
                    args,
                    curriculum=curriculum,
                    update_index=update_index,
                )
                checkpoint_extra["checkpoint_curriculum_eval"] = checkpoint_eval
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
                    extra=checkpoint_extra,
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
                    checkpoint_eval=checkpoint_eval,
                )
            summary_record = {
                "update": update_index,
                "records": len(buffer),
                "buffer_path": str(buffer_path),
                "checkpoint_path": str(checkpoint_path),
                "metrics": metrics,
                "rollout_summary": rollout_summary,
                "curriculum_stage": curriculum_stage,
                "checkpoint_eval": checkpoint_eval or {},
            }
            summaries.append(summary_record)
            score = _checkpoint_resume_score(rollout_summary, checkpoint_eval)
            if best_checkpoint is None or score > float(best_checkpoint["score"]):
                best_checkpoint = {
                    "update": update_index,
                    "score": score,
                    "score_source": _checkpoint_resume_score_source(checkpoint_eval),
                    "checkpoint_path": str(checkpoint_path),
                    "buffer_path": str(buffer_path),
                    "rollout_summary": rollout_summary,
                    "curriculum_stage": curriculum_stage,
                    "checkpoint_eval": checkpoint_eval or {},
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
        snapshot_verify_restored_state=args.snapshot_verify_restored_state,
        snapshot_verify_tick_tolerance=args.snapshot_verify_tick_tolerance,
        snapshot_verify_stream_tick=args.snapshot_verify_stream_tick,
        snapshot_verify_position_tolerance_fp=args.snapshot_verify_position_tolerance_fp,
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


def _build_training_curriculum(
    args: argparse.Namespace,
    reset_start: dict[str, object],
) -> dict[str, object]:
    if args.snapshot_curriculum is not None:
        if args.curriculum != "none":
            raise ValueError("--snapshot-curriculum cannot be combined with --curriculum")
        if reset_start:
            raise ValueError("--snapshot-curriculum cannot be combined with --reset-start-*")
        return load_snapshot_curriculum(
            args.snapshot_curriculum,
            mode=args.curriculum_mode,
            start_index=args.curriculum_start_index,
            seed=args.seed,
        )
    return build_curriculum(
        name=args.curriculum,
        manual_reset_start=reset_start,
        mode=args.curriculum_mode,
        start_index=args.curriculum_start_index,
        seed=args.seed,
    )


def _env_config_for_stage(
    args: argparse.Namespace,
    curriculum: dict[str, object],
    curriculum_stage: dict[str, object],
    *,
    run_id: str,
    max_steps: int | None = None,
) -> DoomEnvConfig:
    """Builds environment config with auditable curriculum metadata."""
    return replace(
        _env_config_for_start(args, curriculum_stage.get("reset_start", {})),
        run_id=run_id,
        max_steps=args.max_steps if max_steps is None else int(max_steps),
        curriculum=_curriculum_metadata(curriculum),
        curriculum_stage=dict(curriculum_stage),
        reset_mode="snapshot" if _stage_uses_snapshot(curriculum_stage) else "episode",
        snapshot=dict(curriculum_stage.get("snapshot", {}))
        if isinstance(curriculum_stage.get("snapshot"), dict)
        else None,
    )


async def _collect_curriculum_stage_rollout(
    trainer: PPOTrainer,
    env: DoomAgentEnv,
    args: argparse.Namespace,
    *,
    curriculum: dict[str, object],
    curriculum_stage: dict[str, object],
    update_index: int,
) -> object:
    """Collects a rollout for one curriculum stage, restoring snapshots per reset."""
    if _stage_uses_snapshot(curriculum_stage):
        active_stage: dict[str, object] = dict(curriculum_stage)

        def before_reset(reset_index: int) -> None:
            nonlocal active_stage
            active_stage = _prepare_stage_for_reset(
                curriculum_stage,
                args,
                update_index=update_index,
                reset_index=reset_index,
            )
            env.config = _env_config_for_stage(
                args,
                curriculum,
                active_stage,
                run_id=f"{args.run_id}-{active_stage['name']}-snapshot{reset_index:03d}",
            )

        return await trainer.collect_rollout(
            env,
            steps=args.rollout_steps,
            seed=args.seed + update_index,
            before_reset=before_reset,
        )

    env.config = _env_config_for_stage(
        args,
        curriculum,
        curriculum_stage,
        run_id=f"{args.run_id}-{curriculum_stage['name']}",
    )
    buffer = await trainer.collect_rollout(
        env,
        steps=args.rollout_steps,
        seed=args.seed + update_index,
    )
    _annotate_buffer_curriculum(buffer, curriculum, curriculum_stage)
    return buffer


async def _collect_mixed_curriculum_rollout(
    trainer: PPOTrainer,
    env: DoomAgentEnv,
    args: argparse.Namespace,
    *,
    curriculum: dict[str, object],
    update_index: int,
) -> object:
    """Collects one PPO buffer while rotating curriculum stages between resets."""
    segment_tics = _effective_rollout_stage_segment_tics(args)

    def before_reset(reset_index: int) -> None:
        stage = _mixed_curriculum_stage_for_reset(
            curriculum,
            update_index=update_index,
            reset_index=reset_index,
            mode=args.rollout_stage_mix,
        )
        stage = _prepare_stage_for_reset(
            stage,
            args,
            update_index=update_index,
            reset_index=reset_index,
        )
        env.config = _env_config_for_stage(
            args,
            curriculum,
            stage,
            run_id=f"{args.run_id}-{stage['name']}-mix{reset_index:03d}",
            max_steps=segment_tics,
        )

    return await trainer.collect_rollout(
        env,
        steps=args.rollout_steps,
        seed=args.seed + update_index,
        before_reset=before_reset,
    )


def _rollout_stage_mixing_enabled(
    args: argparse.Namespace,
    curriculum: dict[str, object],
) -> bool:
    if getattr(args, "rollout_stage_mix", "off") == "off":
        return False
    stages = curriculum.get("stages", [])
    return isinstance(stages, list) and len(stages) > 1


def _mixed_curriculum_stage_for_reset(
    curriculum: dict[str, object],
    *,
    update_index: int,
    reset_index: int,
    mode: str,
) -> dict[str, object]:
    """Selects the curriculum stage used by one reset inside a mixed rollout."""
    stages = curriculum.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum has no stages")
    start_index = int(curriculum.get("start_index", 0))
    if mode == "round_robin":
        index = (start_index + int(update_index) + int(reset_index)) % len(stages)
    elif mode == "random":
        rng = random.Random(
            int(curriculum.get("seed", 0)) + int(update_index) * 1009 + int(reset_index)
        )
        index = rng.randrange(len(stages))
    else:
        raise ValueError(f"unsupported rollout stage mix mode {mode!r}")
    stage = dict(stages[index])
    stage["selected_index"] = index
    stage["mixed_rollout_reset_index"] = int(reset_index)
    return stage


def _prepare_stage_for_reset(
    curriculum_stage: dict[str, object],
    args: argparse.Namespace,
    *,
    update_index: int,
    reset_index: int,
) -> dict[str, object]:
    stage = dict(curriculum_stage)
    if _stage_uses_snapshot(stage):
        stage["snapshot_restore"] = _restore_snapshot_for_stage(
            stage,
            args,
            update_index=update_index,
            reset_index=reset_index,
        )
    return stage


def _stage_uses_snapshot(curriculum_stage: dict[str, object]) -> bool:
    return (
        curriculum_stage.get("reset_mode") == "snapshot"
        or bool(curriculum_stage.get("requires_progressed_state"))
        or isinstance(curriculum_stage.get("snapshot"), dict)
    )


def _restore_snapshot_for_stage(
    curriculum_stage: dict[str, object],
    args: argparse.Namespace,
    *,
    update_index: int,
    reset_index: int,
) -> dict[str, object]:
    command_template = getattr(args, "snapshot_restore_command", None)
    if not command_template:
        slot = _snapshot_slot(curriculum_stage)
        if slot is None:
            raise ValueError(
                "snapshot curriculum stage "
                f"{curriculum_stage.get('name')!r} requires --snapshot-restore-command "
                "or snapshot.slot"
            )
        snapshot = curriculum_stage.get("snapshot", {})
        snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
        return {
            "schema": "restfuldoom.snapshot_restore.v1",
            "stage_name": curriculum_stage.get("name"),
            "stage_index": curriculum_stage.get("selected_index", curriculum_stage.get("index")),
            "snapshot_id": snapshot_dict.get("id"),
            "snapshot_path": snapshot_dict.get("path"),
            "snapshot_ref": snapshot_dict.get("ref"),
            "api_method": "grpc_load_snapshot",
            "restore_command_configured": False,
            "returncode": 0,
            "slot": slot,
            "update_index": int(update_index),
            "reset_index": int(reset_index),
        }
    command = _render_snapshot_restore_command(
        str(command_template),
        curriculum_stage,
        update_index=update_index,
        reset_index=reset_index,
    )
    argv = shlex.split(command)
    if not argv:
        raise ValueError("--snapshot-restore-command rendered an empty command")
    start = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=args.snapshot_restore_cwd,
        timeout=args.snapshot_restore_timeout_seconds,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.monotonic() - start, 4)
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot restore command failed "
            f"for stage {curriculum_stage.get('name')!r} "
            f"with exit {completed.returncode}: {_restore_output_tail(completed.stderr)}"
        )
    snapshot = curriculum_stage.get("snapshot", {})
    snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
    return {
        "schema": "restfuldoom.snapshot_restore.v1",
        "stage_name": curriculum_stage.get("name"),
        "stage_index": curriculum_stage.get("selected_index", curriculum_stage.get("index")),
        "snapshot_id": snapshot_dict.get("id"),
        "snapshot_path": snapshot_dict.get("path"),
        "snapshot_ref": snapshot_dict.get("ref"),
        "restore_command_configured": True,
        "restore_command_argv": _redacted_restore_argv(argv),
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "update_index": int(update_index),
        "reset_index": int(reset_index),
    }


def _snapshot_slot(curriculum_stage: dict[str, object]) -> int | None:
    snapshot = curriculum_stage.get("snapshot", {})
    snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
    value = snapshot_dict.get("slot")
    if value is None:
        ref = snapshot_dict.get("ref")
        if isinstance(ref, str) and ref.startswith("save_slot:"):
            value = ref.removeprefix("save_slot:")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _render_snapshot_restore_command(
    command_template: str,
    curriculum_stage: dict[str, object],
    *,
    update_index: int,
    reset_index: int,
) -> str:
    snapshot = curriculum_stage.get("snapshot", {})
    snapshot_dict = snapshot if isinstance(snapshot, dict) else {}
    expected = curriculum_stage.get("expected_state", {})
    expected_dict = expected if isinstance(expected, dict) else {}
    values = {
        "stage_name": curriculum_stage.get("name", ""),
        "stage_index": curriculum_stage.get("selected_index", curriculum_stage.get("index", "")),
        "update_index": int(update_index),
        "reset_index": int(reset_index),
        "snapshot_id": snapshot_dict.get("id", ""),
        "snapshot_path": snapshot_dict.get("path", ""),
        "snapshot_ref": snapshot_dict.get("ref", ""),
        "microvm_id": snapshot_dict.get("microvm_id", ""),
        "capsule": snapshot_dict.get("capsule", ""),
        "expected_tick": expected_dict.get("tick", ""),
        "expected_level_time": expected_dict.get("level_time", ""),
        "expected_x_fp": expected_dict.get("x_fp", ""),
        "expected_y_fp": expected_dict.get("y_fp", ""),
    }
    values.update({f"{key}_sh": shlex.quote(str(value)) for key, value in values.items()})
    return command_template.format_map(_DefaultFormatMap(values))


class _DefaultFormatMap(dict):
    """String formatter that leaves unknown restore placeholders empty."""

    def __missing__(self, key: str) -> str:
        return ""


def _restore_output_tail(output: str, *, limit: int = 400) -> str:
    text = (output or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _redacted_restore_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    secret_flags = {"--token", "--auth", "--password", "--secret", "--api-key"}
    for arg in argv:
        lower = arg.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if lower in secret_flags:
            redacted.append(arg)
            redact_next = True
        elif any(marker in lower for marker in ("token=", "secret=", "password=", "api_key=")):
            key = arg.split("=", 1)[0]
            redacted.append(f"{key}=<redacted>")
        else:
            redacted.append(arg)
    return redacted


def _mixed_curriculum_stage_descriptor(
    args: argparse.Namespace,
    curriculum: dict[str, object],
) -> dict[str, object]:
    """Describes an update whose records contain multiple curriculum stages."""
    stages = curriculum.get("stages", [])
    stage_names = [
        str(stage.get("name", f"stage_{index}"))
        for index, stage in enumerate(stages)
        if isinstance(stage, dict)
    ]
    return {
        "schema": "restfuldoom.ppo_mixed_curriculum_stage.v1",
        "name": "mixed_curriculum",
        "mode": args.rollout_stage_mix,
        "segment_tics": int(args.rollout_stage_segment_tics),
        "effective_segment_tics": _effective_rollout_stage_segment_tics(args),
        "stage_count": len(stage_names),
        "stages": stage_names,
    }


def _effective_rollout_stage_segment_tics(args: argparse.Namespace) -> int:
    max_steps = max(1, int(args.max_steps))
    requested = int(args.rollout_stage_segment_tics)
    if requested > 0:
        return min(max_steps, requested)
    return max_steps


def _curriculum_metadata(curriculum: dict[str, object]) -> dict[str, object]:
    return {
        "schema": curriculum.get("schema"),
        "name": curriculum.get("name"),
        "mode": curriculum.get("mode"),
        "start_index": curriculum.get("start_index"),
    }


def _annotate_buffer_curriculum(
    buffer: object,
    curriculum: dict[str, object],
    curriculum_stage: dict[str, object],
) -> None:
    for record in getattr(buffer, "records", []):
        if isinstance(getattr(record, "info", None), dict):
            record.info["curriculum"] = _curriculum_metadata(curriculum)
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
    checkpoint_eval: dict[str, object] | None = None,
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
        "checkpoint_eval": checkpoint_eval or {},
        "eval_history": [],
    }
    memory.data["ppo_policy"] = record
    score = _checkpoint_resume_score(rollout_summary, checkpoint_eval)
    score_source = _checkpoint_resume_score_source(checkpoint_eval)
    previous_best = memory.data.get("ppo_best_checkpoint")
    if _should_replace_best_checkpoint(previous_best, score, score_source):
        memory.data["ppo_best_checkpoint"] = {
            "schema": "restfuldoom.ppo_best_checkpoint.v1",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_selection_score": score,
            "checkpoint_selection_source": score_source,
            "goal_preset": goal_preset,
            "update_index": update_index,
            "buffer_path": str(buffer_path),
            "rollout_summary": rollout_summary,
            "curriculum": curriculum or {},
            "curriculum_stage": curriculum_stage or {},
            "checkpoint_eval": checkpoint_eval or {},
            "updated_at": _iso_now(),
        }
    checkpoints = memory.data.setdefault("ppo_checkpoints", [])
    checkpoints.append(
        {
            "checkpoint_path": str(checkpoint_path),
            "update_index": update_index,
            "buffer_path": str(buffer_path),
            "rollout_summary": rollout_summary,
            "curriculum": curriculum or {},
            "curriculum_stage": curriculum_stage or {},
            "checkpoint_eval": checkpoint_eval or {},
        }
    )
    memory.data["updated_at"] = _iso_now()
    memory.save()


async def _evaluate_checkpoint_curriculum(
    checkpoint_path: Path,
    args: argparse.Namespace,
    *,
    curriculum: dict[str, object],
    update_index: int,
) -> dict[str, object]:
    """Evaluates one checkpoint across every reset stage in the active curriculum."""
    stages = curriculum.get("stages", [])
    if not isinstance(stages, list) or not stages:
        stages = [
            {
                "index": 0,
                "name": "fresh_spawn",
                "reset_start": {},
            }
        ]

    stage_records: list[dict[str, object]] = []
    for stage_index, stage in enumerate(stages):
        stage_dict = dict(stage) if isinstance(stage, dict) else {}
        stage_name = str(stage_dict.get("name", f"stage_{stage_index}"))
        env_config = _env_config_for_stage(
            args,
            curriculum,
            stage_dict,
            run_id=f"{args.run_id}-checkpoint-eval-{update_index:04d}-{stage_name}",
            max_steps=args.checkpoint_eval_max_steps,
        )

        def before_reset(env: DoomAgentEnv, episode_index: int) -> None:
            active_stage = _prepare_stage_for_reset(
                stage_dict,
                args,
                update_index=update_index,
                reset_index=episode_index,
            )
            env.config = _env_config_for_stage(
                args,
                curriculum,
                active_stage,
                run_id=(
                    f"{args.run_id}-checkpoint-eval-{update_index:04d}-"
                    f"{stage_name}-episode{episode_index:03d}"
                ),
                max_steps=args.checkpoint_eval_max_steps,
            )

        result = await evaluate_checkpoint(
            str(checkpoint_path),
            env_config,
            episodes=args.checkpoint_eval_episodes,
            max_steps=args.checkpoint_eval_max_steps,
            seed=args.seed + update_index * 1000 + stage_index * 100,
            device=args.device,
            deterministic=not args.checkpoint_eval_sample,
            before_reset=before_reset,
        )
        stage_score = _policy_eval_selection_score(result)
        stage_records.append(
            {
                "stage": stage_dict,
                "selection_score": stage_score,
                "result": result.to_dict(),
            }
        )

    stage_scores = [float(record["selection_score"]) for record in stage_records]
    mean_score = sum(stage_scores) / max(1, len(stage_scores))
    worst_score = min(stage_scores) if stage_scores else 0.0
    aggregate_score = round(mean_score * 0.7 + worst_score * 0.3, 4)
    return {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "checkpoint_path": str(checkpoint_path),
        "curriculum": {
            "schema": curriculum.get("schema"),
            "name": curriculum.get("name"),
            "mode": curriculum.get("mode"),
            "start_index": curriculum.get("start_index"),
        },
        "episodes_per_stage": int(args.checkpoint_eval_episodes),
        "max_steps": int(args.checkpoint_eval_max_steps),
        "sample": bool(args.checkpoint_eval_sample),
        "stage_count": len(stage_records),
        "mean_stage_score": round(mean_score, 4),
        "worst_stage_score": round(worst_score, 4),
        "selection_score": aggregate_score,
        "score_formula": "0.7 * mean_stage_score + 0.3 * worst_stage_score",
        "stages": stage_records,
    }


def _policy_eval_selection_score(eval_result: object) -> float:
    """Scores deterministic eval results for resume selection, not promotion."""
    result = getattr(eval_result, "result", eval_result)
    return round(
        float(getattr(result, "mean_reward", 0.0))
        + float(getattr(result, "mean_kills", 0.0)) * 100.0
        + float(getattr(result, "level_completion_rate", 0.0)) * 500.0
        + float(getattr(result, "survival_rate", 0.0)) * 20.0
        - float(getattr(result, "mean_stuck_events", 0.0)) * 2.0,
        4,
    )


def _checkpoint_resume_score(
    rollout_summary: dict[str, object],
    checkpoint_eval: dict[str, object] | None = None,
) -> float:
    """Returns the score used for best-checkpoint resume selection."""
    if isinstance(checkpoint_eval, dict) and checkpoint_eval.get("selection_score") is not None:
        return float(checkpoint_eval["selection_score"])
    return float(rollout_summary.get("checkpoint_selection_score", 0.0))


def _checkpoint_resume_score_source(checkpoint_eval: dict[str, object] | None = None) -> str:
    """Returns the source of the best-checkpoint resume score."""
    if isinstance(checkpoint_eval, dict) and checkpoint_eval.get("selection_score") is not None:
        return "checkpoint_curriculum_eval"
    return "rollout_summary"


def _should_replace_best_checkpoint(
    previous_best: object,
    score: float,
    score_source: str,
) -> bool:
    """Returns whether a checkpoint should become the memory resume candidate."""
    if not isinstance(previous_best, dict):
        return True
    previous_source = _best_checkpoint_score_source(previous_best)
    if previous_source != score_source:
        return score_source == "checkpoint_curriculum_eval"
    return score > float(previous_best.get("checkpoint_selection_score", -1e12))


def _best_checkpoint_score_source(best: dict[str, object]) -> str:
    """Returns the score source for a stored best checkpoint, including legacy rows."""
    explicit = best.get("checkpoint_selection_source")
    if isinstance(explicit, str) and explicit:
        return explicit
    checkpoint_eval = best.get("checkpoint_eval", {})
    if isinstance(checkpoint_eval, dict) and checkpoint_eval.get("selection_score") is not None:
        return "checkpoint_curriculum_eval"
    return "rollout_summary"


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
        if (context := _learning_trace_group(record, "contact"))
    ]
    topology_contexts = [
        context
        for record in records
        if (context := _learning_trace_group(record, "topology"))
    ]
    visible_contact_contexts = [
        context
        for record in records
        if (context := _learning_trace_group(record, "visible_contact"))
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
    reset_contexts = [
        record.info.get("reset_context", {})
        for record in records
        if isinstance(record.info, dict) and record.info.get("reset_context")
    ]
    unique_reset_contexts: dict[object, dict[str, object]] = {}
    for index, context in enumerate(reset_contexts):
        key = context.get("episode_index", index)
        unique_reset_contexts.setdefault(key, context)
    reset_contexts = list(unique_reset_contexts.values())
    reset_sources = Counter(str(context.get("source", "unknown")) for context in reset_contexts)
    snapshot_contexts = [
        context for context in reset_contexts if context.get("source") == "snapshot_restore"
    ]
    snapshot_stage_counts = Counter(
        str(record.info.get("curriculum_stage", {}).get("name", "unknown"))
        for record in records
        if isinstance(record.info, dict)
        and isinstance(record.info.get("reset_context", {}), dict)
        and record.info.get("reset_context", {}).get("source") == "snapshot_restore"
        and record.info.get("curriculum_stage")
    )
    snapshot_restore_durations = [
        float(context.get("restore", {}).get("elapsed_seconds", 0.0))
        for context in snapshot_contexts
        if isinstance(context.get("restore", {}), dict)
    ]
    snapshot_verifications = [
        context.get("restored_state_verification", {})
        for context in snapshot_contexts
        if isinstance(context.get("restored_state_verification", {}), dict)
    ]
    kill_delta = sum(int(transition.get("kill_delta", 0)) for transition in transitions)
    damage_delta = sum(int(transition.get("damage_delta", 0)) for transition in transitions)
    kill_gains = [_earned_kill_gain(record) for record in records]
    snapshot_kill_gains = [
        _earned_kill_gain(record)
        for record in records
        if _record_reset_source(record) == "snapshot_restore"
    ]
    snapshot_transitions = [
        record.info.get("transition", {})
        for record in records
        if isinstance(record.info, dict)
        and _record_reset_source(record) == "snapshot_restore"
        and isinstance(record.info.get("transition", {}), dict)
    ]
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
        "topology_frontier_active_steps": sum(
            1 for context in topology_contexts if context.get("topology_frontier_active")
        ),
        "mean_topology_current_cell_visits_norm": _mean_context_value(
            topology_contexts,
            "topology_current_cell_visits_norm",
        ),
        "mean_topology_open_cell_min_visit_norm": _mean_context_value(
            topology_contexts,
            "topology_open_cell_min_visit_norm",
        ),
        "mean_topology_exhausted_open_ratio": _mean_context_value(
            topology_contexts,
            "topology_exhausted_open_ratio",
        ),
        "visible_contact_active_steps": sum(
            1
            for context in visible_contact_contexts
            if context.get("visible_contact_active")
        ),
        "visible_contact_needs_closure_steps": sum(
            1
            for context in visible_contact_contexts
            if context.get("visible_contact_needs_closure")
        ),
        "visible_contact_shootable_steps": sum(
            1
            for context in visible_contact_contexts
            if context.get("visible_contact_shootable")
        ),
        "visible_contact_aligned_steps": sum(
            1
            for context in visible_contact_contexts
            if context.get("visible_contact_aligned")
        ),
        "visible_contact_close_steps": sum(
            1
            for context in visible_contact_contexts
            if context.get("visible_contact_close")
        ),
        "mean_visible_contact_distance_norm": _mean_context_value(
            [
                context
                for context in visible_contact_contexts
                if context.get("visible_contact_active")
            ],
            "visible_contact_distance_norm",
        ),
        "mean_action_mask_size": round(
            sum(action_mask_sizes) / max(1, len(action_mask_sizes)),
            4,
        ),
        "invalid_action_steps": invalid_action_steps,
        "reset_warmup_tics": sum(int(warmup.get("tics", 0)) for warmup in warmups),
        "reset_warmup_steps": sum(int(warmup.get("steps", 0)) for warmup in warmups),
        "reset_warmup_stop_reasons": dict(sorted(warmup_reasons.items())),
        "reset_context_sources": dict(sorted(reset_sources.items())),
        "snapshot_restore_count": len(snapshot_contexts),
        "snapshot_restore_failures": sum(
            1
            for context in snapshot_contexts
            if isinstance(context.get("restore", {}), dict)
            and int(context.get("restore", {}).get("returncode", 0)) != 0
        ),
        "snapshot_verification_count": len(snapshot_verifications),
        "snapshot_verification_failures": sum(
            1 for verification in snapshot_verifications if not verification.get("valid")
        ),
        "snapshot_verification_skipped": sum(
            1 for verification in snapshot_verifications if verification.get("skipped")
        ),
        "snapshot_stage_counts": dict(sorted(snapshot_stage_counts.items())),
        "mean_snapshot_restore_seconds": round(
            sum(snapshot_restore_durations) / max(1, len(snapshot_restore_durations)),
            4,
        ),
        "max_snapshot_restore_seconds": round(
            max(snapshot_restore_durations, default=0.0),
            4,
        ),
        "positive_reward_steps": sum(1 for record in records if record.reward > 0),
        "negative_reward_steps": sum(1 for record in records if record.reward < 0),
        "done_count": sum(1 for record in records if record.done),
        "kill_delta": kill_delta,
        "damage_delta": damage_delta,
        "max_kill_gain": max(kill_gains, default=0),
        "snapshot_kill_delta": sum(
            int(transition.get("kill_delta", 0)) for transition in snapshot_transitions
        ),
        "snapshot_damage_delta": sum(
            int(transition.get("damage_delta", 0)) for transition in snapshot_transitions
        ),
        "snapshot_max_kill_gain": max(snapshot_kill_gains, default=0),
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


def _learning_trace_group(record: object, group_name: str) -> dict[str, object]:
    """Returns a compact observation feature group from a rollout record."""
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
    group = groups.get(group_name, {})
    return group if isinstance(group, dict) else {}


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


def _mean_context_value(
    contexts: list[dict[str, object]],
    key: str,
) -> float:
    values = [float(context.get(key, 0.0)) for context in contexts]
    return round(sum(values) / max(1, len(values)), 4)


def _earned_kill_gain(record: object) -> int:
    info = getattr(record, "info", {})
    if not isinstance(info, dict):
        return 0
    state = info.get("state", {})
    if not isinstance(state, dict):
        return 0
    kills = _int_field(state, "kills")
    baseline = _record_kill_baseline(info)
    return max(0, kills - baseline)


def _record_kill_baseline(info: dict[str, object]) -> int:
    reset_context = info.get("reset_context", {})
    if isinstance(reset_context, dict):
        for key in ("actual_first_state", "expected_state"):
            value = reset_context.get(key, {})
            if isinstance(value, dict) and "kills" in value:
                return _int_field(value, "kills")
    curriculum_stage = info.get("curriculum_stage", {})
    if isinstance(curriculum_stage, dict):
        expected = curriculum_stage.get("expected_state", {})
        if isinstance(expected, dict) and "kills" in expected:
            return _int_field(expected, "kills")
    return 0


def _record_reset_source(record: object) -> str:
    info = getattr(record, "info", {})
    if not isinstance(info, dict):
        return ""
    reset_context = info.get("reset_context", {})
    if not isinstance(reset_context, dict):
        return ""
    return str(reset_context.get("source", ""))


def _int_field(values: dict[str, object], key: str) -> int:
    try:
        return int(values.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _checkpoint_selection_score(summary: dict[str, object]) -> float:
    """Scores a rollout for resume selection without changing promotion rules."""
    total_reward = float(summary.get("total_reward", 0.0))
    earned_kills = max(
        int(summary.get("kill_delta", 0)),
        int(summary.get("max_kill_gain", summary.get("max_kills", 0))),
    )
    damage_delta = int(summary.get("damage_delta", 0))
    first_shootable_contacts = int(summary.get("first_shootable_contacts", 0))
    fire_on_shootable = int(summary.get("fire_on_shootable_steps", 0))
    missed_shootable_fire = int(summary.get("missed_shootable_fire_steps", 0))
    route_failed = int(summary.get("route_failed_steps", 0))
    return round(
        total_reward
        + earned_kills * 75.0
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
        "--snapshot-curriculum",
        type=Path,
        help=(
            "JSON manifest with snapshot-backed progressed-state curriculum stages. "
            "Cannot be combined with --curriculum or --reset-start-*."
        ),
    )
    parser.add_argument(
        "--snapshot-restore-command",
        help=(
            "Command template run before each snapshot reset. Supports placeholders "
            "such as {snapshot_id}, {snapshot_path_sh}, {stage_name}, and {reset_index}. "
            "Optional when each snapshot stage has snapshot.slot or ref=save_slot:N."
        ),
    )
    parser.add_argument("--snapshot-restore-cwd", type=Path)
    parser.add_argument("--snapshot-restore-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--no-snapshot-verify-restored-state",
        dest="snapshot_verify_restored_state",
        action="store_false",
        default=True,
        help="Disable expected_state checks after snapshot restore. Intended for debugging only.",
    )
    parser.add_argument("--snapshot-verify-tick-tolerance", type=int, default=35)
    parser.add_argument(
        "--snapshot-verify-stream-tick",
        action="store_true",
        help=(
            "Also verify GameState.tick after snapshot restore. Normally disabled "
            "because native Doom saves restore level_time while stream tick keeps advancing."
        ),
    )
    parser.add_argument(
        "--snapshot-verify-position-tolerance-fp",
        type=int,
        default=160 * 65536,
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
    parser.add_argument(
        "--rollout-stage-mix",
        choices=["off", "round_robin", "random"],
        default="off",
        help=(
            "Rotate curriculum stages between episode resets inside one PPO rollout "
            "buffer. Stage changes happen only after done=True boundaries."
        ),
    )
    parser.add_argument(
        "--rollout-stage-segment-tics",
        type=int,
        default=0,
        help=(
            "When stage mixing is enabled, cap each reset episode to this many Doom "
            "tics so short rollouts can include multiple stages. 0 uses --max-steps."
        ),
    )
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
    parser.add_argument(
        "--checkpoint-eval-curriculum",
        action="store_true",
        help=(
            "After each PPO update, evaluate the checkpoint across every active "
            "curriculum stage and use that aggregate score for best-checkpoint resume."
        ),
    )
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=1)
    parser.add_argument("--checkpoint-eval-max-steps", type=int, default=256)
    parser.add_argument("--checkpoint-eval-sample", action="store_true")
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
