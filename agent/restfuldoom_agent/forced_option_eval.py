"""Forced high-level option evaluation for contact-to-combat primitives."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .curriculum import build_curriculum, stage_for_update
from .env import DoomAgentEnv, DoomEnvConfig, SKILL_ACTIONS
from .learning_trace import build_learning_trace
from .ppo import RolloutBuffer
from .ppo_agent import _summarize_buffer
from .snapshot_curriculum import load_snapshot_curriculum

FORCED_OPTION_EVAL_SCHEMA = "restfuldoom.forced_option_eval.v1"


@dataclass(frozen=True)
class ForcedOptionEvalConfig:
    """Configuration for forced option micro-evaluation."""

    endpoint: str = "127.0.0.1:50051"
    token: str | None = None
    agent_port: int = 50051
    tls: bool = False
    authority: str | None = None
    skill: int = 2
    episode: int = 1
    map: int = 1
    seed: int = 0
    run_id: str = "forced-option-eval"
    goal_preset: str = "combat"
    max_steps: int = 64
    macro_steps: int = 64
    memory_path: Path | None = Path("agent_memory/e1m1.json")
    reset_timeout_seconds: float = 5.0
    reset_attempts: int = 2
    curriculum: str = "e1m1-contact-to-combat"
    snapshot_curriculum: Path | None = None
    curriculum_mode: str = "fixed"
    curriculum_start_index: int = 0
    stage_indexes: tuple[int, ...] = ()
    forced_skills: tuple[str, ...] = (
        "close_visible_contact",
        "seek_enemy",
        "open_use_line",
    )
    first_shootable_bonus: float = 0.0
    visible_contact_progress_reward: float = 0.001
    terminate_on_first_visible: bool = False
    terminate_on_first_shootable: bool = False
    terminate_on_required_kills: bool = False
    shootable_handoff_skill: str | None = None
    snapshot_verify_restored_state: bool = True
    snapshot_verify_tick_tolerance: int = 35
    snapshot_verify_stream_tick: bool = False
    snapshot_verify_position_tolerance_fp: int = 160 * 65536
    output_path: Path | None = None
    jsonl_path: Path | None = None


async def run_forced_option_eval(config: ForcedOptionEvalConfig) -> dict[str, Any]:
    """Runs forced skill rollouts and returns a JSON-serializable report."""
    curriculum = _build_curriculum(config)
    stages = _selected_stages(curriculum, config.stage_indexes)
    runs: list[dict[str, Any]] = []
    started = time.time()
    for stage in stages:
        for forced_skill in config.forced_skills:
            runs.append(await _run_one(config, curriculum, stage, forced_skill))
    report = {
        "schema": FORCED_OPTION_EVAL_SCHEMA,
        "created_at_epoch_seconds": int(started),
        "elapsed_seconds": round(time.time() - started, 4),
        "curriculum": _curriculum_metadata(curriculum),
        "forced_skills": list(config.forced_skills),
        "macro_steps": int(config.macro_steps),
        "runs": runs,
        "comparison": _comparison(runs),
    }
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


async def _run_one(
    config: ForcedOptionEvalConfig,
    curriculum: dict[str, Any],
    stage: dict[str, Any],
    forced_skill: str,
) -> dict[str, Any]:
    action_index = _skill_action_index(forced_skill)
    env = DoomAgentEnv(_env_config_for_stage(config, curriculum, stage, forced_skill))
    buffer = RolloutBuffer()
    reset_error: str | None = None
    termination_reason: str | None = None
    termination_step: int | None = None
    started = time.time()
    try:
        try:
            obs = await env.reset(seed=config.seed)
        except Exception as error:  # noqa: BLE001 - report reset failures as experiment data.
            reset_error = str(error)
            return {
                "schema": "restfuldoom.forced_option_eval_run.v1",
                "forced_skill": forced_skill,
                "action_index": action_index,
                "stage": _compact_stage(stage),
                "ok": False,
                "reset_error": reset_error,
                "elapsed_seconds": round(time.time() - started, 4),
            }
        for _ in range(max(1, int(config.macro_steps))):
            action_mask = env.action_mask()
            selected_action_index = action_index
            selected_skill = forced_skill
            forced_action_allowed = _action_allowed(action_mask, action_index)
            handoff_applied = False
            if (
                not forced_action_allowed
                and config.shootable_handoff_skill is not None
            ):
                handoff_index = _skill_action_index(config.shootable_handoff_skill)
                if _action_allowed(action_mask, handoff_index):
                    selected_action_index = handoff_index
                    selected_skill = config.shootable_handoff_skill
                    handoff_applied = True
            if not forced_action_allowed and not handoff_applied:
                termination_reason = _forced_option_stop_reason(
                    forced_skill,
                    action_mask,
                )
                termination_step = len(buffer.records)
                break
            transition = await env.step(selected_action_index)
            info = dict(transition.info)
            info["forced_skill"] = forced_skill
            info["forced_action_index"] = action_index
            info["forced_action_allowed"] = forced_action_allowed
            info["selected_forced_skill"] = selected_skill
            info["selected_action_index"] = selected_action_index
            info["selected_action_allowed"] = _action_allowed(
                action_mask,
                selected_action_index,
            )
            info["shootable_handoff_skill"] = config.shootable_handoff_skill
            info["shootable_handoff_applied"] = handoff_applied
            info["learning_trace"] = build_learning_trace(
                obs=obs,
                action_mask=action_mask,
                action=selected_action_index,
                reward=transition.reward,
                done=transition.done,
                info=info,
            )
            buffer.add(
                obs=obs,
                action_mask=action_mask,
                action=selected_action_index,
                reward=transition.reward,
                done=transition.done,
                value=0.0,
                logprob=0.0,
                info=info,
            )
            _write_jsonl_record(config, stage, forced_skill, buffer.records[-1])
            obs = transition.observation
            if transition.done:
                termination_reason = str(info.get("done_reason") or "env_done")
                termination_step = len(buffer.records)
                break
    finally:
        await env.close()
    summary = _summarize_buffer(buffer)
    forced_summary = _forced_summary(buffer)
    return {
        "schema": "restfuldoom.forced_option_eval_run.v1",
        "forced_skill": forced_skill,
        "action_index": action_index,
        "stage": _compact_stage(stage),
        "ok": True,
        "elapsed_seconds": round(time.time() - started, 4),
        "termination_reason": termination_reason,
        "termination_step": termination_step,
        "summary": summary,
        "forced_summary": forced_summary,
    }


def _build_curriculum(config: ForcedOptionEvalConfig) -> dict[str, Any]:
    if config.snapshot_curriculum is not None:
        return load_snapshot_curriculum(
            config.snapshot_curriculum,
            mode=config.curriculum_mode,
            start_index=config.curriculum_start_index,
            seed=config.seed,
        )
    return build_curriculum(
        name=config.curriculum,
        manual_reset_start={},
        mode=config.curriculum_mode,
        start_index=config.curriculum_start_index,
        seed=config.seed,
    )


def _selected_stages(
    curriculum: dict[str, Any],
    requested_indexes: tuple[int, ...],
) -> list[dict[str, Any]]:
    stages = curriculum.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum has no stages")
    if not requested_indexes:
        return [stage_for_update(curriculum, 0)]
    selected = []
    for index in requested_indexes:
        if index < 0 or index >= len(stages):
            raise ValueError(f"stage index {index} outside range 0..{len(stages) - 1}")
        stage = dict(stages[index])
        stage["selected_index"] = int(index)
        selected.append(stage)
    return selected


def _env_config_for_stage(
    config: ForcedOptionEvalConfig,
    curriculum: dict[str, Any],
    stage: dict[str, Any],
    forced_skill: str,
) -> DoomEnvConfig:
    reset_start = stage.get("reset_start", {})
    start = reset_start if isinstance(reset_start, dict) else {}
    snapshot = stage.get("snapshot") if isinstance(stage.get("snapshot"), dict) else None
    reset_mode = "snapshot" if snapshot is not None or stage.get("reset_mode") == "snapshot" else "episode"
    return DoomEnvConfig(
        endpoint=config.endpoint,
        token=config.token,
        agent_port=config.agent_port,
        tls=config.tls,
        authority=config.authority,
        skill=config.skill,
        episode=config.episode,
        map=config.map,
        seed=config.seed,
        run_id=f"{config.run_id}-{stage.get('name', 'stage')}-{forced_skill}",
        goal_preset=config.goal_preset,
        max_steps=config.max_steps,
        memory_path=config.memory_path,
        reset_timeout_seconds=config.reset_timeout_seconds,
        reset_attempts=config.reset_attempts,
        reset_start_x_fp=start.get("x_fp"),
        reset_start_y_fp=start.get("y_fp"),
        reset_start_angle_degrees=int(start.get("angle_degrees", 0)),
        reset_start_face_nearest_enemy=bool(start.get("face_nearest_enemy", False)),
        reset_start_health=start.get("health"),
        reset_start_armor=start.get("armor"),
        reset_start_ammo_bullets=start.get("ammo_bullets"),
        first_shootable_bonus=config.first_shootable_bonus,
        visible_contact_progress_reward=config.visible_contact_progress_reward,
        terminate_on_first_visible=config.terminate_on_first_visible,
        terminate_on_first_shootable=config.terminate_on_first_shootable,
        terminate_on_required_kills=config.terminate_on_required_kills,
        curriculum=_curriculum_metadata(curriculum),
        curriculum_stage=dict(stage),
        reset_mode=reset_mode,
        snapshot=snapshot,
        snapshot_verify_restored_state=config.snapshot_verify_restored_state,
        snapshot_verify_tick_tolerance=config.snapshot_verify_tick_tolerance,
        snapshot_verify_stream_tick=config.snapshot_verify_stream_tick,
        snapshot_verify_position_tolerance_fp=config.snapshot_verify_position_tolerance_fp,
    )


def _forced_summary(buffer: RolloutBuffer) -> dict[str, Any]:
    records = buffer.records
    allowed_steps = sum(
        1 for record in records if bool(record.info.get("forced_action_allowed"))
    )
    selected_allowed_steps = sum(
        1 for record in records if bool(record.info.get("selected_action_allowed"))
    )
    handoff_steps = sum(
        1 for record in records if bool(record.info.get("shootable_handoff_applied"))
    )
    visible_seen = False
    lost_visible_after_contact = 0
    first_shootable_step: int | None = None
    for index, record in enumerate(records):
        had_visible = bool(record.info.get("had_visible_enemy"))
        had_shootable = bool(record.info.get("had_shootable_target"))
        if had_visible:
            visible_seen = True
        elif visible_seen:
            lost_visible_after_contact += 1
        if first_shootable_step is None and had_shootable:
            first_shootable_step = index
    actual_skills: dict[str, int] = {}
    decision_skills: dict[str, int] = {}
    stuck_steps = 0
    recovery_steps = 0
    first_contact_use_line_close_step: int | None = None
    for record in records:
        skill = str(record.info.get("skill", "unknown"))
        actual_skills[skill] = actual_skills.get(skill, 0) + 1
    for index, record in enumerate(records):
        decision = record.info.get("decision", {})
        decision_skill = (
            str(decision.get("skill", "unknown"))
            if isinstance(decision, dict)
            else "unknown"
        )
        decision_skills[decision_skill] = decision_skills.get(decision_skill, 0) + 1
        if isinstance(decision, dict) and bool(decision.get("stuck")):
            stuck_steps += 1
        if "recover" in decision_skill or "unstick" in decision_skill:
            recovery_steps += 1
        contact_context = _record_trace_group(record, "contact")
        if (
            first_contact_use_line_close_step is None
            and contact_context.get("contact_use_line_close")
        ):
            first_contact_use_line_close_step = index
    return {
        "records": len(records),
        "forced_allowed_steps": allowed_steps,
        "forced_disallowed_steps": len(records) - allowed_steps,
        "forced_action_allowed_steps": allowed_steps,
        "forced_action_disallowed_steps": len(records) - allowed_steps,
        "selected_allowed_steps": selected_allowed_steps,
        "selected_disallowed_steps": len(records) - selected_allowed_steps,
        "selected_action_allowed_steps": selected_allowed_steps,
        "selected_action_disallowed_steps": len(records) - selected_allowed_steps,
        "shootable_handoff_steps": handoff_steps,
        "forced_handoff_disallowed_steps": handoff_steps,
        "unhandled_forced_disallowed_steps": len(records) - allowed_steps - handoff_steps,
        "actual_skill_counts": dict(sorted(actual_skills.items())),
        "decision_skill_counts": dict(sorted(decision_skills.items())),
        "lost_visible_contact_steps": lost_visible_after_contact,
        "first_shootable_step": first_shootable_step,
        "first_contact_use_line_close_step": first_contact_use_line_close_step,
        "stuck_steps": stuck_steps,
        "recovery_steps": recovery_steps,
    }


def _forced_option_stop_reason(forced_skill: str, action_mask: list[bool]) -> str:
    """Classify pre-step stops so completed options are not reported as invalid."""
    if forced_skill == "close_visible_contact":
        fire_index = SKILL_ACTIONS.index("fire")
        if _action_allowed(action_mask, fire_index):
            return "forced_option_completed_shootable"
    return "forced_option_disallowed"


def _comparison(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        summary = run.get("summary", {}) if run.get("ok") else {}
        forced = run.get("forced_summary", {}) if run.get("ok") else {}
        rows.append(
            {
                "stage": run.get("stage", {}).get("name"),
                "forced_skill": run.get("forced_skill"),
                "ok": bool(run.get("ok")),
                "first_shootable_contacts": int(summary.get("first_shootable_contacts", 0)),
                "shootable_target_steps": int(summary.get("shootable_target_steps", 0)),
                "damage_delta": int(summary.get("damage_delta", 0)),
                "kill_delta": int(summary.get("kill_delta", 0)),
                "enemy_distance_delta": float(summary.get("enemy_distance_delta", 0.0)),
                "visible_contact_distance_delta": float(
                    summary.get("visible_contact_distance_delta", 0.0)
                ),
                "contact_use_line_close_steps": int(
                    summary.get("contact_use_line_close_steps", 0)
                ),
                "invalid_action_steps": int(summary.get("invalid_action_steps", 0)),
                "lost_visible_contact_steps": int(
                    forced.get("lost_visible_contact_steps", 0)
                ),
                "forced_disallowed_steps": int(
                    forced.get("forced_disallowed_steps", 0)
                ),
                "selected_disallowed_steps": int(
                    forced.get("selected_disallowed_steps", 0)
                ),
                "shootable_handoff_steps": int(
                    forced.get("shootable_handoff_steps", 0)
                ),
                "unhandled_forced_disallowed_steps": int(
                    forced.get("unhandled_forced_disallowed_steps", 0)
                ),
                "termination_reason": run.get("termination_reason"),
                "termination_step": run.get("termination_step"),
                "stuck_steps": int(forced.get("stuck_steps", 0)),
                "recovery_steps": int(forced.get("recovery_steps", 0)),
                "reset_error": run.get("reset_error"),
            }
        )
    return rows


def _record_trace_group(record: Any, group: str) -> dict[str, Any]:
    trace = record.info.get("learning_trace", {})
    if not isinstance(trace, dict):
        return {}
    observation = trace.get("observation", {})
    if not isinstance(observation, dict):
        return {}
    groups = observation.get("groups", {})
    if not isinstance(groups, dict):
        return {}
    selected = groups.get(group, {})
    return selected if isinstance(selected, dict) else {}


def _write_jsonl_record(
    config: ForcedOptionEvalConfig,
    stage: dict[str, Any],
    forced_skill: str,
    record: Any,
) -> None:
    if config.jsonl_path is None:
        return
    config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "restfuldoom.forced_option_eval_record.v1",
        "stage": _compact_stage(stage),
        "forced_skill": forced_skill,
        "record": {
            "obs": list(record.obs),
            "action": record.action,
            "action_mask": list(record.action_mask),
            "reward": record.reward,
            "done": record.done,
            "info": record.info,
        },
    }
    with config.jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _skill_action_index(skill: str) -> int:
    try:
        return SKILL_ACTIONS.index(skill)
    except ValueError as error:
        choices = ", ".join(SKILL_ACTIONS)
        raise ValueError(f"unknown forced skill {skill!r}; choose one of: {choices}") from error


def _action_allowed(action_mask: list[bool], action_index: int) -> bool:
    return 0 <= action_index < len(action_mask) and bool(action_mask[action_index])


def _curriculum_metadata(curriculum: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": curriculum.get("schema"),
        "name": curriculum.get("name"),
        "mode": curriculum.get("mode"),
        "start_index": curriculum.get("start_index"),
    }


def _compact_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": stage.get("index"),
        "selected_index": stage.get("selected_index"),
        "name": stage.get("name"),
        "requires_progressed_state": bool(stage.get("requires_progressed_state")),
        "snapshot": stage.get("snapshot"),
        "expected_state": stage.get("expected_state"),
        "reset_start": stage.get("reset_start"),
    }


def _config_from_args(args: argparse.Namespace) -> ForcedOptionEvalConfig:
    stage_indexes = tuple(int(index) for index in args.stage_index)
    return ForcedOptionEvalConfig(
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
        max_steps=args.max_steps,
        macro_steps=args.macro_steps,
        memory_path=args.memory_path,
        reset_timeout_seconds=args.reset_timeout_seconds,
        reset_attempts=args.reset_attempts,
        curriculum=args.curriculum,
        snapshot_curriculum=args.snapshot_curriculum,
        curriculum_mode=args.curriculum_mode,
        curriculum_start_index=args.curriculum_start_index,
        stage_indexes=stage_indexes,
        forced_skills=tuple(args.force_skill),
        first_shootable_bonus=args.first_shootable_bonus,
        visible_contact_progress_reward=args.visible_contact_progress_reward,
        terminate_on_first_visible=args.terminate_on_first_visible,
        terminate_on_first_shootable=args.terminate_on_first_shootable,
        terminate_on_required_kills=args.terminate_on_required_kills,
        shootable_handoff_skill=args.shootable_handoff_skill,
        snapshot_verify_restored_state=not args.no_snapshot_verify_restored_state,
        snapshot_verify_tick_tolerance=args.snapshot_verify_tick_tolerance,
        snapshot_verify_stream_tick=args.snapshot_verify_stream_tick,
        snapshot_verify_position_tolerance_fp=args.snapshot_verify_position_tolerance_fp,
        output_path=args.output,
        jsonl_path=args.jsonl,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--token")
    parser.add_argument("--agent-port", type=int, default=50051)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--authority")
    parser.add_argument("--skill", type=int, default=2)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--map", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="forced-option-eval")
    parser.add_argument("--goal-preset", default="combat")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--macro-steps", type=int, default=64)
    parser.add_argument("--memory-path", type=Path, default=Path("agent_memory/e1m1.json"))
    parser.add_argument("--reset-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--reset-attempts", type=int, default=2)
    parser.add_argument("--curriculum", default="e1m1-contact-to-combat")
    parser.add_argument("--snapshot-curriculum", type=Path)
    parser.add_argument("--curriculum-mode", default="fixed")
    parser.add_argument("--curriculum-start-index", type=int, default=0)
    parser.add_argument("--stage-index", action="append", type=int, default=[])
    parser.add_argument(
        "--force-skill",
        action="append",
        default=[],
        choices=SKILL_ACTIONS,
        help="skill to force; defaults to close_visible_contact, seek_enemy, open_use_line",
    )
    parser.add_argument("--first-shootable-bonus", type=float, default=0.0)
    parser.add_argument("--visible-contact-progress-reward", type=float, default=0.001)
    parser.add_argument("--terminate-on-first-visible", action="store_true")
    parser.add_argument("--terminate-on-first-shootable", action="store_true")
    parser.add_argument("--terminate-on-required-kills", action="store_true")
    parser.add_argument(
        "--shootable-handoff-skill",
        choices=SKILL_ACTIONS,
        help=(
            "optional skill to use when the forced skill becomes masked but the "
            "handoff skill is currently allowed"
        ),
    )
    parser.add_argument("--no-snapshot-verify-restored-state", action="store_true")
    parser.add_argument("--snapshot-verify-tick-tolerance", type=int, default=35)
    parser.add_argument("--snapshot-verify-stream-tick", action="store_true")
    parser.add_argument(
        "--snapshot-verify-position-tolerance-fp",
        type=int,
        default=160 * 65536,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args(argv)
    if not args.force_skill:
        args.force_skill = ["close_visible_contact", "seek_enemy", "open_use_line"]
    report = asyncio.run(run_forced_option_eval(_config_from_args(args)))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
