"""Validate strict post-combat exit-routing eval artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

POST_COMBAT_EXIT_GATE_SCHEMA = "restfuldoom.post_combat_exit_gate.v1"
DEFAULT_ALLOWED_SKILLS = ("route_progression", "press_exit", "open_use_line")


def validate_post_combat_exit_gate(
    payload: dict[str, Any],
    *,
    stage_name: str | None = None,
    allowed_skills: tuple[str, ...] = DEFAULT_ALLOWED_SKILLS,
    required_start_kills: int = 5,
    min_episodes: int = 1,
    min_exit_route_attempt_steps: int = 1,
    min_exit_route_progress_units: float = 0.0,
    require_snapshot_restore: bool = True,
    require_level_complete: bool = True,
    require_strict_skill_filter: bool = True,
) -> dict[str, Any]:
    """Returns a pass/fail report for a post-combat exit-routing gate."""
    allowed = tuple(allowed_skills or DEFAULT_ALLOWED_SKILLS)
    blocks = _extract_eval_blocks(payload, stage_name=stage_name)
    failures: list[dict[str, Any]] = []
    failed_episode_keys: set[tuple[str, int]] = set()
    summary: dict[str, Any] = {
        "stage_count": len(blocks),
        "episode_count": 0,
        "passed_episodes": 0,
        "level_transition_episode_count": 0,
        "level_transition_delta_total": 0,
        "level_complete_episode_count": 0,
        "invalid_action_steps": 0,
        "selected_disallowed_steps": 0,
        "action_mask_fallback_steps": 0,
        "allowed_skill_filter_steps": 0,
        "allowed_skill_filter_fallback_steps": 0,
        "strict_allowed_skill_filter_steps": 0,
        "strict_allowed_skill_fallback_steps": 0,
        "snapshot_verification_failures": 0,
        "exit_route_attempt_steps": 0,
        "exit_route_reached_steps": 0,
        "exit_route_failed_steps": 0,
        "exit_route_progress_units": 0.0,
        "min_start_kills": None,
        "disallowed_skill_counts": {},
        "configured_allowed_skills": [],
    }
    configured_allowed: set[str] = set()
    disallowed_skill_counts: Counter[str] = Counter()

    if not blocks:
        failures.append(
            {
                "stage": stage_name,
                "episode_index": None,
                "reason": "no_eval_episodes",
                "detail": "input did not contain a matching PolicyEval payload",
            }
        )

    for block_index, block in enumerate(blocks):
        stage_label = str(block.get("stage_name") or f"stage_{block_index}")
        eval_payload = block.get("eval", {})
        metadata = block.get("metadata", {})
        if isinstance(metadata, dict):
            configured_allowed.update(
                str(skill)
                for skill in metadata.get("allowed_skills", []) or []
                if str(skill)
            )
        strict_metadata = bool(
            isinstance(metadata, dict) and metadata.get("strict_allowed_skills")
        )
        episodes = eval_payload.get("episodes", []) if isinstance(eval_payload, dict) else []
        if not isinstance(episodes, list) or not episodes:
            failure = {
                "stage": stage_label,
                "episode_index": None,
                "reason": "no_stage_episodes",
            }
            failures.append(failure)
            continue

        for episode_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            episode_key = (stage_label, episode_index)
            summary["episode_count"] += 1
            start_kills = _int_field(episode, "start_kills")
            min_start = summary["min_start_kills"]
            summary["min_start_kills"] = (
                start_kills if min_start is None else min(int(min_start), start_kills)
            )
            level_transition_delta = _int_field(episode, "level_transition_delta")
            summary["level_transition_delta_total"] += level_transition_delta
            if level_transition_delta > 0:
                summary["level_transition_episode_count"] += 1
            if _episode_level_completed(episode):
                summary["level_complete_episode_count"] += 1
            for field in (
                "invalid_action_steps",
                "selected_disallowed_steps",
                "action_mask_fallback_steps",
                "allowed_skill_filter_steps",
                "allowed_skill_filter_fallback_steps",
                "strict_allowed_skill_filter_steps",
                "strict_allowed_skill_fallback_steps",
                "snapshot_verification_failures",
                "exit_route_attempt_steps",
                "exit_route_reached_steps",
                "exit_route_failed_steps",
            ):
                summary[field] += _int_field(episode, field)
            summary["exit_route_progress_units"] += _float_field(
                episode,
                "exit_route_progress_units",
            )

            def fail(reason: str, **details: Any) -> None:
                failed_episode_keys.add(episode_key)
                failures.append(
                    {
                        "stage": stage_label,
                        "episode_index": episode_index,
                        "seed": episode.get("seed"),
                        "reason": reason,
                        **details,
                    }
                )

            if require_snapshot_restore and episode.get("reset_source") != "snapshot_restore":
                fail(
                    "not_snapshot_restored",
                    reset_source=episode.get("reset_source"),
                )
            if start_kills < int(required_start_kills):
                fail(
                    "post_combat_start_kills_below_threshold",
                    start_kills=start_kills,
                    required_start_kills=int(required_start_kills),
                )
            if level_transition_delta != 1:
                fail(
                    "missing_level_transition",
                    level_transition_delta=level_transition_delta,
                )
            if require_level_complete and not _episode_level_completed(episode):
                fail(
                    "missing_level_complete_done",
                    done_reason=episode.get("done_reason"),
                    level_completed=episode.get("level_completed"),
                )
            if _int_field(episode, "invalid_action_steps") != 0:
                fail(
                    "invalid_action_steps_present",
                    invalid_action_steps=_int_field(episode, "invalid_action_steps"),
                )
            if _int_field(episode, "selected_disallowed_steps") != 0:
                fail(
                    "selected_disallowed_steps_present",
                    selected_disallowed_steps=_int_field(
                        episode,
                        "selected_disallowed_steps",
                    ),
                )
            if _int_field(episode, "action_mask_fallback_steps") != 0:
                fail(
                    "action_mask_fallback_steps_present",
                    action_mask_fallback_steps=_int_field(
                        episode,
                        "action_mask_fallback_steps",
                    ),
                )
            if _int_field(episode, "strict_allowed_skill_fallback_steps") != 0:
                fail(
                    "strict_allowed_skill_fallback_steps_present",
                    strict_allowed_skill_fallback_steps=_int_field(
                        episode,
                        "strict_allowed_skill_fallback_steps",
                    ),
                )
            if _int_field(episode, "snapshot_verification_failures") != 0:
                fail(
                    "snapshot_verification_failures_present",
                    snapshot_verification_failures=_int_field(
                        episode,
                        "snapshot_verification_failures",
                    ),
                )
            if require_strict_skill_filter:
                if _int_field(episode, "allowed_skill_filter_steps") <= 0:
                    fail("allowed_skill_filter_missing")
                if (
                    not strict_metadata
                    and _int_field(episode, "strict_allowed_skill_filter_steps") <= 0
                ):
                    fail("strict_allowed_skill_filter_missing")
            if _int_field(episode, "exit_route_attempt_steps") < int(
                min_exit_route_attempt_steps
            ):
                fail(
                    "exit_route_attempt_steps_below_threshold",
                    exit_route_attempt_steps=_int_field(
                        episode,
                        "exit_route_attempt_steps",
                    ),
                    minimum=int(min_exit_route_attempt_steps),
                )
            if _float_field(episode, "exit_route_progress_units") < float(
                min_exit_route_progress_units
            ):
                fail(
                    "exit_route_progress_units_below_threshold",
                    exit_route_progress_units=_float_field(
                        episode,
                        "exit_route_progress_units",
                    ),
                    minimum=float(min_exit_route_progress_units),
                )

            for skill, count in _skill_counts(episode).items():
                if skill not in allowed:
                    disallowed_skill_counts[skill] += count
            disallowed = {
                skill: count
                for skill, count in _skill_counts(episode).items()
                if skill not in allowed
            }
            if disallowed:
                fail(
                    "disallowed_skill_executed",
                    disallowed_skill_counts=dict(sorted(disallowed.items())),
                )

    if summary["episode_count"] < int(min_episodes):
        failures.append(
            {
                "stage": stage_name,
                "episode_index": None,
                "reason": "episode_count_below_threshold",
                "episode_count": summary["episode_count"],
                "minimum": int(min_episodes),
            }
        )

    summary["passed_episodes"] = summary["episode_count"] - len(failed_episode_keys)
    summary["exit_route_progress_units"] = round(
        float(summary["exit_route_progress_units"]),
        4,
    )
    summary["disallowed_skill_counts"] = dict(sorted(disallowed_skill_counts.items()))
    summary["configured_allowed_skills"] = sorted(configured_allowed)

    return {
        "schema": POST_COMBAT_EXIT_GATE_SCHEMA,
        "created_at_epoch_seconds": int(time.time()),
        "ok": not failures,
        "config": {
            "stage_name": stage_name,
            "allowed_skills": list(allowed),
            "required_start_kills": int(required_start_kills),
            "min_episodes": int(min_episodes),
            "min_exit_route_attempt_steps": int(min_exit_route_attempt_steps),
            "min_exit_route_progress_units": float(min_exit_route_progress_units),
            "require_snapshot_restore": bool(require_snapshot_restore),
            "require_level_complete": bool(require_level_complete),
            "require_strict_skill_filter": bool(require_strict_skill_filter),
        },
        "summary": summary,
        "failures": failures,
    }


def _extract_eval_blocks(
    payload: dict[str, Any],
    *,
    stage_name: str | None = None,
) -> list[dict[str, Any]]:
    """Extracts PolicyEval-shaped payloads from ppo_agent eval JSON variants."""
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("checkpoint_eval"), dict):
        return _extract_eval_blocks(payload["checkpoint_eval"], stage_name=stage_name)
    if isinstance(payload.get("candidate"), dict):
        return _extract_eval_blocks(payload["candidate"], stage_name=stage_name)
    if isinstance(payload.get("result"), dict) and isinstance(payload.get("episodes"), list):
        return [
            {
                "stage_name": str(stage_name or payload.get("stage_name") or "eval"),
                "eval": payload,
                "metadata": payload,
            }
        ]
    stages = payload.get("stages")
    if isinstance(stages, list):
        blocks: list[dict[str, Any]] = []
        metadata = {
            "allowed_skills": payload.get("allowed_skills", []),
            "strict_allowed_skills": payload.get("strict_allowed_skills", False),
        }
        for index, stage_record in enumerate(stages):
            if not isinstance(stage_record, dict):
                continue
            stage = stage_record.get("stage", {})
            if not isinstance(stage, dict):
                stage = {}
            name = str(stage.get("name") or f"stage_{index}")
            if stage_name is not None and name != stage_name:
                continue
            result = stage_record.get("result", {})
            if isinstance(result, dict) and isinstance(result.get("episodes"), list):
                blocks.append(
                    {
                        "stage_name": name,
                        "eval": result,
                        "metadata": metadata,
                    }
                )
        return blocks
    return []


def _episode_level_completed(episode: dict[str, Any]) -> bool:
    return bool(episode.get("level_completed")) or episode.get("done_reason") == "level_complete"


def _skill_counts(episode: dict[str, Any]) -> dict[str, int]:
    counts = episode.get("skill_counts", {})
    if not isinstance(counts, dict):
        return {}
    return {
        str(skill): max(0, _int_value(count))
        for skill, count in counts.items()
        if str(skill)
    }


def _int_field(payload: dict[str, Any], field: str) -> int:
    return _int_value(payload.get(field, 0))


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_field(payload: dict[str, Any], field: str) -> float:
    try:
        return float(payload.get(field, 0.0))
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_json", type=Path)
    parser.add_argument("--stage-name")
    parser.add_argument(
        "--allowed-skill",
        action="append",
        default=[],
        help="Executed skill allowed by the gate. Defaults to route/exit skills.",
    )
    parser.add_argument("--required-start-kills", type=int, default=5)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--min-exit-route-attempt-steps", type=int, default=1)
    parser.add_argument("--min-exit-route-progress-units", type=float, default=0.0)
    parser.add_argument("--allow-non-snapshot", action="store_true")
    parser.add_argument("--allow-missing-level-complete", action="store_true")
    parser.add_argument("--allow-nonstrict-skill-filter", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    report = validate_post_combat_exit_gate(
        payload,
        stage_name=args.stage_name,
        allowed_skills=tuple(args.allowed_skill or DEFAULT_ALLOWED_SKILLS),
        required_start_kills=args.required_start_kills,
        min_episodes=args.min_episodes,
        min_exit_route_attempt_steps=args.min_exit_route_attempt_steps,
        min_exit_route_progress_units=args.min_exit_route_progress_units,
        require_snapshot_restore=not args.allow_non_snapshot,
        require_level_complete=not args.allow_missing_level_complete,
        require_strict_skill_filter=not args.allow_nonstrict_skill_filter,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
