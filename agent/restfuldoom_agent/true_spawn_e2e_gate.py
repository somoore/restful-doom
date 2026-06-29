"""Validate true-spawn end-to-end PPO eval artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .brain import POST_COMBAT_EXIT_KILLS
from .schemas import PPO_SKILL_ACTIONS

TRUE_SPAWN_E2E_GATE_SCHEMA = "restfuldoom.true_spawn_e2e_gate.v1"
DEFAULT_ALLOWED_SKILLS = tuple(PPO_SKILL_ACTIONS)
DEFAULT_MIN_KILL_GAIN = POST_COMBAT_EXIT_KILLS
CHAIN_FAILURE_REASONS = {
    "first_visible_contacts_below_threshold",
    "first_shootable_contacts_below_threshold",
    "kill_gain_below_threshold",
    "route_attempt_steps_below_threshold",
    "exit_route_attempt_steps_below_threshold",
    "missing_level_transition",
    "missing_level_complete_done",
}


def validate_true_spawn_e2e_gate(
    payload: dict[str, Any],
    *,
    stage_name: str | None = None,
    allowed_skills: tuple[str, ...] = DEFAULT_ALLOWED_SKILLS,
    min_episodes: int = 1,
    min_level_completions: int = 1,
    min_first_visible_contacts: int = 1,
    min_first_shootable_contacts: int = 1,
    min_kill_gain: int = DEFAULT_MIN_KILL_GAIN,
    min_route_attempt_steps: int = 1,
    min_exit_route_attempt_steps: int = 1,
    require_level_complete: bool = True,
    require_strict_skill_filter: bool = True,
    required_start_episode: int | None = None,
    required_start_map: int | None = None,
) -> dict[str, Any]:
    """Returns a pass/fail report for a fresh episode spawn completion gate."""
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
        "true_spawn_episode_count": 0,
        "seed_applied_episode_count": 0,
        "snapshot_reset_episode_count": 0,
        "non_episode_reset_episode_count": 0,
        "invalid_action_steps": 0,
        "selected_disallowed_steps": 0,
        "action_mask_fallback_steps": 0,
        "allowed_skill_filter_steps": 0,
        "allowed_skill_filter_fallback_steps": 0,
        "strict_allowed_skill_filter_steps": 0,
        "strict_allowed_skill_fallback_steps": 0,
        "snapshot_verification_failures": 0,
        "visible_enemy_steps": 0,
        "first_visible_contacts": 0,
        "first_shootable_contacts": 0,
        "shootable_target_steps": 0,
        "fire_on_shootable_steps": 0,
        "missed_shootable_fire_steps": 0,
        "kill_delta_total": 0,
        "max_kill_gain_total": 0,
        "max_single_episode_kill_gain": 0,
        "damage_delta_total": 0,
        "route_attempt_steps": 0,
        "route_reached_steps": 0,
        "route_failed_steps": 0,
        "route_progress_units": 0.0,
        "exit_route_attempt_steps": 0,
        "exit_route_reached_steps": 0,
        "exit_route_failed_steps": 0,
        "exit_route_progress_units": 0.0,
        "min_start_kills": None,
        "max_start_kills": None,
        "min_start_items": None,
        "max_start_items": None,
        "min_start_secrets": None,
        "max_start_secrets": None,
        "reset_source_counts": {},
        "done_reasons": {},
        "skill_counts": {},
        "disallowed_skill_counts": {},
        "configured_allowed_skills": [],
        "bottleneck_counts": {},
        "blocking_failure_count": 0,
    }
    configured_allowed: set[str] = set()
    reset_source_counts: Counter[str] = Counter()
    done_reasons: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    disallowed_skill_counts: Counter[str] = Counter()
    bottleneck_counts: Counter[str] = Counter()

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
        _collect_aggregate_reset_failures(
            eval_payload,
            failures=failures,
            stage_label=stage_label,
        )
        episodes = eval_payload.get("episodes", []) if isinstance(eval_payload, dict) else []
        if not isinstance(episodes, list) or not episodes:
            failures.append(
                {
                    "stage": stage_label,
                    "episode_index": None,
                    "reason": "no_stage_episodes",
                }
            )
            continue

        for episode_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                continue
            episode_key = (stage_label, episode_index)
            summary["episode_count"] += 1
            reset_source = str(episode.get("reset_source") or "unknown")
            reset_source_counts[reset_source] += 1
            if reset_source == "episode":
                summary["true_spawn_episode_count"] += 1
                if bool(episode.get("seed_applied", False)):
                    summary["seed_applied_episode_count"] += 1
            elif reset_source == "snapshot_restore":
                summary["snapshot_reset_episode_count"] += 1
                summary["non_episode_reset_episode_count"] += 1
            else:
                summary["non_episode_reset_episode_count"] += 1
            done_reason = str(episode.get("done_reason") or "none")
            done_reasons[done_reason] += 1
            start_kills = _int_field(episode, "start_kills")
            start_items = _int_field(episode, "start_items")
            start_secrets = _int_field(episode, "start_secrets")
            _update_min_max(summary, "start_kills", start_kills)
            _update_min_max(summary, "start_items", start_items)
            _update_min_max(summary, "start_secrets", start_secrets)
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
                "visible_enemy_steps",
                "first_visible_contacts",
                "first_shootable_contacts",
                "shootable_target_steps",
                "fire_on_shootable_steps",
                "missed_shootable_fire_steps",
                "route_attempt_steps",
                "route_reached_steps",
                "route_failed_steps",
                "exit_route_attempt_steps",
                "exit_route_reached_steps",
                "exit_route_failed_steps",
            ):
                summary[field] += _int_field(episode, field)
            for field in ("route_progress_units", "exit_route_progress_units"):
                summary[field] += _float_field(episode, field)
            kill_delta = _int_field(episode, "kill_delta")
            max_kill_gain = _episode_kill_gain(episode)
            summary["kill_delta_total"] += kill_delta
            summary["max_kill_gain_total"] += max_kill_gain
            summary["max_single_episode_kill_gain"] = max(
                int(summary["max_single_episode_kill_gain"]),
                max_kill_gain,
            )
            summary["damage_delta_total"] += _int_field(episode, "damage_delta")
            episode_skill_counts = _skill_counts(episode)
            skill_counts.update(episode_skill_counts)

            def fail(reason: str, **details: Any) -> None:
                failed_episode_keys.add(episode_key)
                failure = {
                    "stage": stage_label,
                    "episode_index": episode_index,
                    "seed": episode.get("seed"),
                    "reason": reason,
                    "bottleneck": _classify_bottleneck(
                        episode,
                        min_first_visible_contacts=min_first_visible_contacts,
                        min_first_shootable_contacts=min_first_shootable_contacts,
                        min_kill_gain=min_kill_gain,
                        min_route_attempt_steps=min_route_attempt_steps,
                        min_exit_route_attempt_steps=min_exit_route_attempt_steps,
                    ),
                    **details,
                }
                failures.append(failure)

            if reset_source != "episode":
                fail("not_true_spawn_episode_reset", reset_source=reset_source)
            if start_kills != 0:
                fail("start_kills_not_fresh", start_kills=start_kills)
            if start_items != 0:
                fail("start_items_not_fresh", start_items=start_items)
            if start_secrets != 0:
                fail("start_secrets_not_fresh", start_secrets=start_secrets)
            if required_start_episode is not None and _int_field(
                episode,
                "start_episode",
            ) != int(required_start_episode):
                fail(
                    "start_episode_mismatch",
                    start_episode=_int_field(episode, "start_episode"),
                    required_start_episode=int(required_start_episode),
                )
            if required_start_map is not None and _int_field(
                episode,
                "start_map",
            ) != int(required_start_map):
                fail(
                    "start_map_mismatch",
                    start_map=_int_field(episode, "start_map"),
                    required_start_map=int(required_start_map),
                )
            if _int_field(episode, "first_visible_contacts") < int(
                min_first_visible_contacts
            ):
                fail(
                    "first_visible_contacts_below_threshold",
                    first_visible_contacts=_int_field(episode, "first_visible_contacts"),
                    minimum=int(min_first_visible_contacts),
                )
            if _int_field(episode, "first_shootable_contacts") < int(
                min_first_shootable_contacts
            ):
                fail(
                    "first_shootable_contacts_below_threshold",
                    first_shootable_contacts=_int_field(
                        episode,
                        "first_shootable_contacts",
                    ),
                    minimum=int(min_first_shootable_contacts),
                )
            if max_kill_gain < int(min_kill_gain):
                fail(
                    "kill_gain_below_threshold",
                    max_kill_gain=max_kill_gain,
                    minimum=int(min_kill_gain),
                )
            if _int_field(episode, "route_attempt_steps") < int(min_route_attempt_steps):
                fail(
                    "route_attempt_steps_below_threshold",
                    route_attempt_steps=_int_field(episode, "route_attempt_steps"),
                    minimum=int(min_route_attempt_steps),
                )
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
            if _int_field(episode, "allowed_skill_filter_fallback_steps") != 0:
                fail(
                    "allowed_skill_filter_fallback_steps_present",
                    allowed_skill_filter_fallback_steps=_int_field(
                        episode,
                        "allowed_skill_filter_fallback_steps",
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
            for skill, count in episode_skill_counts.items():
                if skill not in allowed:
                    disallowed_skill_counts[skill] += count
            disallowed = {
                skill: count
                for skill, count in episode_skill_counts.items()
                if skill not in allowed
            }
            if disallowed:
                fail(
                    "disallowed_skill_executed",
                    disallowed_skill_counts=dict(sorted(disallowed.items())),
                )

            if episode_key in failed_episode_keys:
                bottleneck_counts[
                    _classify_bottleneck(
                        episode,
                        min_first_visible_contacts=min_first_visible_contacts,
                        min_first_shootable_contacts=min_first_shootable_contacts,
                        min_kill_gain=min_kill_gain,
                        min_route_attempt_steps=min_route_attempt_steps,
                        min_exit_route_attempt_steps=min_exit_route_attempt_steps,
                    )
                ] += 1

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
    if (
        int(summary["passed_episodes"]) < int(min_level_completions)
        and not _has_blocking_failures(failures)
    ):
        failures.append(
            {
                "stage": stage_name,
                "episode_index": None,
                "reason": "passed_episode_count_below_threshold",
                "passed_episodes": summary["passed_episodes"],
                "level_complete_episode_count": summary["level_complete_episode_count"],
                "minimum": int(min_level_completions),
            }
        )

    blocking_failure_count = sum(
        1 for failure in failures if _is_blocking_failure(failure)
    )
    summary["blocking_failure_count"] = blocking_failure_count
    for field in ("route_progress_units", "exit_route_progress_units"):
        summary[field] = round(float(summary[field]), 4)
    summary["reset_source_counts"] = dict(sorted(reset_source_counts.items()))
    summary["done_reasons"] = dict(sorted(done_reasons.items()))
    summary["skill_counts"] = dict(sorted(skill_counts.items()))
    summary["disallowed_skill_counts"] = dict(sorted(disallowed_skill_counts.items()))
    summary["configured_allowed_skills"] = sorted(configured_allowed)
    summary["bottleneck_counts"] = dict(sorted(bottleneck_counts.items()))

    return {
        "schema": TRUE_SPAWN_E2E_GATE_SCHEMA,
        "created_at_epoch_seconds": int(time.time()),
        "ok": blocking_failure_count == 0,
        "config": {
            "stage_name": stage_name,
            "allowed_skills": list(allowed),
            "min_episodes": int(min_episodes),
            "min_level_completions": int(min_level_completions),
            "min_first_visible_contacts": int(min_first_visible_contacts),
            "min_first_shootable_contacts": int(min_first_shootable_contacts),
            "min_kill_gain": int(min_kill_gain),
            "min_route_attempt_steps": int(min_route_attempt_steps),
            "min_exit_route_attempt_steps": int(min_exit_route_attempt_steps),
            "require_level_complete": bool(require_level_complete),
            "require_strict_skill_filter": bool(require_strict_skill_filter),
            "required_start_episode": required_start_episode,
            "required_start_map": required_start_map,
        },
        "summary": summary,
        "failures": failures,
    }


def _collect_aggregate_reset_failures(
    eval_payload: Any,
    *,
    failures: list[dict[str, Any]],
    stage_label: str,
) -> None:
    if not isinstance(eval_payload, dict):
        return
    result = eval_payload.get("result", {})
    if not isinstance(result, dict):
        return
    reset_source_breakdown = result.get("reset_source_breakdown", {})
    if not isinstance(reset_source_breakdown, dict):
        return
    non_episode = {
        str(source): data
        for source, data in reset_source_breakdown.items()
        if str(source) != "episode"
    }
    if non_episode:
        failures.append(
            {
                "stage": stage_label,
                "episode_index": None,
                "reason": "non_episode_reset_source_in_aggregate",
                "reset_source_breakdown": non_episode,
            }
        )


def _has_blocking_failures(failures: list[dict[str, Any]]) -> bool:
    return any(_is_blocking_failure(failure) for failure in failures)


def _is_blocking_failure(failure: dict[str, Any]) -> bool:
    return str(failure.get("reason") or "") not in CHAIN_FAILURE_REASONS


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


def _classify_bottleneck(
    episode: dict[str, Any],
    *,
    min_first_visible_contacts: int,
    min_first_shootable_contacts: int,
    min_kill_gain: int,
    min_route_attempt_steps: int,
    min_exit_route_attempt_steps: int,
) -> str:
    if str(episode.get("reset_source") or "") != "episode":
        return "gate_integrity"
    if (
        _int_field(episode, "invalid_action_steps") != 0
        or _int_field(episode, "selected_disallowed_steps") != 0
        or _int_field(episode, "action_mask_fallback_steps") != 0
        or _int_field(episode, "strict_allowed_skill_fallback_steps") != 0
        or _int_field(episode, "snapshot_verification_failures") != 0
    ):
        return "gate_integrity"
    if _int_field(episode, "first_visible_contacts") < int(min_first_visible_contacts):
        if _int_field(episode, "route_attempt_steps") <= 0:
            return "spawn_route"
        return "first_contact"
    if _int_field(episode, "first_shootable_contacts") < int(min_first_shootable_contacts):
        return "first_contact"
    if _episode_kill_gain(episode) < int(min_kill_gain):
        return "combat"
    if _int_field(episode, "route_attempt_steps") < int(min_route_attempt_steps):
        return "post_combat_route"
    if _int_field(episode, "exit_route_attempt_steps") < int(min_exit_route_attempt_steps):
        return "post_combat_route"
    if _int_field(episode, "level_transition_delta") != 1 or not _episode_level_completed(
        episode
    ):
        return "final_line"
    return "complete"


def _episode_level_completed(episode: dict[str, Any]) -> bool:
    return bool(episode.get("level_completed")) or episode.get("done_reason") == "level_complete"


def _episode_kill_gain(episode: dict[str, Any]) -> int:
    absolute_gain = max(0, _int_field(episode, "max_kills") - _int_field(episode, "start_kills"))
    return max(_int_field(episode, "kill_delta"), _int_field(episode, "max_kill_gain"), absolute_gain)


def _skill_counts(episode: dict[str, Any]) -> dict[str, int]:
    counts = episode.get("skill_counts", {})
    if not isinstance(counts, dict):
        return {}
    return {
        str(skill): max(0, _int_value(count))
        for skill, count in counts.items()
        if str(skill)
    }


def _update_min_max(summary: dict[str, Any], field: str, value: int) -> None:
    min_key = f"min_{field}"
    max_key = f"max_{field}"
    current_min = summary[min_key]
    current_max = summary[max_key]
    summary[min_key] = value if current_min is None else min(int(current_min), value)
    summary[max_key] = value if current_max is None else max(int(current_max), value)


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
        help="Executed skill allowed by the gate. Defaults to all PPO skills.",
    )
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--min-level-completions", type=int, default=1)
    parser.add_argument("--min-first-visible-contacts", type=int, default=1)
    parser.add_argument("--min-first-shootable-contacts", type=int, default=1)
    parser.add_argument("--min-kill-gain", type=int, default=DEFAULT_MIN_KILL_GAIN)
    parser.add_argument("--min-route-attempt-steps", type=int, default=1)
    parser.add_argument("--min-exit-route-attempt-steps", type=int, default=1)
    parser.add_argument("--required-start-episode", type=int)
    parser.add_argument("--required-start-map", type=int)
    parser.add_argument("--allow-missing-level-complete", action="store_true")
    parser.add_argument("--allow-nonstrict-skill-filter", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    report = validate_true_spawn_e2e_gate(
        payload,
        stage_name=args.stage_name,
        allowed_skills=tuple(args.allowed_skill or DEFAULT_ALLOWED_SKILLS),
        min_episodes=args.min_episodes,
        min_level_completions=args.min_level_completions,
        min_first_visible_contacts=args.min_first_visible_contacts,
        min_first_shootable_contacts=args.min_first_shootable_contacts,
        min_kill_gain=args.min_kill_gain,
        min_route_attempt_steps=args.min_route_attempt_steps,
        min_exit_route_attempt_steps=args.min_exit_route_attempt_steps,
        require_level_complete=not args.allow_missing_level_complete,
        require_strict_skill_filter=not args.allow_nonstrict_skill_filter,
        required_start_episode=args.required_start_episode,
        required_start_map=args.required_start_map,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
