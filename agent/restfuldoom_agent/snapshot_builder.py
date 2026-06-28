"""Build snapshot-backed PPO curriculum manifests from trajectory evidence."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .snapshot_curriculum import (
    SNAPSHOT_CURRICULUM_SCHEMA,
    _sha256_path,
    validate_snapshot_curriculum,
)

AUTO_SELECTORS = frozenset(
    {
        "first-visible",
        "first-shootable",
        "first-enemy-shootable",
        "first-damage",
        "first-kill",
        "post-combat",
        "post-combat-exit-route",
        "level-transition",
    }
)
POST_COMBAT_KILL_THRESHOLD = 5


def build_snapshot_curriculum_from_trajectory(
    trajectory_path: str | Path,
    *,
    output_path: str | Path | None = None,
    name: str = "e1m1-progressed-bottlenecks",
    indexes: list[int] | None = None,
    auto_selectors: list[str] | None = None,
    snapshot_dir: str | Path = "snapshots",
    save_slot_base: int | None = None,
    capsule: str = "agent-doom",
    microvm_id: str | None = None,
    capture_command: str | None = None,
    capture_cwd: str | Path | None = None,
    capture_timeout_seconds: float = 60.0,
    require_capture_artifacts: bool = False,
    post_combat_kills: int = POST_COMBAT_KILL_THRESHOLD,
) -> dict[str, Any]:
    """Build a versioned snapshot curriculum from selected trajectory rows."""
    if post_combat_kills < 0:
        raise ValueError("post_combat_kills must be non-negative")
    trajectory = Path(trajectory_path)
    records = _read_jsonl_records(trajectory)
    if not records:
        raise ValueError(f"trajectory has no JSON records: {trajectory}")

    selected = _select_records(
        records,
        indexes=indexes or [],
        auto_selectors=auto_selectors or [],
        post_combat_kills=post_combat_kills,
    )
    if not selected:
        raise ValueError("no snapshot stages selected")

    output = Path(output_path) if output_path is not None else None
    snapshot_directory = Path(snapshot_dir)
    stages: list[dict[str, Any]] = []
    for order, selection in enumerate(selected):
        line_index = selection["line_index"]
        record = selection["record"]
        selectors = selection["selectors"]
        selector = _primary_selector(selectors)
        stage = _stage_from_record(
            record,
            line_index=line_index,
            order=order,
            selector=selector,
            selectors=selectors,
            trajectory=trajectory,
            name=name,
            snapshot_dir=snapshot_directory,
            save_slot=(save_slot_base + order) if save_slot_base is not None else None,
            capsule=capsule,
            microvm_id=microvm_id,
        )
        if capture_command:
            capture = _run_capture_command(
                capture_command,
                stage,
                trajectory=trajectory,
                cwd=Path(capture_cwd) if capture_cwd is not None else None,
                timeout_seconds=capture_timeout_seconds,
            )
            stage["capture"] = capture
            if capture["returncode"] != 0:
                raise RuntimeError(
                    "snapshot capture command failed for "
                    f"{stage['name']} with exit code {capture['returncode']}"
                )

        artifact_path = _resolve_artifact_path(
            stage["snapshot"]["path"],
            output_path=output,
            capture_cwd=Path(capture_cwd) if capture_cwd is not None else None,
        )
        if artifact_path.exists():
            stage["snapshot"]["digest"] = f"sha256:{_sha256_path(artifact_path)}"
            stage["validated"] = True
        elif require_capture_artifacts:
            raise FileNotFoundError(
                f"snapshot artifact for {stage['name']} was not created: "
                f"{stage['snapshot']['path']}"
            )
        stages.append(stage)

    manifest = {
        "schema": SNAPSHOT_CURRICULUM_SCHEMA,
        "name": name,
        "source": {
            "schema": "restfuldoom.snapshot_curriculum_source.v1",
            "trajectory_jsonl": str(trajectory),
            "selection": {
                "indexes": indexes or [],
                "auto": auto_selectors or [],
                "post_combat_kills": int(post_combat_kills),
            },
            "generated_at_epoch_seconds": int(time.time()),
        },
        "stages": stages,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_index + 1}: {error}") from error
            if isinstance(record, dict):
                records.append({"line_index": line_index, "record": record})
    return records


def _select_records(
    records: list[dict[str, Any]],
    *,
    indexes: list[int],
    auto_selectors: list[str],
    post_combat_kills: int = POST_COMBAT_KILL_THRESHOLD,
) -> list[dict[str, Any]]:
    by_line = {int(entry["line_index"]): entry for entry in records}
    selected: dict[int, dict[str, Any]] = {}

    for index in indexes:
        if index not in by_line:
            raise ValueError(f"trajectory row {index} was not found")
        selected[index] = {
            **by_line[index],
            "selectors": ["explicit"],
        }

    for selector in auto_selectors:
        if selector not in AUTO_SELECTORS:
            choices = ", ".join(sorted(AUTO_SELECTORS))
            raise ValueError(f"unknown auto selector {selector!r}; choose one of: {choices}")
        entry = _first_matching_record(
            records,
            selector,
            post_combat_kills=post_combat_kills,
        )
        if entry is None:
            raise ValueError(f"auto selector {selector!r} did not match any trajectory row")
        line_index = int(entry["line_index"])
        if line_index in selected:
            selected[line_index]["selectors"].append(selector)
        else:
            selected[line_index] = {
                **entry,
                "selectors": [selector],
            }

    return [selected[index] for index in sorted(selected)]


def _primary_selector(selectors: list[str]) -> str:
    if "post-combat-exit-route" in selectors:
        return "post-combat-exit-route"
    return selectors[0] if selectors else "explicit"


def _first_matching_record(
    records: list[dict[str, Any]],
    selector: str,
    *,
    post_combat_kills: int = POST_COMBAT_KILL_THRESHOLD,
) -> dict[str, Any] | None:
    start_episode_map = _episode_map(_record_state(records[0]["record"]))
    previous_kills = _int_or_none(_record_state(records[0]["record"]).get("kills"))
    for entry in records:
        record = entry["record"]
        state = _record_state(record)
        if selector == "first-visible" and _has_visible_enemy(record):
            return entry
        if selector == "first-shootable" and _has_shootable_target(record):
            return entry
        if selector == "first-enemy-shootable" and _has_enemy_shootable_target(record):
            return entry
        if selector == "first-damage" and _damage_delta(record) > 0:
            return entry
        if selector == "first-kill":
            kills = _int_or_none(state.get("kills"))
            if _kill_delta(record) > 0 or (
                previous_kills is not None and kills is not None and kills > previous_kills
            ):
                return entry
            if kills is not None:
                previous_kills = kills
        if selector == "post-combat" and _is_post_combat(
            record,
            min_kills=post_combat_kills,
        ):
            return entry
        if selector == "post-combat-exit-route" and _is_post_combat_exit_route(
            record,
            min_kills=post_combat_kills,
        ):
            return entry
        if selector == "level-transition" and _episode_map(state) != start_episode_map:
            return entry
    return None


def _stage_from_record(
    record: dict[str, Any],
    *,
    line_index: int,
    order: int,
    selector: str,
    selectors: list[str],
    trajectory: Path,
    name: str,
    snapshot_dir: Path,
    save_slot: int | None,
    capsule: str,
    microvm_id: str | None,
) -> dict[str, Any]:
    state = _record_state(record)
    stage_slug = _slug(f"{line_index:04d}-{selector}")
    snapshot_id = f"{_slug(name)}-{stage_slug}"
    snapshot_path = snapshot_dir / f"{snapshot_id}.snap"
    snapshot: dict[str, Any] = {
        "id": snapshot_id,
        "path": str(snapshot_path),
        "digest": "sha256:<fill-after-capture>",
        "capsule": capsule,
    }
    if microvm_id:
        snapshot["microvm_id"] = microvm_id
    if save_slot is not None:
        snapshot["slot"] = int(save_slot)
        snapshot["ref"] = f"save_slot:{int(save_slot)}"

    return {
        "index": order,
        "name": f"{stage_slug}_snapshot",
        "note": _stage_note(selector),
        "validated": False,
        "reset_start": _reset_start_from_state(state),
        "snapshot": snapshot,
        "expected_state": _expected_state(record),
        "evidence": {
            "schema": "restfuldoom.snapshot_stage_evidence.v1",
            "trajectory_jsonl": str(trajectory),
            "source_record_index": line_index,
            "selector": selector,
            "selectors": selectors,
            "skill": _policy_skill(record),
            "reward": _compact_record_value(record.get("reward")),
            "transition": _compact_record_value(_record_info(record).get("transition")),
        },
    }


def _run_capture_command(
    template: str,
    stage: dict[str, Any],
    *,
    trajectory: Path,
    cwd: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = _render_capture_command(template, stage, trajectory=trajectory)
    started = time.time()
    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd is not None else None,
        timeout=timeout_seconds,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "schema": "restfuldoom.snapshot_capture.v1",
        "command": _redact_command(command),
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 4),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _render_capture_command(
    template: str,
    stage: dict[str, Any],
    *,
    trajectory: Path,
) -> str:
    snapshot = stage.get("snapshot", {})
    expected = stage.get("expected_state", {})
    values = {
        "stage_name": stage.get("name", ""),
        "stage_name_sh": shlex.quote(str(stage.get("name", ""))),
        "snapshot_id": snapshot.get("id", ""),
        "snapshot_id_sh": shlex.quote(str(snapshot.get("id", ""))),
        "snapshot_path": snapshot.get("path", ""),
        "snapshot_path_sh": shlex.quote(str(snapshot.get("path", ""))),
        "snapshot_path_py": repr(str(snapshot.get("path", ""))),
        "snapshot_ref": snapshot.get("ref", ""),
        "snapshot_ref_sh": shlex.quote(str(snapshot.get("ref", ""))),
        "microvm_id": snapshot.get("microvm_id", ""),
        "microvm_id_sh": shlex.quote(str(snapshot.get("microvm_id", ""))),
        "capsule": snapshot.get("capsule", ""),
        "capsule_sh": shlex.quote(str(snapshot.get("capsule", ""))),
        "record_index": stage.get("evidence", {}).get("source_record_index", ""),
        "tick": expected.get("tick", ""),
        "trajectory": str(trajectory),
        "trajectory_sh": shlex.quote(str(trajectory)),
    }
    return template.format(**values)


def _redact_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return command.replace("x-aws-proxy-auth", "x-aws-proxy-auth=<redacted>")
    redacted: list[str] = []
    redact_next = False
    secret_flags = {"--token", "--auth-token", "--authorization", "--x-aws-proxy-auth"}
    for part in parts:
        lower = part.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if lower in secret_flags:
            redacted.append(part)
            redact_next = True
            continue
        if any(lower.startswith(f"{flag}=") for flag in secret_flags):
            flag, _, _value = part.partition("=")
            redacted.append(f"{flag}=<redacted>")
            continue
        redacted.append(part)
    return " ".join(shlex.quote(part) for part in redacted)


def _resolve_artifact_path(
    raw_path: str,
    *,
    output_path: Path | None,
    capture_cwd: Path | None,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if capture_cwd is not None and (capture_cwd / path).exists():
        return capture_cwd / path
    if path.exists():
        return path
    if output_path is not None and (output_path.parent / path).exists():
        return output_path.parent / path
    return path


def _record_state(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("state")
    if isinstance(state, dict):
        return state
    info = _record_info(record)
    state = info.get("state")
    return state if isinstance(state, dict) else {}


def _record_info(record: dict[str, Any]) -> dict[str, Any]:
    info = record.get("info")
    return info if isinstance(info, dict) else {}


def _policy_decision(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        decision = metadata.get("policy_decision")
        if isinstance(decision, dict):
            return decision
    return {}


def _policy_skill(record: dict[str, Any]) -> str | None:
    decision = _policy_decision(record)
    skill = decision.get("skill")
    if isinstance(skill, str):
        return skill
    info_skill = _record_info(record).get("skill")
    return info_skill if isinstance(info_skill, str) else None


def _expected_state(record: dict[str, Any]) -> dict[str, Any]:
    state = _record_state(record)
    expected: dict[str, Any] = {}
    for key in (
        "tick",
        "level_time",
        "episode",
        "map",
        "health",
        "armor",
        "kills",
        "items",
        "secrets",
    ):
        if key in state:
            expected[key] = state[key]
    if "position_fp" in state:
        expected["position_fp"] = state["position_fp"]
    expected["visible_enemy"] = _has_visible_enemy(record)
    expected["shootable_target"] = _has_shootable_target(record)
    combat = state.get("combat")
    if isinstance(combat, dict) and "target_is_enemy" in combat:
        expected["target_is_enemy"] = bool(combat.get("target_is_enemy"))
    route = _route_waypoint(record)
    if route:
        expected["route_waypoint_exit"] = bool(route.get("exit"))
        line = route.get("line")
        if isinstance(line, dict) and _int_or_none(line.get("line_id")) is not None:
            expected["route_waypoint_line_id"] = int(line["line_id"])
    expected["damage_delta"] = _damage_delta(record)
    expected["kill_delta"] = _kill_delta(record)
    done_reason = _record_info(record).get("done_reason")
    if isinstance(done_reason, str) and done_reason:
        expected["done_reason"] = done_reason
    return expected


def _reset_start_from_state(state: dict[str, Any]) -> dict[str, Any]:
    position = state.get("position_fp")
    if not isinstance(position, list) or len(position) < 2:
        return {}
    reset_start = {
        "x_fp": int(position[0]),
        "y_fp": int(position[1]),
    }
    for key in ("health", "armor", "ammo_bullets"):
        if key in state:
            reset_start[key] = int(state[key])
    return reset_start


def _has_visible_enemy(record: dict[str, Any]) -> bool:
    info = _record_info(record)
    if bool(info.get("had_visible_enemy")):
        return True
    decision = _policy_decision(record)
    visible = decision.get("visible_enemies")
    if _int_or_none(visible) and int(visible) > 0:
        return True
    enemies = decision.get("visible_enemy")
    if isinstance(enemies, list) and enemies:
        return True
    return False


def _has_shootable_target(record: dict[str, Any]) -> bool:
    info = _record_info(record)
    if bool(info.get("had_shootable_target")):
        return True
    state = _record_state(record)
    combat = state.get("combat")
    if isinstance(combat, dict) and combat.get("has_shootable_target"):
        return True
    decision_combat = _policy_decision(record).get("combat")
    return bool(
        isinstance(decision_combat, dict)
        and decision_combat.get("has_shootable_target")
    )


def _has_enemy_shootable_target(record: dict[str, Any]) -> bool:
    if not _has_shootable_target(record):
        return False
    state = _record_state(record)
    combat = state.get("combat")
    if isinstance(combat, dict) and "target_is_enemy" in combat:
        return bool(combat.get("target_is_enemy"))
    decision_combat = _policy_decision(record).get("combat")
    if isinstance(decision_combat, dict) and "target_is_enemy" in decision_combat:
        return bool(decision_combat.get("target_is_enemy"))
    return False


def _is_post_combat(
    record: dict[str, Any],
    *,
    min_kills: int = POST_COMBAT_KILL_THRESHOLD,
) -> bool:
    kills = _int_or_none(_record_state(record).get("kills"))
    return (
        kills is not None
        and kills >= int(min_kills)
        and not _has_visible_enemy(record)
        and not _has_shootable_target(record)
    )


def _is_post_combat_exit_route(
    record: dict[str, Any],
    *,
    min_kills: int = POST_COMBAT_KILL_THRESHOLD,
) -> bool:
    if not _is_post_combat(record, min_kills=min_kills):
        return False
    route = _route_waypoint(record)
    line = route.get("line") if isinstance(route, dict) else None
    line_id = _int_or_none(line.get("line_id")) if isinstance(line, dict) else None
    return bool(route and route.get("exit") is True and line_id is not None and line_id > 0)


def _route_waypoint(record: dict[str, Any]) -> dict[str, Any]:
    for navigation in _navigation_dicts(record):
        route = navigation.get("route_waypoint")
        if isinstance(route, dict):
            return route
    return {}


def _navigation_dicts(record: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    state = _record_state(record)
    navigation = state.get("navigation")
    if isinstance(navigation, dict):
        out.append(navigation)
    decision_navigation = _policy_decision(record).get("navigation")
    if isinstance(decision_navigation, dict):
        out.append(decision_navigation)
    return out


def _damage_delta(record: dict[str, Any]) -> int:
    info_transition = _record_info(record).get("transition")
    if isinstance(info_transition, dict):
        value = _int_or_none(info_transition.get("damage_delta"))
        if value:
            return value
    reward = record.get("reward")
    if isinstance(reward, dict):
        value = _int_or_none(reward.get("damage_delta"))
        if value:
            return value
    return 0


def _kill_delta(record: dict[str, Any]) -> int:
    info_transition = _record_info(record).get("transition")
    if isinstance(info_transition, dict):
        value = _int_or_none(info_transition.get("kill_delta"))
        if value:
            return value
    reward = record.get("reward")
    if isinstance(reward, dict):
        value = _int_or_none(reward.get("kill_delta"))
        if value:
            return value
    return 0


def _episode_map(state: dict[str, Any]) -> tuple[int | None, int | None]:
    return _int_or_none(state.get("episode")), _int_or_none(state.get("map"))


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_record_value(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {
        key: item
        for key, item in value.items()
        if key in {"reward", "done", "kill_delta", "damage_delta", "enemy_distance_delta"}
    }


def _stage_note(selector: str) -> str:
    notes = {
        "explicit": "Restore an exact progressed state selected from a trajectory row.",
        "first-visible": "Restore the first trajectory state with visible enemy contact.",
        "first-shootable": "Restore the first trajectory state with any shootable target.",
        "first-enemy-shootable": "Restore the first trajectory state with a shootable enemy target.",
        "first-damage": "Restore the first trajectory state where the agent dealt damage.",
        "first-kill": "Restore the first trajectory state where the agent scored a kill.",
        "post-combat": "Restore the first trajectory state after the post-combat kill threshold.",
        "post-combat-exit-route": (
            "Restore the first post-combat state with an exit route waypoint."
        ),
        "level-transition": "Restore the trajectory state around level transition.",
    }
    return notes.get(selector, notes["explicit"])


def _slug(value: str) -> str:
    out = []
    last_dash = False
    for character in value.lower():
        if character.isalnum():
            out.append(character)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "snapshot"


def _parse_indexes(raw: str | None) -> list[int]:
    if not raw:
        return []
    indexes = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            indexes.append(int(part))
    return indexes


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="e1m1-progressed-bottlenecks")
    parser.add_argument("--indexes", help="comma-separated zero-based JSONL row indexes")
    parser.add_argument(
        "--auto",
        action="append",
        default=[],
        choices=sorted(AUTO_SELECTORS),
        help="auto-select the first row matching this milestone; may be repeated",
    )
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument(
        "--save-slot-base",
        type=int,
        help="optional first Doom agent save slot assigned to generated stages",
    )
    parser.add_argument(
        "--post-combat-kills",
        type=int,
        default=POST_COMBAT_KILL_THRESHOLD,
        help="minimum absolute kill count for post-combat auto selectors",
    )
    parser.add_argument("--capsule", default="agent-doom")
    parser.add_argument("--microvm-id")
    parser.add_argument(
        "--capture-command",
        help=(
            "optional shell command template run per stage; placeholders include "
            "{snapshot_id_sh}, {snapshot_path_sh}, {stage_name_sh}, {record_index}, and {tick}"
        ),
    )
    parser.add_argument("--capture-cwd", type=Path)
    parser.add_argument("--capture-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--require-capture-artifacts", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    indexes = _parse_indexes(args.indexes)
    if not indexes and not args.auto:
        parser.error("choose at least one --indexes value or --auto selector")

    build_snapshot_curriculum_from_trajectory(
        args.trajectory,
        output_path=args.output,
        name=args.name,
        indexes=indexes,
        auto_selectors=args.auto,
        snapshot_dir=args.snapshot_dir,
        save_slot_base=args.save_slot_base,
        capsule=args.capsule,
        microvm_id=args.microvm_id,
        capture_command=args.capture_command,
        capture_cwd=args.capture_cwd,
        capture_timeout_seconds=args.capture_timeout_seconds,
        require_capture_artifacts=args.require_capture_artifacts,
        post_combat_kills=args.post_combat_kills,
    )
    if args.validate:
        report = validate_snapshot_curriculum(
            args.output,
            require_artifacts=args.require_capture_artifacts,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
