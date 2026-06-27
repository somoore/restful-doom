"""Capture native Doom agent save slots during a structured-brain rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brain import AgentMemory, BrainPolicy
from .client import (
    BackoffConfig,
    DoomAgentClient,
    RolloutStep,
    summarize_state,
)
from .env import _verify_snapshot_restored_state
from .reward import RewardEngine, goal_preset
from .rollout_config import safe_endpoint_host
from .skill_policy import SkillPolicyModel
from .snapshot_builder import (
    AUTO_SELECTORS,
    _damage_delta,
    _episode_map,
    _has_shootable_target,
    _has_visible_enemy,
    _int_or_none,
    _kill_delta,
    _record_state,
    _stage_from_record,
)
from .snapshot_curriculum import SNAPSHOT_CURRICULUM_SCHEMA, validate_snapshot_curriculum

SNAPSHOT_CAPTURE_SOURCE_SCHEMA = "restfuldoom.snapshot_capture_source.v1"
NATIVE_SNAPSHOT_CAPTURE_SCHEMA = "restfuldoom.native_snapshot_capture.v1"
SNAPSHOT_LOAD_VERIFICATION_SCHEMA = "restfuldoom.snapshot_load_verification.v1"


@dataclass(frozen=True)
class SnapshotCaptureConfig:
    """Runtime configuration for native snapshot curriculum capture."""

    endpoint: str = "127.0.0.1:50051"
    token: str | None = None
    agent_port: int = 50051
    tls: bool = False
    authority: str | None = None
    memory_path: Path = Path("agent_memory/e1m1.json")
    skill_model_path: Path | None = None
    trajectory_jsonl: Path | None = Path("trajectories/snapshot-capture.jsonl")
    output_path: Path = Path("trajectories/snapshot-curriculum.json")
    name: str = "e1m1-progressed-bottlenecks"
    snapshot_dir: Path = Path("snapshots")
    auto_selectors: tuple[str, ...] = (
        "first-visible",
        "first-shootable",
        "first-damage",
    )
    save_slot_base: int = 0
    capsule: str = "agent-doom"
    microvm_id: str | None = None
    goal_preset: str = "combat"
    mission: str = "capture progressed-map bottleneck snapshots"
    max_states: int = 12000
    policy_id: str = "snapshot_capture_brain"
    reconnect: bool = True
    max_reconnects: int = 5
    stop_after_captured: bool = True
    settle_states_after_capture: int = 2
    verify_loads: bool = False
    verify_timeout_seconds: float = 4.0


class SnapshotMilestoneTracker:
    """Tracks first occurrence of requested milestone selectors."""

    def __init__(self, selectors: tuple[str, ...]) -> None:
        unknown = sorted(set(selectors) - AUTO_SELECTORS)
        if unknown:
            choices = ", ".join(sorted(AUTO_SELECTORS))
            raise ValueError(
                f"unknown snapshot selector(s): {', '.join(unknown)}; choose from {choices}"
            )
        self.selectors = tuple(dict.fromkeys(selectors))
        self.captured: set[str] = set()
        self._start_episode_map: tuple[int | None, int | None] | None = None
        self._previous_kills: int | None = None

    @property
    def complete(self) -> bool:
        """Returns true once every requested selector has been captured."""
        return all(selector in self.captured for selector in self.selectors)

    def observe(self, record: dict[str, Any]) -> list[str]:
        """Returns newly matched selectors for one rollout record."""
        state = _record_state(record)
        if self._start_episode_map is None:
            self._start_episode_map = _episode_map(state)
        if self._previous_kills is None:
            self._previous_kills = _int_or_none(state.get("kills"))

        matches: list[str] = []
        for selector in self.selectors:
            if selector in self.captured:
                continue
            if selector == "first-visible" and _has_visible_enemy(record):
                matches.append(selector)
            elif selector == "first-shootable" and _has_shootable_target(record):
                matches.append(selector)
            elif selector == "first-damage" and _damage_delta(record) > 0:
                matches.append(selector)
            elif selector == "first-kill" and self._matches_first_kill(record):
                matches.append(selector)
            elif (
                selector == "level-transition"
                and _episode_map(state) != self._start_episode_map
            ):
                matches.append(selector)

        kills = _int_or_none(state.get("kills"))
        if kills is not None:
            self._previous_kills = kills
        return matches

    def mark_captured(self, selectors: list[str]) -> None:
        """Marks selectors as captured after their save slot is queued."""
        self.captured.update(selectors)

    def _matches_first_kill(self, record: dict[str, Any]) -> bool:
        if _kill_delta(record) > 0:
            return True
        kills = _int_or_none(_record_state(record).get("kills"))
        return (
            self._previous_kills is not None
            and kills is not None
            and kills > self._previous_kills
        )


async def capture_snapshot_curriculum(config: SnapshotCaptureConfig) -> dict[str, Any]:
    """Runs the structured brain and captures native save slots at milestones."""
    memory = AgentMemory.load(config.memory_path)
    params = memory.best_params()
    skill_model_path = _resolve_skill_model_path(config.skill_model_path, memory)
    skill_model = SkillPolicyModel.load(skill_model_path) if skill_model_path else None
    policy = BrainPolicy(
        memory=memory,
        params=params,
        policy_id=config.policy_id,
        skill_model=skill_model,
    )
    reward = RewardEngine(goal_preset(config.goal_preset))
    client = DoomAgentClient(
        config.endpoint,
        token=config.token,
        agent_port=config.agent_port,
        tls=config.tls,
        authority=config.authority,
    )
    tracker = SnapshotMilestoneTracker(config.auto_selectors)
    run_id = f"snapshot-capture-{uuid.uuid4().hex[:12]}"
    trajectory = config.trajectory_jsonl or Path("<stream>")
    stages: list[dict[str, Any]] = []
    records_seen = 0
    last_capture_index: int | None = None

    metadata = {
        "source": "snapshot-capture",
        "run_id": run_id,
        "policy_id": config.policy_id,
        "goal_preset": config.goal_preset,
        "mission": config.mission,
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "memory_path": str(config.memory_path),
        "selectors": list(config.auto_selectors),
        "save_slot_base": config.save_slot_base,
    }
    if skill_model_path is not None:
        metadata["skill_model_path"] = str(skill_model_path)

    async def capture_before_action_send(step: RolloutStep) -> None:
        nonlocal last_capture_index
        record = _record_from_rollout_step(step)
        selectors = tracker.observe(record)
        if not selectors:
            return
        stage = await _capture_stage(
            client,
            config,
            record,
            selectors=selectors,
            line_index=step.index,
            order=len(stages),
            trajectory=trajectory,
            run_id=run_id,
        )
        stages.append(stage)
        tracker.mark_captured(selectors)
        last_capture_index = step.index

    try:
        async for step in client.stream_rollout(
            policy,
            reward_engine=reward,
            max_states=config.max_states,
            trajectory_jsonl=config.trajectory_jsonl,
            reconnect=config.reconnect,
            backoff=BackoffConfig(max_attempts=config.max_reconnects),
            on_step_before_action_send=capture_before_action_send,
            rollout_metadata=metadata,
        ):
            records_seen += 1
            if (
                config.stop_after_captured
                and tracker.complete
                and last_capture_index is not None
                and step.index - last_capture_index >= config.settle_states_after_capture
            ):
                break

        if not stages:
            raise RuntimeError(
                "snapshot capture finished without matching any requested milestones"
            )

        if config.verify_loads:
            await _verify_captured_slots(client, stages, timeout_seconds=config.verify_timeout_seconds)
    finally:
        await client.close()

    manifest = _build_manifest(
        config,
        run_id=run_id,
        stages=stages,
        records_seen=records_seen,
        skill_model_path=skill_model_path,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_snapshot_curriculum(config.output_path, require_artifacts=False)
    manifest["validation"] = validation
    config.output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


async def _capture_stage(
    client: DoomAgentClient,
    config: SnapshotCaptureConfig,
    record: dict[str, Any],
    *,
    selectors: list[str],
    line_index: int,
    order: int,
    trajectory: Path,
    run_id: str,
) -> dict[str, Any]:
    selector = selectors[0]
    slot = config.save_slot_base + order
    description = f"{selector}-{line_index}"
    response = await client.save_snapshot(
        slot=slot,
        description=description,
        run_id=f"{run_id}-slot-{slot}",
    )
    if not response.accepted or not response.save_queued:
        raise RuntimeError(
            f"snapshot save rejected for slot {slot}: {response.message}"
        )
    stage = _stage_from_record(
        record,
        line_index=line_index,
        order=order,
        selector=selector,
        selectors=selectors,
        trajectory=trajectory,
        name=config.name,
        snapshot_dir=config.snapshot_dir,
        save_slot=slot,
        capsule=config.capsule,
        microvm_id=config.microvm_id,
    )
    stage["validated"] = True
    stage["capture"] = {
        "schema": NATIVE_SNAPSHOT_CAPTURE_SCHEMA,
        "method": "grpc_save_snapshot",
        "slot": response.slot,
        "accepted": response.accepted,
        "save_queued": response.save_queued,
        "load_queued": response.load_queued,
        "message": response.message,
        "description": description,
        "captured_at_epoch_seconds": int(time.time()),
    }
    return stage


def _record_from_rollout_step(step: RolloutStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "state": step.state_summary,
        "reward": step.reward_summary,
        "next_action": step.action_summary,
        "last_seen_tick": step.last_seen_tick,
        "reconnect_attempts": step.reconnect_attempts,
        "metadata": step.metadata,
    }


def _build_manifest(
    config: SnapshotCaptureConfig,
    *,
    run_id: str,
    stages: list[dict[str, Any]],
    records_seen: int,
    skill_model_path: Path | None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "schema": SNAPSHOT_CAPTURE_SOURCE_SCHEMA,
        "capture_run_id": run_id,
        "trajectory_jsonl": str(config.trajectory_jsonl) if config.trajectory_jsonl else None,
        "selection": {
            "auto": list(config.auto_selectors),
        },
        "save_slot_base": config.save_slot_base,
        "records_seen": records_seen,
        "generated_at_epoch_seconds": int(time.time()),
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "memory_path": str(config.memory_path),
        "policy_id": config.policy_id,
        "goal_preset": config.goal_preset,
        "settle_states_after_capture": config.settle_states_after_capture,
        "snapshot_dir": str(config.snapshot_dir),
        "verify_loads": config.verify_loads,
    }
    if skill_model_path is not None:
        source["skill_model_path"] = str(skill_model_path)
    return {
        "schema": SNAPSHOT_CURRICULUM_SCHEMA,
        "name": config.name,
        "source": source,
        "stages": stages,
    }


async def _verify_captured_slots(
    client: DoomAgentClient,
    stages: list[dict[str, Any]],
    *,
    timeout_seconds: float,
) -> None:
    for stage in stages:
        snapshot = stage.get("snapshot", {})
        slot = _int_or_none(snapshot.get("slot"))
        if slot is None:
            continue
        response = await client.load_snapshot(
            slot=slot,
            run_id=f"verify-{stage.get('name', 'snapshot')}",
        )
        verification = {
            "schema": SNAPSHOT_LOAD_VERIFICATION_SCHEMA,
            "slot": slot,
            "accepted": response.accepted,
            "load_queued": response.load_queued,
            "message": response.message,
        }
        if response.accepted and response.load_queued:
            observed = await _observe_matching_state(
                client,
                stage.get("expected_state", {}),
                timeout_seconds=timeout_seconds,
            )
            verification.update(observed)
        stage["load_verification"] = verification


async def _observe_matching_state(
    client: DoomAgentClient,
    expected: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    stream = client.observe(include_delta_state=False)
    deadline = time.monotonic() + timeout_seconds
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    try:
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                state = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
            except (asyncio.TimeoutError, StopAsyncIteration):
                break
            summary = summarize_state(state)
            first = first or summary
            last = summary
            verification = _verify_snapshot_restored_state(
                actual=summary,
                expected=expected,
                raw_state=state,
                enabled=True,
                tick_tolerance=35,
                verify_stream_tick=False,
                position_tolerance_fp=160 * 65536,
            )
            if verification["valid"]:
                return {
                    "matched": True,
                    "observed_state": _compact_observed_state(summary),
                    "verification": verification,
                }
    finally:
        await stream.aclose()
    return {
        "matched": False,
        "first_observed_state": _compact_observed_state(first or {}),
        "last_observed_state": _compact_observed_state(last or {}),
    }


def _compact_observed_state(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {}
    combat = summary.get("combat") if isinstance(summary.get("combat"), dict) else {}
    return {
        "tick": summary.get("tick"),
        "level_time": summary.get("level_time"),
        "episode": summary.get("episode"),
        "map": summary.get("map"),
        "health": summary.get("health"),
        "kills": summary.get("kills"),
        "items": summary.get("items"),
        "position_fp": summary.get("position_fp"),
        "shootable_target": combat.get("has_shootable_target"),
        "target_is_enemy": combat.get("target_is_enemy"),
    }


def _resolve_skill_model_path(
    skill_model_path: Path | None,
    memory: AgentMemory,
) -> Path | None:
    if skill_model_path is not None:
        return skill_model_path if skill_model_path.exists() else None
    learned = memory.data.get("learned_policy")
    if not isinstance(learned, dict):
        return None
    checkpoint = learned.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        return None
    path = Path(checkpoint)
    return path if path.exists() else None


def _config_from_args(args: argparse.Namespace) -> SnapshotCaptureConfig:
    endpoint = args.endpoint
    token = args.token
    agent_port = args.agent_port
    tls = args.tls
    microvm_id = args.microvm_id
    capsule = args.capsule
    if args.token_json is not None:
        with args.token_json.open("r", encoding="utf-8") as handle:
            token_data = json.load(handle)
        endpoint = endpoint or token_data.get("endpoint")
        token = token or token_data.get("token")
        agent_port = agent_port or int(token_data.get("port") or 50051)
        tls = tls or bool(token_data.get("tls"))
        microvm_id = microvm_id or token_data.get("microvm_id")
        capsule = capsule or token_data.get("capsule")
    if not endpoint:
        raise ValueError("--endpoint or --token-json with endpoint is required")
    selectors = tuple(args.auto or ())
    if not selectors:
        raise ValueError("choose at least one --auto selector")
    return SnapshotCaptureConfig(
        endpoint=str(endpoint),
        token=token,
        agent_port=int(agent_port or 50051),
        tls=bool(tls),
        authority=args.authority,
        memory_path=args.memory_path,
        skill_model_path=args.skill_model_path,
        trajectory_jsonl=args.trajectory_jsonl,
        output_path=args.output,
        name=args.name,
        snapshot_dir=args.snapshot_dir,
        auto_selectors=selectors,
        save_slot_base=args.save_slot_base,
        capsule=capsule or "agent-doom",
        microvm_id=microvm_id,
        goal_preset=args.goal_preset,
        mission=args.mission,
        max_states=args.max_states,
        policy_id=args.policy_id,
        reconnect=not args.no_reconnect,
        max_reconnects=args.max_reconnects,
        stop_after_captured=not args.no_stop_after_captured,
        settle_states_after_capture=args.settle_states_after_capture,
        verify_loads=args.verify_loads,
        verify_timeout_seconds=args.verify_timeout_seconds,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint")
    parser.add_argument("--token")
    parser.add_argument("--token-json", type=Path)
    parser.add_argument("--agent-port", type=int)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--authority")
    parser.add_argument("--memory-path", type=Path, default=Path("agent_memory/e1m1.json"))
    parser.add_argument("--skill-model-path", type=Path)
    parser.add_argument(
        "--trajectory-jsonl",
        type=Path,
        default=Path("trajectories/snapshot-capture.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="e1m1-progressed-bottlenecks")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument(
        "--auto",
        action="append",
        choices=sorted(AUTO_SELECTORS),
        help="capture the first live state matching this milestone; may be repeated",
    )
    parser.add_argument("--save-slot-base", type=int, default=0)
    parser.add_argument("--capsule")
    parser.add_argument("--microvm-id")
    parser.add_argument("--goal-preset", default="combat")
    parser.add_argument(
        "--mission",
        default="capture progressed-map bottleneck snapshots",
    )
    parser.add_argument("--max-states", type=int, default=12000)
    parser.add_argument("--policy-id", default="snapshot_capture_brain")
    parser.add_argument("--no-reconnect", action="store_true")
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--no-stop-after-captured", action="store_true")
    parser.add_argument("--settle-states-after-capture", type=int, default=2)
    parser.add_argument("--verify-loads", action="store_true")
    parser.add_argument("--verify-timeout-seconds", type=float, default=4.0)
    args = parser.parse_args(argv)

    try:
        config = _config_from_args(args)
        manifest = asyncio.run(capture_snapshot_curriculum(config))
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "schema": "restfuldoom.snapshot_capture_result.v1",
                "output": str(config.output_path),
                "stage_count": len(manifest["stages"]),
                "selectors": list(config.auto_selectors),
                "validation_valid": manifest.get("validation", {}).get("valid"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
