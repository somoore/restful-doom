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
    agent_pb2,
    summarize_state,
)
from .env import (
    ACTION_SCHEMA,
    DoomAgentEnv,
    DoomEnvConfig,
    OBSERVATION_SCHEMA,
    SKILL_ACTIONS,
    SkillController,
    _has_shootable_enemy,
    _route_outcome,
    _verify_snapshot_restored_state,
)
from .ppo import PPOTrainer
from .reward import RewardEngine, goal_preset
from .rollout_config import safe_endpoint_host
from .skill_policy import SkillPolicyModel
from .snapshot_builder import (
    AUTO_SELECTORS,
    POST_COMBAT_KILL_THRESHOLD,
    _damage_delta,
    _episode_map,
    _has_enemy_shootable_target,
    _has_shootable_target,
    _has_visible_enemy,
    _is_post_combat,
    _is_post_combat_exit_route,
    _int_or_none,
    _kill_delta,
    _record_state,
    _stage_from_record,
)
from .snapshot_curriculum import SNAPSHOT_CURRICULUM_SCHEMA, validate_snapshot_curriculum

SNAPSHOT_CAPTURE_SOURCE_SCHEMA = "restfuldoom.snapshot_capture_source.v1"
NATIVE_SNAPSHOT_CAPTURE_SCHEMA = "restfuldoom.native_snapshot_capture.v1"
SNAPSHOT_LOAD_VERIFICATION_SCHEMA = "restfuldoom.snapshot_load_verification.v1"
MAX_NATIVE_SAVE_SLOT = 9


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
        "first-enemy-shootable",
        "first-damage",
    )
    post_combat_kills: int = POST_COMBAT_KILL_THRESHOLD
    save_slot_base: int = 0
    capsule: str = "agent-doom"
    microvm_id: str | None = None
    goal_preset: str = "combat"
    mission: str = "capture progressed-map bottleneck snapshots"
    max_states: int = 12000
    attempts: int = 1
    reset_before_attempt: bool = False
    reset_skill: int = 2
    reset_episode: int = 1
    reset_map: int = 1
    reset_seed_base: int = 0
    reset_ready_level_time: int = 5
    policy_id: str = "snapshot_capture_brain"
    ppo_checkpoint: Path | None = None
    ppo_device: str = "cpu"
    ppo_sample: bool = False
    allowed_skills: tuple[str, ...] = ()
    strict_allowed_skills: bool = False
    reconnect: bool = True
    max_reconnects: int = 5
    stop_after_captured: bool = True
    settle_states_after_capture: int = 2
    verify_loads: bool = False
    verify_timeout_seconds: float = 4.0
    ppo_env_capture: bool = False


class SnapshotMilestoneTracker:
    """Tracks first occurrence of requested milestone selectors."""

    def __init__(
        self,
        selectors: tuple[str, ...],
        *,
        post_combat_kills: int = POST_COMBAT_KILL_THRESHOLD,
    ) -> None:
        unknown = sorted(set(selectors) - AUTO_SELECTORS)
        if unknown:
            choices = ", ".join(sorted(AUTO_SELECTORS))
            raise ValueError(
                f"unknown snapshot selector(s): {', '.join(unknown)}; choose from {choices}"
            )
        self.selectors = tuple(dict.fromkeys(selectors))
        self.post_combat_kills = int(post_combat_kills)
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
            elif selector == "first-enemy-shootable" and _has_enemy_shootable_target(record):
                matches.append(selector)
            elif selector == "first-damage" and _damage_delta(record) > 0:
                matches.append(selector)
            elif selector == "first-kill" and self._matches_first_kill(record):
                matches.append(selector)
            elif selector == "post-combat" and _is_post_combat(
                record,
                min_kills=self.post_combat_kills,
            ):
                matches.append(selector)
            elif (
                selector == "post-combat-exit-route"
                and _is_post_combat_exit_route(
                    record,
                    min_kills=self.post_combat_kills,
                )
            ):
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


class PPOSnapshotCapturePolicy:
    """Drives snapshot capture with a trained PPO skill selector."""

    def __init__(
        self,
        *,
        trainer: PPOTrainer,
        memory: AgentMemory,
        params: Any,
        policy_id: str,
        allowed_skills: tuple[str, ...] = (),
        strict_allowed_skills: bool = False,
        deterministic: bool = True,
        ready_episode: int = 1,
        ready_map: int = 1,
        ready_level_time: int = 5,
    ) -> None:
        self.trainer = trainer
        self.controller = SkillController(
            memory=memory,
            params=params,
            policy_id=f"{policy_id}-controller",
        )
        self.policy_id = policy_id
        self.allowed_skills = tuple(allowed_skills)
        self.strict_allowed_skills = bool(strict_allowed_skills)
        self.deterministic = bool(deterministic)
        self.ready_episode = int(ready_episode)
        self.ready_map = int(ready_map)
        self.ready_level_time = max(0, int(ready_level_time))
        self._policy_started = False
        self.last_decision: dict[str, Any] = {}
        self.last_error: str | None = None
        self.error_count = 0
        self.fallback_count = 0
        self._previous_state: Any | None = None
        self._previous_action_index: int | None = None
        self._previous_decision: dict[str, Any] | None = None
        self._previous_had_shootable_target = False

    async def next_action(self, state: Any) -> Any:
        """Selects one PPO skill and dispatches it through SkillController."""
        if not self._policy_ready(state):
            self.last_decision = {
                "policy_source": "ppo_checkpoint",
                "policy_id": self.policy_id,
                "ppo_waiting_for_reset_ready": True,
                "ready_episode": self.ready_episode,
                "ready_map": self.ready_map,
                "ready_level_time": self.ready_level_time,
                "live_episode": _live_state_episode_map(state)[0],
                "live_map": _live_state_episode_map(state)[1],
                "live_level_time": _live_state_level_time(state),
            }
            return agent_pb2.PlayerAction()
        if not self._policy_started:
            self.controller.reset_episode_context()
            self._policy_started = True
        self._record_previous_action_outcome(state)
        observation = self.controller.observation(state)
        raw_mask = list(self.controller.action_mask(state))
        action_mask, filter_info = _filter_allowed_skill_mask(
            raw_mask,
            allowed_skills=self.allowed_skills,
            strict_allowed_skills=self.strict_allowed_skills,
            heuristic_action_index=self.controller.heuristic_action_index(state),
        )
        action_index, logprob, value = self.trainer.model.act(
            observation,
            action_mask=action_mask,
            deterministic=self.deterministic,
        )
        action, decision = self.controller.action_for(action_index, state)
        decision = dict(decision)
        decision["policy_source"] = "ppo_checkpoint"
        decision["policy_id"] = self.policy_id
        decision["ppo_logprob"] = round(float(logprob), 6)
        decision["ppo_value"] = round(float(value), 6)
        decision["ppo_deterministic"] = self.deterministic
        decision["action_mask_allowed_count"] = sum(1 for allowed in action_mask if allowed)
        decision["raw_action_mask_allowed_count"] = sum(1 for allowed in raw_mask if allowed)
        decision["allowed_skill_filter"] = filter_info
        if bool(filter_info.get("fallback_applied")):
            self.fallback_count += 1

        self.controller.last_decision = decision
        self.controller.policy.last_decision = decision
        self.last_decision = decision
        self._previous_state = state
        self._previous_action_index = action_index
        self._previous_decision = decision
        self._previous_had_shootable_target = _has_shootable_enemy(state)
        return action

    def _policy_ready(self, state: Any) -> bool:
        episode, map_number = _live_state_episode_map(state)
        level_time = _live_state_level_time(state)
        return (
            episode == self.ready_episode
            and map_number == self.ready_map
            and level_time is not None
            and level_time >= self.ready_level_time
        )

    def _record_previous_action_outcome(self, current_state: Any) -> None:
        if self._previous_state is None or self._previous_action_index is None:
            return
        if _live_state_episode_map(current_state) != _live_state_episode_map(
            self._previous_state
        ):
            self.controller.reset_episode_context()
            self._previous_state = None
            self._previous_action_index = None
            self._previous_decision = None
            self._previous_had_shootable_target = False
            return
        skill = SKILL_ACTIONS[self._previous_action_index]
        outcome = _route_outcome(
            skill,
            self._previous_state,
            current_state,
            decision=self._previous_decision,
        )
        self.controller.record_action_history(
            action_index=self._previous_action_index,
            had_shootable_target=self._previous_had_shootable_target,
            route_outcome=outcome,
        )


def _filter_allowed_skill_mask(
    mask: list[bool],
    *,
    allowed_skills: tuple[str, ...],
    strict_allowed_skills: bool,
    heuristic_action_index: int | None = None,
) -> tuple[list[bool], dict[str, Any]]:
    """Applies the PPO skill allowlist with DoomEnv strict-mask semantics."""
    allowed_skills = tuple(allowed_skills or ())
    if not allowed_skills:
        return mask, {
            "schema": "restfuldoom.allowed_skill_filter.v1",
            "configured": False,
        }
    allowed = set(allowed_skills)
    filtered = [
        bool(value) and skill in allowed
        for skill, value in zip(SKILL_ACTIONS, mask, strict=False)
    ]
    info: dict[str, Any] = {
        "schema": "restfuldoom.allowed_skill_filter.v1",
        "configured": True,
        "strict": bool(strict_allowed_skills),
        "allowed_skills": list(allowed_skills),
        "raw_allowed_count": sum(1 for value in mask if value),
        "filtered_allowed_count": sum(1 for value in filtered if value),
        "fallback_applied": False,
        "fallback_skill": None,
    }
    if any(filtered):
        return filtered, info
    if not strict_allowed_skills:
        info["fallback_applied"] = True
        info["fallback_skill"] = "unfiltered_mask"
        return mask, info
    if heuristic_action_index is not None and 0 <= heuristic_action_index < len(SKILL_ACTIONS):
        heuristic_skill = SKILL_ACTIONS[int(heuristic_action_index)]
        if heuristic_skill in allowed:
            fallback = [False for _ in SKILL_ACTIONS]
            fallback[int(heuristic_action_index)] = True
            info["fallback_applied"] = True
            info["fallback_skill"] = heuristic_skill
            info["fallback_reason"] = "heuristic_allowed_skill"
            info["filtered_allowed_count"] = 1
            return fallback, info
    for skill in allowed_skills:
        if skill in SKILL_ACTIONS:
            fallback = [False for _ in SKILL_ACTIONS]
            fallback[SKILL_ACTIONS.index(skill)] = True
            info["fallback_applied"] = True
            info["fallback_skill"] = skill
            info["fallback_reason"] = "first_allowed_skill"
            info["filtered_allowed_count"] = 1
            return fallback, info
    return filtered, info


def _live_state_episode_map(state: Any) -> tuple[int | None, int | None]:
    """Returns episode/map for a live protobuf state or summarized state dict."""
    if isinstance(state, dict):
        return _episode_map(state)
    level = getattr(state, "level", None)
    return (
        _int_or_none(getattr(level, "episode", None)),
        _int_or_none(getattr(level, "map", None)),
    )


def _live_state_level_time(state: Any) -> int | None:
    """Returns level time for a live protobuf state or summarized state dict."""
    if isinstance(state, dict):
        return _int_or_none(state.get("level_time"))
    level = getattr(state, "level", None)
    return _int_or_none(getattr(level, "level_time", None))


async def capture_snapshot_curriculum(config: SnapshotCaptureConfig) -> dict[str, Any]:
    """Runs the structured brain and captures native save slots at milestones."""
    memory = AgentMemory.load(config.memory_path)
    params = memory.best_params()
    skill_model_path = _resolve_skill_model_path(config.skill_model_path, memory)
    skill_model = SkillPolicyModel.load(skill_model_path) if skill_model_path else None
    client = DoomAgentClient(
        config.endpoint,
        token=config.token,
        agent_port=config.agent_port,
        tls=config.tls,
        authority=config.authority,
    )
    tracker = SnapshotMilestoneTracker(
        config.auto_selectors,
        post_combat_kills=config.post_combat_kills,
    )
    run_id = f"snapshot-capture-{uuid.uuid4().hex[:12]}"
    stages: list[dict[str, Any]] = []
    records_seen = 0
    last_capture_index: int | None = None
    attempt_reports: list[dict[str, Any]] = []

    base_metadata = {
        "source": "snapshot-capture",
        "run_id": run_id,
        "policy_id": config.policy_id,
        "goal_preset": config.goal_preset,
        "mission": config.mission,
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "memory_path": str(config.memory_path),
        "selectors": list(config.auto_selectors),
        "post_combat_kills": int(config.post_combat_kills),
        "save_slot_base": config.save_slot_base,
        "attempts": config.attempts,
        "reset_before_attempt": config.reset_before_attempt,
        "reset_ready_level_time": int(config.reset_ready_level_time),
    }
    if config.ppo_checkpoint is not None:
        base_metadata["ppo_checkpoint"] = str(config.ppo_checkpoint)
        base_metadata["ppo_device"] = config.ppo_device
        base_metadata["ppo_sample"] = bool(config.ppo_sample)
        base_metadata["allowed_skills"] = list(config.allowed_skills)
        base_metadata["strict_allowed_skills"] = bool(config.strict_allowed_skills)
    if skill_model_path is not None:
        base_metadata["skill_model_path"] = str(skill_model_path)

    async def capture_before_action_send(
        step: RolloutStep,
        *,
        attempt: int,
        trajectory_path: Path,
        global_record_index: int,
    ) -> None:
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
            trajectory=trajectory_path,
            run_id=run_id,
            attempt=attempt,
            global_record_index=global_record_index,
        )
        stages.append(stage)
        tracker.mark_captured(selectors)
        last_capture_index = global_record_index

    try:
        for attempt in range(1, max(1, config.attempts) + 1):
            attempt_started_records = records_seen
            attempt_trajectory = _attempt_trajectory_path(
                config.trajectory_jsonl,
                attempt=attempt,
                attempts=config.attempts,
            )
            attempt_report: dict[str, Any] = {
                "attempt": attempt,
                "trajectory_jsonl": str(attempt_trajectory) if attempt_trajectory else None,
                "records_before": records_seen,
                "captured_before": sorted(tracker.captured),
            }
            if config.reset_before_attempt:
                attempt_report["reset"] = await _reset_capture_attempt(
                    client,
                    config,
                    run_id=run_id,
                    attempt=attempt,
                )

            if config.ppo_checkpoint is not None:
                trainer = PPOTrainer.load_checkpoint(
                    config.ppo_checkpoint,
                    device=config.ppo_device,
                    target_obs_dim=len(OBSERVATION_SCHEMA["feature_names"]),
                    target_action_dim=len(ACTION_SCHEMA["actions"]),
                )
                policy = PPOSnapshotCapturePolicy(
                    trainer=trainer,
                    memory=memory,
                    params=params,
                    policy_id=f"{config.policy_id}-attempt-{attempt}",
                    allowed_skills=config.allowed_skills,
                    strict_allowed_skills=config.strict_allowed_skills,
                    deterministic=not config.ppo_sample,
                    ready_episode=int(config.reset_episode),
                    ready_map=int(config.reset_map),
                    ready_level_time=int(config.reset_ready_level_time),
                )
            else:
                policy = BrainPolicy(
                    memory=memory,
                    params=params,
                    policy_id=f"{config.policy_id}-attempt-{attempt}",
                    skill_model=skill_model,
                )
            reward = RewardEngine(goal_preset(config.goal_preset))
            metadata = {
                **base_metadata,
                "attempt": attempt,
                "trajectory_jsonl": str(attempt_trajectory) if attempt_trajectory else None,
                "reset": attempt_report.get("reset"),
            }
            stop_attempt = False

            async def attempt_capture(step: RolloutStep) -> None:
                await capture_before_action_send(
                    step,
                    attempt=attempt,
                    trajectory_path=attempt_trajectory or Path("<stream>"),
                    global_record_index=attempt_started_records + step.index,
                )

            async for step in client.stream_rollout(
                policy,
                reward_engine=reward,
                max_states=config.max_states,
                trajectory_jsonl=attempt_trajectory,
                reconnect=config.reconnect,
                backoff=BackoffConfig(max_attempts=config.max_reconnects),
                on_step_before_action_send=attempt_capture,
                rollout_metadata=metadata,
            ):
                records_seen += 1
                if (
                    config.stop_after_captured
                    and tracker.complete
                    and last_capture_index is not None
                    and records_seen - last_capture_index >= config.settle_states_after_capture
                ):
                    stop_attempt = True
                    break

            attempt_report["records_after"] = records_seen
            attempt_report["records_seen"] = records_seen - attempt_started_records
            attempt_report["captured_after"] = sorted(tracker.captured)
            attempt_report["complete"] = tracker.complete
            attempt_reports.append(attempt_report)
            if stop_attempt or tracker.complete:
                break

        if not stages:
            raise RuntimeError(
                "snapshot capture finished without matching any requested milestones"
            )

        if config.verify_loads:
            await _verify_captured_slots(
                client,
                stages,
                timeout_seconds=config.verify_timeout_seconds,
            )
    finally:
        await client.close()

    manifest = _build_manifest(
        config,
        run_id=run_id,
        stages=stages,
        records_seen=records_seen,
        skill_model_path=skill_model_path,
        attempt_reports=attempt_reports,
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


async def capture_ppo_env_snapshot_curriculum(
    config: SnapshotCaptureConfig,
) -> dict[str, Any]:
    """Runs PPO through DoomAgentEnv and captures native save slots at milestones."""
    if config.ppo_checkpoint is None:
        raise ValueError("--ppo-env-capture requires --ppo-checkpoint")

    trainer = PPOTrainer.load_checkpoint(
        config.ppo_checkpoint,
        device=config.ppo_device,
        target_obs_dim=len(OBSERVATION_SCHEMA["feature_names"]),
        target_action_dim=len(ACTION_SCHEMA["actions"]),
    )
    tracker = SnapshotMilestoneTracker(
        config.auto_selectors,
        post_combat_kills=config.post_combat_kills,
    )
    run_id = f"ppo-env-snapshot-capture-{uuid.uuid4().hex[:12]}"
    stages: list[dict[str, Any]] = []
    records_seen = 0
    last_capture_index: int | None = None
    attempt_reports: list[dict[str, Any]] = []

    base_metadata = {
        "source": "ppo-env-snapshot-capture",
        "run_id": run_id,
        "policy_id": config.policy_id,
        "goal_preset": config.goal_preset,
        "mission": config.mission,
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "memory_path": str(config.memory_path),
        "selectors": list(config.auto_selectors),
        "post_combat_kills": int(config.post_combat_kills),
        "save_slot_base": config.save_slot_base,
        "attempts": config.attempts,
        "reset_ready_level_time": int(config.reset_ready_level_time),
        "ppo_checkpoint": str(config.ppo_checkpoint),
        "ppo_device": config.ppo_device,
        "ppo_sample": bool(config.ppo_sample),
        "allowed_skills": list(config.allowed_skills),
        "strict_allowed_skills": bool(config.strict_allowed_skills),
        "ppo_env_capture": True,
    }

    for attempt in range(1, max(1, config.attempts) + 1):
        attempt_started_records = records_seen
        attempt_trajectory = _attempt_trajectory_path(
            config.trajectory_jsonl,
            attempt=attempt,
            attempts=config.attempts,
        )
        seed = int(config.reset_seed_base) + max(0, attempt - 1)
        metadata = {
            **base_metadata,
            "attempt": attempt,
            "trajectory_jsonl": str(attempt_trajectory) if attempt_trajectory else None,
            "reset": {
                "schema": "restfuldoom.snapshot_capture_reset.v1",
                "attempt": int(attempt),
                "skill": int(config.reset_skill),
                "episode": int(config.reset_episode),
                "map": int(config.reset_map),
                "seed": int(seed),
            },
        }
        attempt_report: dict[str, Any] = {
            "attempt": attempt,
            "trajectory_jsonl": str(attempt_trajectory) if attempt_trajectory else None,
            "records_before": records_seen,
            "captured_before": sorted(tracker.captured),
            "seed": seed,
        }
        trace_handle = None
        env = DoomAgentEnv(
            DoomEnvConfig(
                endpoint=config.endpoint,
                token=config.token,
                agent_port=config.agent_port,
                tls=config.tls,
                authority=config.authority,
                skill=config.reset_skill,
                episode=config.reset_episode,
                map=config.reset_map,
                seed=seed,
                run_id=f"{run_id}-attempt-{attempt}",
                goal_preset=config.goal_preset,
                max_steps=config.max_states,
                memory_path=config.memory_path,
                reset_ready_level_time=config.reset_ready_level_time,
                allowed_skills=config.allowed_skills,
                strict_allowed_skills=config.strict_allowed_skills,
            )
        )
        try:
            if attempt_trajectory is not None:
                attempt_trajectory.parent.mkdir(parents=True, exist_ok=True)
                trace_handle = attempt_trajectory.open("w", encoding="utf-8")
            obs = await env.reset(seed=seed)
            stop_attempt = False
            for step_index in range(1, int(config.max_states) + 1):
                action_mask = env.action_mask()
                action_index, _logprob, _value = trainer.model.act(
                    obs,
                    deterministic=not config.ppo_sample,
                    action_mask=action_mask,
                )
                step = await env.step(action_index)
                record = _record_from_env_step(
                    policy_id=f"ppo:{config.ppo_checkpoint}",
                    episode_index=attempt - 1,
                    seed=seed,
                    step_index=step_index,
                    action_index=action_index,
                    action_mask=action_mask,
                    reward=step.reward,
                    done=step.done,
                    observation=step.observation,
                    info=step.info,
                    rollout_metadata=metadata,
                )
                if trace_handle is not None:
                    json.dump(record, trace_handle, default=str, sort_keys=True)
                    trace_handle.write("\n")
                    trace_handle.flush()
                obs = step.observation
                records_seen += 1
                selectors = tracker.observe(record)
                if selectors:
                    if env.client is None:
                        raise RuntimeError("Doom environment client is not available")
                    stage = await _capture_stage(
                        env.client,
                        config,
                        record,
                        selectors=selectors,
                        line_index=step_index,
                        order=len(stages),
                        trajectory=attempt_trajectory or Path("<ppo-env-stream>"),
                        run_id=run_id,
                        attempt=attempt,
                        global_record_index=records_seen,
                    )
                    stages.append(stage)
                    tracker.mark_captured(selectors)
                    last_capture_index = records_seen
                if (
                    config.stop_after_captured
                    and tracker.complete
                    and last_capture_index is not None
                    and records_seen - last_capture_index >= config.settle_states_after_capture
                ):
                    stop_attempt = True
                    break
                if step.done:
                    break
            attempt_report["records_after"] = records_seen
            attempt_report["records_seen"] = records_seen - attempt_started_records
            attempt_report["captured_after"] = sorted(tracker.captured)
            attempt_report["complete"] = tracker.complete
            attempt_reports.append(attempt_report)
            if stop_attempt or tracker.complete:
                break
        finally:
            if trace_handle is not None:
                trace_handle.close()
            await env.close()

    if not stages:
        raise RuntimeError(
            "snapshot capture finished without matching any requested milestones"
        )

    if config.verify_loads:
        client = DoomAgentClient(
            config.endpoint,
            token=config.token,
            agent_port=config.agent_port,
            tls=config.tls,
            authority=config.authority,
        )
        try:
            await _verify_captured_slots(
                client,
                stages,
                timeout_seconds=config.verify_timeout_seconds,
            )
        finally:
            await client.close()

    manifest = _build_manifest(
        config,
        run_id=run_id,
        stages=stages,
        records_seen=records_seen,
        skill_model_path=None,
        attempt_reports=attempt_reports,
    )
    manifest["source"]["source"] = "ppo-env-snapshot-capture"
    manifest["source"]["ppo_env_capture"] = True
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


async def _reset_capture_attempt(
    client: DoomAgentClient,
    config: SnapshotCaptureConfig,
    *,
    run_id: str,
    attempt: int,
) -> dict[str, Any]:
    seed = int(config.reset_seed_base) + max(0, int(attempt) - 1)
    response = await client.reset_episode(
        skill=int(config.reset_skill),
        episode=int(config.reset_episode),
        map=int(config.reset_map),
        seed=seed,
        run_id=f"{run_id}-attempt-{attempt}-reset",
    )
    reset = {
        "schema": "restfuldoom.snapshot_capture_reset.v1",
        "attempt": int(attempt),
        "accepted": bool(response.accepted),
        "message": response.message,
        "skill": int(response.skill),
        "episode": int(response.episode),
        "map": int(response.map),
        "seed": int(response.seed),
        "seed_applied": bool(response.seed_applied),
        "start_queued": bool(response.start_queued),
    }
    if not response.accepted:
        raise RuntimeError(
            f"snapshot capture reset rejected on attempt {attempt}: {response.message}"
        )
    return reset


def _attempt_trajectory_path(
    path: Path | None,
    *,
    attempt: int,
    attempts: int,
) -> Path | None:
    if path is None:
        return None
    if attempts <= 1:
        return path
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}-attempt-{attempt:03d}{suffix}")


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
    attempt: int = 1,
    global_record_index: int | None = None,
) -> dict[str, Any]:
    selector = selectors[0]
    slot = config.save_slot_base + order
    if slot < 0 or slot > MAX_NATIVE_SAVE_SLOT:
        raise RuntimeError(
            "native snapshot save slot range exhausted: "
            f"requested slot {slot}, valid range is 0..{MAX_NATIVE_SAVE_SLOT}"
        )
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
    if response.slot != slot:
        raise RuntimeError(
            "snapshot save slot was normalized by the server: "
            f"requested {slot}, got {response.slot}; choose a save-slot-base "
            f"that keeps all stages in 0..{MAX_NATIVE_SAVE_SLOT}"
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
        save_slot=response.slot,
        capsule=config.capsule,
        microvm_id=config.microvm_id,
    )
    stage["validated"] = True
    stage["evidence"]["capture_attempt"] = int(attempt)
    stage["evidence"]["attempt_record_index"] = int(line_index)
    if global_record_index is not None:
        stage["evidence"]["global_record_index"] = int(global_record_index)
    stage["capture"] = {
        "schema": NATIVE_SNAPSHOT_CAPTURE_SCHEMA,
        "method": "grpc_save_snapshot",
        "attempt": int(attempt),
        "attempt_record_index": int(line_index),
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


def _record_from_env_step(
    *,
    policy_id: str,
    episode_index: int,
    seed: int,
    step_index: int,
    action_index: int,
    action_mask: list[bool],
    reward: float,
    done: bool,
    observation: list[float],
    info: dict[str, Any],
    rollout_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Builds a snapshot-selector trajectory row from one DoomAgentEnv step."""
    transition = info.get("transition")
    transition_summary = transition if isinstance(transition, dict) else {}
    reward_summary = {
        "reward": round(float(reward), 6),
        "done": bool(done),
        "done_reason": info.get("done_reason"),
        **{
            key: transition_summary.get(key)
            for key in (
                "damage_delta",
                "enemy_distance_delta",
                "health_delta",
                "item_delta",
                "kill_delta",
                "progress_delta",
                "secret_delta",
            )
            if key in transition_summary
        },
    }
    decision = info.get("decision")
    policy_decision = dict(decision) if isinstance(decision, dict) else {}
    policy_decision["ppo_skill"] = info.get("skill")
    policy_decision["ppo_action_index"] = int(action_index)
    state = info.get("state")
    return {
        "policy_id": policy_id,
        "episode_index": int(episode_index),
        "seed": int(seed),
        "step_index": int(step_index),
        "index": int(step_index),
        "action_index": int(action_index),
        "action_mask": [bool(value) for value in action_mask],
        "reward": reward_summary,
        "done": bool(done),
        "done_reason": info.get("done_reason"),
        "skill": info.get("skill"),
        "decision": decision if isinstance(decision, dict) else {},
        "action": info.get("action", {}),
        "state": state if isinstance(state, dict) else {},
        "transition": transition_summary,
        "route_outcome": info.get("route_outcome", {}),
        "observation": list(observation),
        "info": info,
        "metadata": {
            "policy_decision": policy_decision,
            "rollout": rollout_metadata,
        },
    }


def _build_manifest(
    config: SnapshotCaptureConfig,
    *,
    run_id: str,
    stages: list[dict[str, Any]],
    records_seen: int,
    skill_model_path: Path | None,
    attempt_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "schema": SNAPSHOT_CAPTURE_SOURCE_SCHEMA,
        "capture_run_id": run_id,
        "trajectory_jsonl": str(config.trajectory_jsonl) if config.trajectory_jsonl else None,
        "selection": {
            "auto": list(config.auto_selectors),
            "post_combat_kills": int(config.post_combat_kills),
        },
        "save_slot_base": config.save_slot_base,
        "records_seen": records_seen,
        "attempts": max(1, int(config.attempts)),
        "max_states_per_attempt": int(config.max_states),
        "reset_before_attempt": bool(config.reset_before_attempt),
        "generated_at_epoch_seconds": int(time.time()),
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "memory_path": str(config.memory_path),
        "policy_id": config.policy_id,
        "goal_preset": config.goal_preset,
        "settle_states_after_capture": config.settle_states_after_capture,
        "snapshot_dir": str(config.snapshot_dir),
        "verify_loads": config.verify_loads,
    }
    if attempt_reports is not None:
        source["attempt_reports"] = attempt_reports
    if config.reset_before_attempt:
        source["reset"] = {
            "schema": "restfuldoom.snapshot_capture_reset_config.v1",
            "skill": int(config.reset_skill),
            "episode": int(config.reset_episode),
            "map": int(config.reset_map),
            "seed_base": int(config.reset_seed_base),
            "ready_level_time": int(config.reset_ready_level_time),
        }
    if skill_model_path is not None:
        source["skill_model_path"] = str(skill_model_path)
    if config.ppo_checkpoint is not None:
        source["ppo_checkpoint"] = str(config.ppo_checkpoint)
        source["ppo_device"] = config.ppo_device
        source["ppo_sample"] = bool(config.ppo_sample)
        source["allowed_skills"] = list(config.allowed_skills)
        source["strict_allowed_skills"] = bool(config.strict_allowed_skills)
    if config.ppo_env_capture:
        source["ppo_env_capture"] = True
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
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.post_combat_kills < 0:
        raise ValueError("--post-combat-kills must be non-negative")
    if args.ppo_checkpoint is not None and not args.ppo_checkpoint.exists():
        raise ValueError(f"--ppo-checkpoint does not exist: {args.ppo_checkpoint}")
    if args.ppo_env_capture and args.ppo_checkpoint is None:
        raise ValueError("--ppo-env-capture requires --ppo-checkpoint")
    max_requested_slot = int(args.save_slot_base) + len(selectors) - 1
    if args.save_slot_base < 0 or max_requested_slot > MAX_NATIVE_SAVE_SLOT:
        raise ValueError(
            "--save-slot-base plus selected milestones must fit native slots "
            f"0..{MAX_NATIVE_SAVE_SLOT}; requested range "
            f"{args.save_slot_base}..{max_requested_slot}"
        )
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
        post_combat_kills=args.post_combat_kills,
        save_slot_base=args.save_slot_base,
        capsule=capsule or "agent-doom",
        microvm_id=microvm_id,
        goal_preset=args.goal_preset,
        mission=args.mission,
        max_states=args.max_states,
        attempts=args.attempts,
        reset_before_attempt=args.reset_before_attempt,
        reset_skill=args.reset_skill,
        reset_episode=args.reset_episode,
        reset_map=args.reset_map,
        reset_seed_base=args.reset_seed_base,
        reset_ready_level_time=args.reset_ready_level_time,
        policy_id=args.policy_id,
        ppo_checkpoint=args.ppo_checkpoint,
        ppo_device=args.ppo_device,
        ppo_sample=bool(args.ppo_sample),
        allowed_skills=tuple(args.allowed_skill or ()),
        strict_allowed_skills=bool(args.strict_allowed_skills),
        reconnect=not args.no_reconnect,
        max_reconnects=args.max_reconnects,
        stop_after_captured=not args.no_stop_after_captured,
        settle_states_after_capture=args.settle_states_after_capture,
        verify_loads=args.verify_loads,
        verify_timeout_seconds=args.verify_timeout_seconds,
        ppo_env_capture=bool(args.ppo_env_capture),
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
    parser.add_argument(
        "--post-combat-kills",
        type=int,
        default=POST_COMBAT_KILL_THRESHOLD,
        help="minimum absolute kill count for post-combat auto selectors",
    )
    parser.add_argument("--capsule")
    parser.add_argument("--microvm-id")
    parser.add_argument("--goal-preset", default="combat")
    parser.add_argument(
        "--mission",
        default="capture progressed-map bottleneck snapshots",
    )
    parser.add_argument("--max-states", type=int, default=12000)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help=(
            "number of independent capture attempts to run; --max-states applies "
            "to each attempt"
        ),
    )
    parser.add_argument(
        "--reset-before-attempt",
        action="store_true",
        help="queue a fast ResetEpisode before each capture attempt",
    )
    parser.add_argument("--reset-skill", type=int, default=2)
    parser.add_argument("--reset-episode", type=int, default=1)
    parser.add_argument("--reset-map", type=int, default=1)
    parser.add_argument("--reset-seed-base", type=int, default=0)
    parser.add_argument(
        "--reset-ready-level-time",
        type=int,
        default=5,
        help=(
            "Delay PPO-driven capture until the reset stream reaches this level "
            "time, matching DoomEnv's true-spawn reset readiness gate."
        ),
    )
    parser.add_argument("--policy-id", default="snapshot_capture_brain")
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        help=(
            "Drive capture with a trained PPO skill checkpoint instead of the "
            "structured BrainPolicy."
        ),
    )
    parser.add_argument("--ppo-device", default="cpu")
    parser.add_argument(
        "--ppo-sample",
        action="store_true",
        help="Sample PPO actions during capture instead of deterministic argmax.",
    )
    parser.add_argument(
        "--ppo-env-capture",
        action="store_true",
        help=(
            "When --ppo-checkpoint is set, drive capture through DoomAgentEnv's "
            "PPO eval cadence instead of the streamed policy wrapper."
        ),
    )
    parser.add_argument(
        "--allowed-skill",
        action="append",
        default=[],
        choices=ACTION_SCHEMA["actions"],
        help=(
            "PPO capture skill allowlist applied after the normal action mask; "
            "repeat to keep multiple skills."
        ),
    )
    parser.add_argument(
        "--strict-allowed-skills",
        action="store_true",
        help=(
            "When --allowed-skill filters out every normal PPO action, keep capture "
            "inside the allowlist instead of falling back to the unfiltered mask."
        ),
    )
    parser.add_argument("--no-reconnect", action="store_true")
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--no-stop-after-captured", action="store_true")
    parser.add_argument("--settle-states-after-capture", type=int, default=2)
    parser.add_argument("--verify-loads", action="store_true")
    parser.add_argument("--verify-timeout-seconds", type=float, default=4.0)
    args = parser.parse_args(argv)

    try:
        config = _config_from_args(args)
        if config.ppo_env_capture:
            manifest = asyncio.run(capture_ppo_env_snapshot_curriculum(config))
        else:
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
                "attempts": config.attempts,
                "validation_valid": manifest.get("validation", {}).get("valid"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
