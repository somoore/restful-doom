"""Gym-style protobuf environment for Doom skill learning."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brain import (
    BT_ATTACK,
    BT_USE,
    CELL_UNITS,
    FP,
    AgentMemory,
    BrainPolicy,
    BrainPolicyParams,
    EXIT_LINE_SPECIALS,
    MANUAL_USE_LINE_SPECIALS,
    cell_key,
    extract_features,
    raw_ticcmd_action,
    raw_turn_for_delta,
)
from .client import DoomAgentClient, agent_pb2, semantic_action, summarize_state
from .client import EpisodeStart as ClientEpisodeStart
from .reward import Goal, RewardEngine, TransitionReward, goal_preset
from .schemas import (
    ACTION_SCHEMA,
    DECISION_CYCLE_SCHEMA,
    MEMORY_CONTRACT,
    OBSERVATION_SCHEMA,
    PPO_SKILL_ACTIONS,
    encode_action_history_features,
    encode_contact_context_features,
    encode_temporal_context_features,
    encode_topology_context_features,
    encode_visible_contact_context_features,
)
from .skill_policy import features_from_tactical

SKILL_ACTIONS = PPO_SKILL_ACTIONS
LOW_HEALTH_RETREAT_STREAK_LIMIT = 96


def _line_is_exit(line: dict[str, Any] | None) -> bool:
    """Returns whether a summarized navigation line is an exit trigger."""
    if not isinstance(line, dict):
        return False
    return int(line.get("special", 0)) in EXIT_LINE_SPECIALS


@dataclass(frozen=True)
class DoomEnvConfig:
    """Configuration for one resettable Doom training environment."""

    endpoint: str = "127.0.0.1:50051"
    token: str | None = None
    agent_port: int = 50051
    tls: bool = False
    authority: str | None = None
    skill: int = 2
    episode: int = 1
    map: int = 1
    seed: int = 0
    run_id: str = "ppo"
    goal_preset: str = "combat"
    target_x_fp: int | None = None
    target_y_fp: int | None = None
    max_steps: int = 700
    level_complete_bonus: float = 100.0
    kill_goal_bonus: float = 10.0
    required_kills: int = 1
    memory_path: Path | None = None
    reset_timeout_seconds: float = 5.0
    reset_attempts: int = 2
    reset_start_x_fp: int | None = None
    reset_start_y_fp: int | None = None
    reset_start_angle_degrees: int = 0
    reset_start_face_nearest_enemy: bool = False
    reset_start_health: int | None = None
    reset_start_armor: int | None = None
    reset_start_ammo_bullets: int | None = None
    max_action_tics: int = 8
    reset_warmup_steps: int = 0
    reset_warmup_max_tics: int = 0
    reset_warmup_until_visible: bool = False
    reset_warmup_until_shootable: bool = False
    shootable_fire_reward: float = 0.5
    missed_fire_penalty: float = 0.05
    blind_fire_penalty: float = 0.02
    route_progress_reward: float = 0.01
    route_reached_reward: float = 0.25
    route_failure_penalty: float = 0.03
    exit_route_progress_reward: float = 0.01
    exit_route_reached_reward: float = 0.5
    exit_route_failure_penalty: float = 0.05
    first_visible_bonus: float = 0.0
    first_shootable_bonus: float = 0.0
    visible_contact_progress_reward: float = 0.0
    terminate_on_first_visible: bool = False
    terminate_on_first_shootable: bool = False
    terminate_on_required_kills: bool = False
    allowed_skills: tuple[str, ...] = ()
    strict_allowed_skills: bool = False
    curriculum: dict[str, Any] | None = None
    curriculum_stage: dict[str, Any] | None = None
    reset_mode: str = "episode"
    snapshot: dict[str, Any] | None = None
    snapshot_verify_restored_state: bool = True
    snapshot_verify_tick_tolerance: int = 35
    snapshot_verify_stream_tick: bool = False
    snapshot_verify_position_tolerance_fp: int = 160 * 65536

    def reward_goal(self) -> Goal:
        """Returns the reward goal for the configured preset."""
        if self.goal_preset == "custom":
            return Goal()
        return goal_preset(
            self.goal_preset,
            target_x_fp=self.target_x_fp,
            target_y_fp=self.target_y_fp,
        )


@dataclass(frozen=True)
class EnvStep:
    """One Doom environment transition."""

    observation: list[float]
    reward: float
    done: bool
    info: dict[str, Any]


class SkillController:
    """Executes PPO-selected high-level skills with the deterministic brain."""

    def __init__(
        self,
        *,
        memory: AgentMemory | None = None,
        params: BrainPolicyParams | None = None,
        policy_id: str = "ppo_skill_controller",
    ) -> None:
        self.memory = memory or AgentMemory.load(Path("/tmp/restfuldoom-ppo-memory.json"))
        self.params = params or BrainPolicyParams()
        self.policy = BrainPolicy(
            memory=self.memory,
            params=self.params,
            policy_id=policy_id,
        )
        self.last_decision: dict[str, Any] = {}
        self._previous_action_index: int | None = None
        self._previous_had_shootable_target = False
        self._same_skill_streak = 0
        self._previous_route_progression = False
        self._previous_route_progress_units = 0.0
        self._route_waypoint_reached_recently = False
        self._route_waypoint_failed_recently = False
        self._failed_route_attempt_count = 0
        self._previous_observation_snapshot: dict[str, Any] | None = None
        self._same_cell_observation_streak = 0
        self._recent_visible_enemy_flags: list[bool] = []
        self._recent_shootable_target_flags: list[bool] = []
        self._recent_route_progress_units: list[float] = []
        self._recent_route_failure_flags: list[bool] = []
        self._recent_contact_use_line: dict[str, int] | None = None
        self._recent_visible_contact: dict[str, int] | None = None

    def observation(self, state: Any) -> list[float]:
        """Encodes a protobuf state as the stable PPO feature vector."""
        features = extract_features(state, self.memory, self.params)
        return [
            *features_from_tactical(features),
            *encode_action_history_features(
                previous_action_index=self._previous_action_index,
                previous_had_shootable_target=self._previous_had_shootable_target,
                same_skill_streak=self._same_skill_streak,
                previous_route_progression=self._previous_route_progression,
                previous_route_progress_units=self._previous_route_progress_units,
                route_waypoint_reached_recently=self._route_waypoint_reached_recently,
                route_waypoint_failed_recently=self._route_waypoint_failed_recently,
                failed_route_attempt_count=self._failed_route_attempt_count,
            ),
            *self._temporal_context_features(features),
            *self._contact_context_features(features),
            *self._topology_context_features(features),
            *self._visible_contact_context_features(features),
        ]

    def reset_episode_context(self) -> None:
        """Clears stateful observation features at episode boundaries."""
        self.last_decision = {}
        self._previous_action_index = None
        self._previous_had_shootable_target = False
        self._same_skill_streak = 0
        self._previous_route_progression = False
        self._previous_route_progress_units = 0.0
        self._route_waypoint_reached_recently = False
        self._route_waypoint_failed_recently = False
        self._failed_route_attempt_count = 0
        self._previous_observation_snapshot = None
        self._same_cell_observation_streak = 0
        self._recent_visible_enemy_flags.clear()
        self._recent_shootable_target_flags.clear()
        self._recent_route_progress_units.clear()
        self._recent_route_failure_flags.clear()
        self._recent_contact_use_line = None
        self._recent_visible_contact = None
        if hasattr(self.policy, "reset_episode_context"):
            self.policy.reset_episode_context()

    def record_action_history(
        self,
        *,
        action_index: int,
        had_shootable_target: bool,
        route_outcome: dict[str, Any] | None = None,
    ) -> None:
        """Records the previous macro action for the next PPO observation."""
        if self._previous_action_index == action_index:
            self._same_skill_streak += 1
        else:
            self._same_skill_streak = 1
        self._previous_action_index = action_index
        self._previous_had_shootable_target = bool(had_shootable_target)
        outcome = route_outcome if isinstance(route_outcome, dict) else {}
        self._previous_route_progression = bool(outcome.get("attempted"))
        self._previous_route_progress_units = float(outcome.get("progress_units") or 0.0)
        self._route_waypoint_reached_recently = bool(outcome.get("reached"))
        self._route_waypoint_failed_recently = bool(outcome.get("failed"))
        if self._route_waypoint_failed_recently:
            self._failed_route_attempt_count += 1
        elif self._previous_route_progression and (
            self._route_waypoint_reached_recently or self._previous_route_progress_units > 4.0
        ):
            self._failed_route_attempt_count = 0
        if self._previous_route_progression:
            self._append_recent(
                self._recent_route_progress_units,
                self._previous_route_progress_units,
            )
            self._append_recent(
                self._recent_route_failure_flags,
                self._route_waypoint_failed_recently,
            )
        if had_shootable_target:
            self._append_recent(self._recent_shootable_target_flags, True)

    def _temporal_context_features(self, features: Any) -> list[float]:
        """Returns bounded temporal context and advances the observation snapshot."""
        current_snapshot = self._observation_snapshot(features)
        previous_snapshot = self._previous_observation_snapshot
        if previous_snapshot is None:
            delta_x = 0.0
            delta_y = 0.0
            movement_distance = 0.0
            enemy_distance_delta = 0.0
            route_distance_delta = 0.0
            cell_changed = False
            self._same_cell_observation_streak = 1
            encoded_same_cell_observation_streak = 0
        else:
            delta_x = current_snapshot["x_units"] - previous_snapshot["x_units"]
            delta_y = current_snapshot["y_units"] - previous_snapshot["y_units"]
            movement_distance = math.hypot(delta_x, delta_y)
            enemy_distance_delta = (
                previous_snapshot["enemy_distance"] - current_snapshot["enemy_distance"]
                if previous_snapshot["enemy_distance"] is not None
                and current_snapshot["enemy_distance"] is not None
                else 0.0
            )
            route_distance_delta = (
                previous_snapshot["route_distance"] - current_snapshot["route_distance"]
                if previous_snapshot["route_distance"] is not None
                and current_snapshot["route_distance"] is not None
                else 0.0
            )
            cell_changed = current_snapshot["cell"] != previous_snapshot["cell"]
            if cell_changed:
                self._same_cell_observation_streak = 1
            else:
                self._same_cell_observation_streak += 1
            encoded_same_cell_observation_streak = self._same_cell_observation_streak

        self._append_recent(
            self._recent_visible_enemy_flags,
            bool(current_snapshot["visible_enemy"]),
        )
        self._append_recent(
            self._recent_shootable_target_flags,
            bool(current_snapshot["shootable_target"]),
        )
        route_attempts = max(1, len(self._recent_route_failure_flags))
        recent_route_failure_ratio = (
            sum(1 for failed in self._recent_route_failure_flags if failed) / route_attempts
            if self._recent_route_failure_flags
            else 0.0
        )
        temporal = encode_temporal_context_features(
            delta_x_units=delta_x,
            delta_y_units=delta_y,
            movement_distance_units=movement_distance,
            enemy_distance_delta_units=enemy_distance_delta,
            route_distance_delta_units=route_distance_delta,
            same_cell_observation_streak=encoded_same_cell_observation_streak,
            cell_changed_recently=cell_changed,
            visible_enemy_seen_recently=any(self._recent_visible_enemy_flags),
            shootable_target_seen_recently=any(self._recent_shootable_target_flags),
            recent_route_progress_units=sum(self._recent_route_progress_units),
            recent_route_failure_ratio=recent_route_failure_ratio,
        )
        self._previous_observation_snapshot = current_snapshot
        return temporal

    @staticmethod
    def _observation_snapshot(features: Any) -> dict[str, Any]:
        route_waypoint = features.navigation.get("route_waypoint", {})
        route_line = (
            route_waypoint.get("line", {})
            if isinstance(route_waypoint, dict)
            and isinstance(route_waypoint.get("line", {}), dict)
            else {}
        )
        enemy_distance = None
        if features.visible_enemies:
            enemy_distance = float(features.visible_enemies[0]["distance"])
        elif features.known_enemies:
            enemy_distance = float(features.known_enemies[0]["distance"])
        return {
            "x_units": float(features.x_units),
            "y_units": float(features.y_units),
            "cell": str(features.cell),
            "enemy_distance": enemy_distance,
            "route_distance": (
                float(route_line["distance"])
                if route_line and "distance" in route_line
                else None
            ),
            "visible_enemy": bool(features.visible_enemies),
            "shootable_target": bool(
                features.combat.get("has_shootable_target")
                and features.combat.get("target_is_enemy")
            ),
        }

    @staticmethod
    def _append_recent(values: list[Any], value: Any, *, max_length: int = 4) -> None:
        values.append(value)
        if len(values) > max_length:
            del values[: len(values) - max_length]

    def _contact_context_features(self, features: Any) -> list[float]:
        """Returns explicit contact/use-line state for PPO observations."""
        shootable = self.policy._shootable_enemy(features) is not None
        current_visible_contact = bool(features.visible_enemies) and not shootable
        line = self._recent_contact_use_line_for(features)
        age_tics = 0
        if line is not None:
            recent = self._recent_contact_use_line or {}
            age_tics = max(0, int(features.tick) - int(recent.get("tick", features.tick)))
        elif current_visible_contact:
            line = self._contact_use_line(features)
        distance = self.policy._line_control_distance(line) if line is not None else 0.0
        angle_delta = self.policy._line_control_angle_delta(line) if line is not None else 0.0
        return encode_contact_context_features(
            recent_contact_active=current_visible_contact or self._recent_contact_active(features),
            contact_use_line_active=line is not None,
            contact_use_line_distance_units=distance,
            contact_use_line_angle_degrees=angle_delta,
            contact_use_line_close=line is not None and distance <= 220.0 and abs(angle_delta) <= 24.0,
            contact_use_line_followthrough_active=self._contact_use_line_followthrough_active(features),
            contact_use_line_age_tics=age_tics,
        )

    def _topology_context_features(self, features: Any) -> list[float]:
        """Returns compact local topology and cell-visit context."""
        persistent_cells = self.memory.data.get("cells", {})
        if not isinstance(persistent_cells, dict):
            persistent_cells = {}
        current_visits = self._cell_visit_count(
            features.cell,
            persistent_cells=persistent_cells,
        )
        open_cells = self._projected_open_probe_cells(
            features,
            persistent_cells=persistent_cells,
        )
        if not open_cells:
            return encode_topology_context_features(
                current_cell_visits=current_visits,
            )
        visits = [int(cell["visits"]) for cell in open_cells]
        best = min(
            open_cells,
            key=lambda cell: (
                int(cell["visits"]),
                abs(float(cell["angle_offset_degrees"])),
            ),
        )
        exhausted = sum(1 for visit_count in visits if visit_count > 1)
        return encode_topology_context_features(
            current_cell_visits=current_visits,
            open_cell_min_visits=min(visits),
            open_cell_mean_visits=sum(visits) / max(1, len(visits)),
            frontier_active=int(best["visits"]) <= 1,
            frontier_angle_degrees=float(best["angle_offset_degrees"]),
            exhausted_open_ratio=exhausted / max(1, len(open_cells)),
        )

    def _visible_contact_context_features(self, features: Any) -> list[float]:
        """Returns current visible-contact geometry for PPO observations."""
        enemy = features.visible_enemies[0] if features.visible_enemies else None
        if enemy is None:
            return encode_visible_contact_context_features()
        shootable = self.policy._shootable_enemy(features) is not None
        angle_delta = float(enemy.get("angle_delta", 0.0))
        distance = float(enemy.get("distance", 0.0))
        return encode_visible_contact_context_features(
            visible_contact_active=True,
            visible_contact_shootable=shootable,
            visible_contact_distance_units=distance,
            visible_contact_angle_degrees=angle_delta,
            visible_contact_aligned=abs(angle_delta) <= self.params.aim_tolerance_degrees,
            visible_contact_close=distance <= 700.0,
        )

    def _projected_open_probe_cells(
        self,
        features: Any,
        *,
        persistent_cells: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Projects open direction probes into coarse map cells with visit counts."""
        navigation = getattr(features, "navigation", {})
        probes = navigation.get("direction_probes", [])
        if not isinstance(probes, list):
            return []
        probe_distance = float(navigation.get("probe_distance_fp", 0) or 0) / FP
        cells: dict[str, dict[str, Any]] = {}
        for probe in probes:
            if not isinstance(probe, dict) or not probe.get("open"):
                continue
            offset = float(probe.get("angle_offset_degrees", 0.0))
            distance = float(probe.get("block_distance_fp", 0) or 0) / FP
            step = max(
                CELL_UNITS,
                min(1.5 * CELL_UNITS, distance or probe_distance or CELL_UNITS),
            )
            heading = math.radians((float(features.angle) + offset) % 360.0)
            projected_x = float(features.x_units) + math.cos(heading) * step
            projected_y = float(features.y_units) + math.sin(heading) * step
            key = cell_key(
                0.0 if abs(projected_x) < 1e-6 else projected_x,
                0.0 if abs(projected_y) < 1e-6 else projected_y,
            )
            visits = self._cell_visit_count(key, persistent_cells=persistent_cells)
            previous = cells.get(key)
            if previous is None or abs(offset) < abs(float(previous["angle_offset_degrees"])):
                cells[key] = {
                    "cell": key,
                    "visits": visits,
                    "angle_offset_degrees": offset,
                }
        return list(cells.values())

    def _cell_visit_count(
        self,
        key: str,
        *,
        persistent_cells: dict[str, Any],
    ) -> int:
        cell = persistent_cells.get(key, {})
        persistent_visits = int(cell.get("visits", 0)) if isinstance(cell, dict) else 0
        episode_visits = int(self.policy._episode_cell_visits.get(key, 0))
        return max(0, persistent_visits + episode_visits)

    def action_for(self, action_index: int, state: Any) -> tuple[Any, dict[str, Any]]:
        """Returns the PlayerAction for a PPO skill index."""
        if action_index < 0 or action_index >= len(SKILL_ACTIONS):
            raise ValueError(f"action_index must be in [0, {len(SKILL_ACTIONS) - 1}]")

        features = extract_features(state, self.memory, self.params)
        self.policy.last_features = features
        self._remember_visible_contact(features)
        if self.policy._start_kills is None:
            self.policy._start_kills = features.kills
        self.policy._episode_cell_visits[features.cell] = (
            self.policy._episode_cell_visits.get(features.cell, 0) + 1
        )
        stuck = self.policy._is_stuck(features)
        skill = SKILL_ACTIONS[action_index]
        action, decision = self._execute_skill(skill, features, stuck)
        decision = dict(decision)
        decision["ppo_skill"] = skill
        decision["ppo_action_index"] = action_index
        decision["previous_ppo_action_index"] = self._previous_action_index
        decision["previous_ppo_skill"] = (
            SKILL_ACTIONS[self._previous_action_index]
            if self._previous_action_index is not None
            else None
        )
        decision["previous_had_shootable_target"] = self._previous_had_shootable_target
        decision["same_skill_streak"] = self._same_skill_streak
        self.policy.last_decision = decision
        self.last_decision = decision
        return action, decision

    def heuristic_action_index(self, state: Any) -> int:
        """Returns a deterministic skill baseline action index for a state."""
        features = extract_features(state, self.memory, self.params)
        stuck = self.policy._is_stuck(features)
        if stuck:
            return SKILL_ACTIONS.index("recover_stuck")
        if self.policy._shootable_enemy(features) is not None:
            return SKILL_ACTIONS.index("fire")
        if features.visible_enemies:
            if features.health <= self.params.retreat_health:
                return SKILL_ACTIONS.index("retreat")
            return SKILL_ACTIONS.index("close_visible_contact")
        if self.policy._select_local_exit_line(features) is not None:
            return SKILL_ACTIONS.index("press_exit")
        if (
            self.policy._select_nearby_use_line(features) is not None
            or self._recent_contact_use_line_for(features) is not None
            or self.policy._select_use_ray(features) is not None
            or self.policy._should_use_ahead(features)
        ):
            return SKILL_ACTIONS.index("open_use_line")
        if self.policy._select_progression_line(features) is not None:
            return SKILL_ACTIONS.index("route_progression")
        if self.policy._select_known_enemy(features) is not None and self._blind_seek_allowed(features):
            return SKILL_ACTIONS.index("seek_enemy")
        return SKILL_ACTIONS.index("route_progression")

    def action_mask(self, state: Any) -> list[bool]:
        """Returns currently feasible PPO skills from protobuf affordances."""
        features = extract_features(state, self.memory, self.params)
        stuck = self.policy._is_stuck(features)
        mask = {skill: False for skill in SKILL_ACTIONS}

        shootable = self.policy._shootable_enemy(features)
        if shootable is None:
            self._remember_visible_contact(features)
        can_fire = shootable is not None and self.policy._can_shoot(features, shootable)
        recent_contact_active = self._recent_contact_active(features)
        low_health_contact = (
            features.health <= self.params.retreat_health
            and (features.visible_enemies or recent_contact_active)
        )
        if low_health_contact and self._low_health_retreat_allowed(
            features,
            recent_contact_active=recent_contact_active,
        ):
            mask["retreat"] = True
            if stuck:
                mask["recover_stuck"] = True
            return [mask[skill] for skill in SKILL_ACTIONS]
        if can_fire:
            mask["fire"] = True
            if stuck:
                mask["recover_stuck"] = True
            return [mask[skill] for skill in SKILL_ACTIONS]
        elif self._contact_use_line_followthrough_active(features):
            mask["open_use_line"] = True
            if stuck:
                mask["recover_stuck"] = True
            return [mask[skill] for skill in SKILL_ACTIONS]

        if features.visible_enemies:
            if not can_fire:
                mask["close_visible_contact"] = True
            if not can_fire and shootable is None and self.policy._select_known_enemy(features) is not None:
                mask["seek_enemy"] = True
            contact_line = self._contact_use_line(features)
            if (
                not can_fire
                and shootable is None
                and contact_line is not None
                and self._contact_use_line_ready_for_visible_contact(features, contact_line)
            ):
                mask["open_use_line"] = True
            nearest_visible = min(
                (float(enemy["distance"]) for enemy in features.visible_enemies),
                default=99999.0,
            )
            if (
                features.health <= self.params.retreat_health
                or (not can_fire and nearest_visible <= self.params.close_enemy_units)
            ):
                mask["retreat"] = True
        elif recent_contact_active:
            mask["close_visible_contact"] = True
            if (
                self.policy._select_known_enemy(features) is not None
                and self._post_contact_seek_allowed(features)
            ):
                mask["seek_enemy"] = True
            contact_line = self._recent_contact_use_line_for(features)
            if (
                contact_line is not None
                and self._contact_use_line_ready_for_visible_contact(features, contact_line)
            ):
                mask["open_use_line"] = True

        if (
            not features.visible_enemies
            and self.policy._select_known_enemy(features) is not None
            and self._blind_seek_allowed(features, recent_contact_active=recent_contact_active)
        ):
            mask["seek_enemy"] = True

        progression_line = self.policy._select_progression_line(features)

        if (
            not can_fire
            and not recent_contact_active
            and (
                self.policy._select_nearby_use_line(features) is not None
                or (
                    self._recent_contact_use_line_for(features) is not None
                    and self._contact_use_line_ready_for_visible_contact(
                        features,
                        self._recent_contact_use_line_for(features),
                    )
                )
                or self.policy._select_use_ray(features) is not None
                or self.policy._should_use_ahead(features)
            )
            and not self._visible_contact_needs_closure(features)
        ):
            mask["open_use_line"] = True

        if not can_fire and progression_line is not None:
            if not self._suppress_route_after_contact_failures(
                features,
                recent_contact_active,
                progression_line=progression_line,
            ):
                mask["route_progression"] = True
        elif not can_fire and not features.visible_enemies and not recent_contact_active:
            mask["route_progression"] = True

        if stuck:
            mask["recover_stuck"] = True

        if not can_fire and self.policy._select_local_exit_line(features) is not None:
            mask["press_exit"] = True

        if not any(mask.values()):
            mask["route_progression"] = True

        return [mask[skill] for skill in SKILL_ACTIONS]

    def _execute_skill(self, skill: str, features: Any, stuck: bool) -> tuple[Any, dict[str, Any]]:
        if features.visible_enemies:
            self.policy._last_visible_enemy_tick = features.tick
            self.policy._last_visible_enemy_id = int(features.visible_enemies[0]["id"])

        if skill == "engage":
            enemy = features.visible_enemies[0] if features.visible_enemies else None
            if enemy is not None:
                return self.policy._engage(features, enemy, stuck)
            contact = self.policy._continue_last_contact_corridor(features, stuck)
            if contact is not None:
                return contact
            return self.policy._explore(features, stuck)

        if skill == "close_visible_contact":
            enemy = features.visible_enemies[0] if features.visible_enemies else None
            if enemy is not None and self.policy._shootable_enemy(features) is None:
                contact_line = self._contact_use_line(features, allow_blocked=False)
                if (
                    contact_line is not None
                    and not self._contact_use_line_ready_for_visible_contact(
                        features,
                        contact_line,
                    )
                ):
                    self._remember_contact_use_line(features, contact_line)
                    return self._use_contact_line(features, contact_line, stuck)
                contact = self.policy._close_visible_contact(
                    features,
                    enemy,
                    stuck,
                    "ppo_close_visible_contact",
                    prefer_open_ray=True,
                )
                if contact is not None:
                    return contact
                return self.policy._turn_toward_or_move(
                    features,
                    enemy["angle_delta"],
                    "ppo_close_visible_contact",
                    stuck,
                    enemy=enemy,
                )
            contact = self.policy._continue_last_contact_corridor(features, stuck)
            if contact is not None:
                return contact
            contact_line = self._recent_contact_use_line_for(features)
            if contact_line is not None:
                return self._use_contact_line(features, contact_line, stuck)
            enemy = self.policy._select_known_enemy(features)
            if enemy is not None:
                return self.policy._turn_toward_or_move(
                    features,
                    enemy["angle_delta"],
                    "ppo_close_recent_contact",
                    stuck,
                    enemy=enemy,
                )
            return self.policy._explore(features, stuck)

        if skill == "fire":
            enemy = self.policy._shootable_enemy(features)
            if enemy is not None and self.policy._can_shoot(features, enemy):
                self.policy._last_shot_tick = features.tick
                return (
                    raw_ticcmd_action(
                        buttons=BT_ATTACK,
                        duration_tics=1,
                        tick=features.tick,
                    ),
                    self.policy._decision("ppo_fire", features, enemy=enemy, stuck=stuck),
                )
            visible = features.visible_enemies[0] if features.visible_enemies else None
            if visible is not None:
                return self.policy._engage(features, visible, stuck)
            return (
                semantic_action(agent_pb2.ACTION_SHOOT, duration_tics=1, tick=features.tick),
                self.policy._decision("ppo_fire_probe", features, stuck=stuck),
            )

        if skill == "seek_enemy":
            enemy = features.visible_enemies[0] if features.visible_enemies else None
            if enemy is not None:
                contact = self.policy._close_visible_contact(
                    features,
                    enemy,
                    stuck,
                    "ppo_seek_visible_contact",
                )
                if contact is not None:
                    return contact
                return self.policy._turn_toward_or_move(
                    features,
                    enemy["angle_delta"],
                    "ppo_seek_visible_enemy",
                    stuck,
                    enemy=enemy,
                )
            enemy = self.policy._select_known_enemy(features)
            contact = self.policy._continue_last_contact_corridor(features, stuck)
            if contact is not None:
                return contact
            if enemy is not None:
                return self.policy._turn_toward_or_move(
                    features,
                    enemy["angle_delta"],
                    "ppo_seek_enemy",
                    stuck,
                    enemy=enemy,
                )
            return self.policy._explore(features, stuck)

        if skill == "open_use_line":
            line = self.policy._select_nearby_use_line(features)
            remember_contact_line = (
                line is not None
                and bool(features.visible_enemies)
                and self.policy._shootable_enemy(features) is None
            )
            contact_line = remember_contact_line
            if line is None:
                line = self._contact_use_line(features)
                remember_contact_line = line is not None
                contact_line = line is not None
            if line is None:
                line = self._recent_contact_use_line_for(features)
                contact_line = line is not None
            if line is not None:
                if contact_line and not self._contact_use_line_ready_for_visible_contact(
                    features,
                    line,
                ):
                    enemy = features.visible_enemies[0] if features.visible_enemies else None
                    if enemy is not None:
                        contact = self.policy._close_visible_contact(
                            features,
                            enemy,
                            stuck,
                            "ppo_close_visible_contact",
                            prefer_open_ray=True,
                        )
                        if contact is not None:
                            return contact
                        return self.policy._turn_toward_or_move(
                            features,
                            enemy["angle_delta"],
                            "ppo_close_visible_contact",
                            stuck,
                            enemy=enemy,
                        )
                if remember_contact_line or contact_line:
                    self._remember_contact_use_line(features, line)
                if contact_line:
                    return self._use_contact_line(features, line, stuck, track_stall=False)
                return self.policy._use_nearby_line(features, line, stuck)
            if self._visible_contact_needs_closure(features):
                enemy = features.visible_enemies[0]
                contact = self.policy._close_visible_contact(
                    features,
                    enemy,
                    stuck,
                    "ppo_close_visible_contact",
                    prefer_open_ray=True,
                )
                if contact is not None:
                    return contact
                return self.policy._turn_toward_or_move(
                    features,
                    enemy["angle_delta"],
                    "ppo_close_visible_contact",
                    stuck,
                    enemy=enemy,
                )
            ray = self.policy._select_use_ray(features)
            if ray is not None:
                return self.policy._use_directional_line(features, ray, stuck)
            self.policy._last_use_tick = features.tick
            return (
                semantic_action(agent_pb2.ACTION_USE, duration_tics=2, tick=features.tick),
                self.policy._decision("ppo_use_ahead", features, stuck=stuck),
            )

        if skill == "route_progression":
            line = self.policy._select_progression_line(features)
            if line is None:
                line = self._contact_route_waypoint(features)
            if line is not None:
                return self.policy._advance_progression_line(features, line, stuck)
            return self.policy._explore(features, stuck)

        if skill == "retreat":
            enemy = (
                features.visible_enemies[0]
                if features.visible_enemies
                else self.policy._select_known_enemy(features)
            )
            if enemy is not None:
                return self.policy._retreat_or_fire(features, enemy, stuck)
            return (
                semantic_action(
                    agent_pb2.ACTION_BACKWARD,
                    amount=self.params.retreat_amount,
                    duration_tics=3,
                    tick=features.tick,
                ),
                self.policy._decision("ppo_retreat", features, stuck=stuck),
            )

        if skill == "recover_stuck":
            return self.policy._recover_from_stuck(features)

        if skill == "press_exit":
            line = self.policy._select_local_exit_line(features)
            if line is not None:
                return self.policy._advance_progression_line(features, line, stuck)
            self.policy._last_use_tick = features.tick
            return (
                raw_ticcmd_action(buttons=BT_USE, duration_tics=2, tick=features.tick),
                self.policy._decision("ppo_press_exit_probe", features, stuck=stuck),
            )

        return self.policy._explore(features, stuck)

    def _contact_route_waypoint(self, features: Any) -> dict[str, Any] | None:
        """Returns a route waypoint usable for visible-but-not-shootable contact."""
        if not features.visible_enemies or self.policy._shootable_enemy(features) is not None:
            return None
        route = features.navigation.get("route_waypoint", {})
        if not isinstance(route, dict):
            return None
        line = route.get("line", {})
        if not isinstance(line, dict) or int(line.get("line_id", 0)) <= 0:
            return None
        if float(line.get("distance", 999999.0)) > 2600.0:
            return None
        return line

    def _contact_use_line(
        self,
        features: Any,
        *,
        allow_blocked: bool = True,
    ) -> dict[str, Any] | None:
        """Returns a manual line worth approaching during visible contact."""
        if not features.visible_enemies or self.policy._shootable_enemy(features) is not None:
            return None
        candidates = []
        for line in features.navigation.get("use_lines", []):
            if int(line.get("special", 0)) not in MANUAL_USE_LINE_SPECIALS:
                continue
            if float(line.get("distance", 999999.0)) > 1200.0:
                continue
            if abs(float(line.get("angle_delta", 999.0))) > 120.0:
                continue
            if not allow_blocked and self.policy._is_line_blocked(features, line):
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                0 if int(line.get("side", 0)) == 0 else 1,
                abs(float(line.get("angle_delta", 999.0))),
                float(line.get("distance", 999999.0)),
            ),
        )

    def _visible_contact_needs_closure(self, features: Any) -> bool:
        """Returns true for visible enemy contact that is not yet shootable."""
        return bool(features.visible_enemies) and self.policy._shootable_enemy(features) is None

    def _contact_use_line_ready_for_visible_contact(
        self,
        features: Any,
        line: dict[str, Any] | None,
    ) -> bool:
        """Gate contact use-lines so far first-visible states close contact first."""
        if line is None:
            return False
        line_distance = self.policy._line_control_distance(line)
        line_angle = abs(self.policy._line_control_angle_delta(line))
        if line_distance <= 220.0 and line_angle <= 24.0:
            return True
        if not features.visible_enemies:
            return False
        enemy = features.visible_enemies[0]
        enemy_distance = float(enemy.get("distance", 999999.0))
        enemy_angle = abs(float(enemy.get("angle_delta", 999.0)))
        return enemy_distance <= 900.0 and enemy_angle <= 45.0

    def _use_contact_line(
        self,
        features: Any,
        line: dict[str, Any],
        stuck: bool,
        *,
        track_stall: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """Approach or press a manual line selected as part of visible contact."""
        if track_stall and not self.policy._record_line_attempt(
            features,
            line,
            allow_immediate_use_bypass=False,
        ):
            self._recent_contact_use_line = None
            return self.policy._explore(features, stuck)
        angle_delta = self.policy._line_control_angle_delta(line)
        distance = self.policy._line_control_distance(line)
        line_record = self.policy._line_record(line)
        if distance <= 220.0 and abs(angle_delta) <= 24.0:
            self.policy._last_use_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    forward_move=max(6, self.params.move_amount // 2),
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self.policy._decision(
                    "contact_use_line",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        if abs(angle_delta) <= 70.0:
            return (
                raw_ticcmd_action(
                    forward_move=self.params.move_amount,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=4,
                    tick=features.tick,
                ),
                self.policy._decision(
                    "contact_approach_use_line",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        return self.policy._use_nearby_line(features, line, stuck)

    def _remember_contact_use_line(self, features: Any, line: dict[str, Any]) -> None:
        line_id = int(line.get("line_id", 0))
        if line_id <= 0:
            return
        self._recent_contact_use_line = {
            "line_id": line_id,
            "tick": int(features.tick),
            "episode": int(features.episode),
            "map": int(features.map),
        }

    def _contact_use_line_followthrough_active(self, features: Any) -> bool:
        """Continue a contact use-line option long enough to activate it."""
        if self._previous_action_index != SKILL_ACTIONS.index("open_use_line"):
            return False
        if self.policy._shootable_enemy(features) is not None:
            return False
        line = self._recent_contact_use_line_for(features)
        if line is None:
            return False
        if (
            self._same_skill_streak >= 16
            and self.policy._line_control_distance(line) > 220.0
        ):
            return False
        return True

    def _remember_visible_contact(self, features: Any) -> None:
        """Remember recent visible-but-not-shootable contact for recovery masks."""
        if not features.visible_enemies:
            return
        if self.policy._shootable_enemy(features) is not None:
            return
        enemy = features.visible_enemies[0]
        self._recent_visible_contact = {
            "enemy_id": int(enemy["id"]),
            "tick": int(features.tick),
            "episode": int(features.episode),
            "map": int(features.map),
        }
        line = self._contact_use_line(features)
        if line is not None and self._contact_use_line_ready_for_visible_contact(features, line):
            self._remember_contact_use_line(features, line)

    def _recent_contact_use_line_for(self, features: Any) -> dict[str, Any] | None:
        recent = self._recent_contact_use_line
        if recent is None:
            return None
        if int(recent.get("episode", 0)) != int(features.episode):
            return None
        if int(recent.get("map", 0)) != int(features.map):
            return None
        if int(features.tick) - int(recent.get("tick", 0)) > 420:
            return None
        line_id = int(recent.get("line_id", 0))
        for line in features.navigation.get("use_lines", []):
            if int(line.get("line_id", 0)) != line_id:
                continue
            if int(line.get("special", 0)) not in MANUAL_USE_LINE_SPECIALS:
                return None
            if float(line.get("distance", 999999.0)) > 1400.0:
                return None
            if abs(float(line.get("angle_delta", 999.0))) > 160.0:
                return None
            if self.policy._is_line_blocked(features, line):
                return None
            return line
        return None

    def _recent_contact_active(self, features: Any) -> bool:
        if self._recent_contact_use_line_for(features) is not None:
            return True
        contact = self.policy._last_contact_ray
        if contact is not None:
            age = int(features.tick) - int(contact.get("tick", 0))
            if 0 <= age <= 90 and self.policy._select_known_enemy(features) is not None:
                return True
        return self._recent_visible_contact_active(features)

    def _low_health_retreat_allowed(
        self,
        features: Any,
        *,
        recent_contact_active: bool,
    ) -> bool:
        """Avoids trapping low-health no-LOS states in endless retreat."""
        if features.visible_enemies:
            return True
        if not recent_contact_active:
            return False
        if self._previous_action_index != SKILL_ACTIONS.index("retreat"):
            return True
        return self._same_skill_streak < LOW_HEALTH_RETREAT_STREAK_LIMIT

    def _blind_seek_allowed(
        self,
        features: Any,
        *,
        recent_contact_active: bool | None = None,
    ) -> bool:
        """Allows known-enemy seeking only after current-episode contact evidence."""
        if recent_contact_active is None:
            recent_contact_active = self._recent_contact_active(features)
        if recent_contact_active:
            return self._post_contact_seek_allowed(features)
        return any(self._recent_visible_enemy_flags)

    def _post_contact_seek_allowed(self, features: Any) -> bool:
        """Returns whether contact recovery has earned a broad known-enemy seek."""
        return False

    def _recent_visible_contact_active(self, features: Any) -> bool:
        recent = self._recent_visible_contact
        if recent is None:
            return False
        if int(recent.get("episode", 0)) != int(features.episode):
            return False
        if int(recent.get("map", 0)) != int(features.map):
            return False
        age = int(features.tick) - int(recent.get("tick", 0))
        if age < 0 or age > 420:
            return False
        enemy_id = int(recent.get("enemy_id", 0))
        for enemy in features.known_enemies:
            if int(enemy["id"]) == enemy_id:
                return not self.policy._is_blocked_target(features, enemy)
        return self.policy._select_known_enemy(features) is not None

    def _suppress_route_after_contact_failures(
        self,
        features: Any,
        recent_contact_active: bool,
        *,
        progression_line: dict[str, Any] | None = None,
    ) -> bool:
        """Avoid re-sampling route when it has repeatedly failed after contact."""
        if not recent_contact_active:
            return False
        if _line_is_exit(progression_line):
            return False
        if features.visible_enemies:
            return False
        if self._failed_route_attempt_count < 1:
            return False
        return self._recent_visible_contact_active(features)


class DoomAgentEnv:
    """Async Gym-style environment backed by the Doom gRPC stream."""

    observation_schema = OBSERVATION_SCHEMA
    action_schema = ACTION_SCHEMA

    def __init__(
        self,
        config: DoomEnvConfig | None = None,
        *,
        client: Any | None = None,
        controller: SkillController | None = None,
    ) -> None:
        self.config = config or DoomEnvConfig()
        memory = (
            AgentMemory.load(self.config.memory_path)
            if self.config.memory_path is not None
            else None
        )
        self.controller = controller or SkillController(memory=memory)
        self.client = client
        self._owns_client = client is None
        self._action_queue: asyncio.Queue[Any | None] | None = None
        self._state_stream: AsyncIterator[Any] | None = None
        self._reward_engine = RewardEngine(self.config.reward_goal())
        self._current_state: Any | None = None
        self._start_level: tuple[int, int] | None = None
        self._start_kills = 0
        self._steps = 0
        self._episode_index = 0
        self._last_reset_warmup: dict[str, Any] = {}
        self._last_reset_context: dict[str, Any] = {}
        self._last_action_mask_filter: dict[str, Any] = {}
        self._episode_seen_visible_enemy = False
        self._episode_seen_shootable_enemy = False

    async def close(self) -> None:
        """Closes stream and channel resources."""
        if self._action_queue is not None:
            try:
                self._action_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self.client is not None and self._owns_client:
            await self.client.close()
        self._state_stream = None
        self._action_queue = None
        self.client = None

    async def reset(self, *, seed: int | None = None) -> list[float]:
        """Resets Doom and returns the first compact observation."""
        self._steps = 0
        self._episode_index += 1
        if hasattr(self.controller, "reset_episode_context"):
            self.controller.reset_episode_context()
        reset_seed = self.config.seed if seed is None else seed
        if self.config.reset_mode == "snapshot":
            state = await self._reset_from_snapshot()
        elif self.config.reset_mode == "episode":
            await self._ensure_stream()
            state = await self._reset_with_retries(reset_seed)
        else:
            raise ValueError(f"unsupported reset_mode {self.config.reset_mode!r}")
        self._last_reset_warmup = {
            "enabled": False,
            "steps": 0,
            "tics": 0,
            "episode_index": self._episode_index,
        }
        if self.config.reset_warmup_steps > 0:
            state = await self._run_reset_warmup(state)
            if hasattr(self.controller, "reset_episode_context"):
                self.controller.reset_episode_context()
        self._last_reset_context = self._build_reset_context(
            seed=reset_seed,
            state=state,
            source="warmup"
            if self.config.reset_mode == "episode" and self.config.reset_warmup_steps > 0
            else self.config.reset_mode,
        )
        self._raise_if_snapshot_restore_unverified()
        self._reward_engine = RewardEngine(self.config.reward_goal())
        self._current_state = state
        self._start_level = (state.level.episode, state.level.map)
        self._start_kills = state.player.kills
        self._episode_seen_visible_enemy = _has_visible_enemy(state)
        self._episode_seen_shootable_enemy = _has_shootable_enemy(state)
        return self.controller.observation(state)

    def _build_reset_context(
        self,
        *,
        seed: int,
        state: Any,
        source: str,
    ) -> dict[str, Any]:
        stage = self.config.curriculum_stage or {}
        snapshot = self.config.snapshot or {}
        restore = stage.get("snapshot_restore", {}) if isinstance(stage, dict) else {}
        if not isinstance(restore, dict):
            restore = {}
        expected = stage.get("expected_state", {}) if isinstance(stage, dict) else {}
        if not isinstance(expected, dict):
            expected = {}
        reset_source = "snapshot_restore" if source == "snapshot" else source
        actual = summarize_state(state)
        verification = _verify_snapshot_restored_state(
            actual=actual,
            expected=expected,
            raw_state=state,
            enabled=reset_source == "snapshot_restore"
            and bool(self.config.snapshot_verify_restored_state),
            tick_tolerance=int(self.config.snapshot_verify_tick_tolerance),
            verify_stream_tick=bool(self.config.snapshot_verify_stream_tick),
            position_tolerance_fp=int(
                self.config.snapshot_verify_position_tolerance_fp
            ),
        )
        return {
            "schema": "restfuldoom.reset_context.v1",
            "source": reset_source,
            "episode_index": self._episode_index,
            "seed_label": int(seed),
            "skipped_reset_episode": self.config.reset_mode == "snapshot",
            "snapshot_id": snapshot.get("id"),
            "snapshot_path": snapshot.get("path"),
            "snapshot_ref": snapshot.get("ref"),
            "snapshot_digest": snapshot.get("digest"),
            "restore": {
                key: restore.get(key)
                for key in (
                    "schema",
                    "api_method",
                    "returncode",
                    "elapsed_seconds",
                    "restore_command_configured",
                    "slot",
                    "accepted",
                    "message",
                    "update_index",
                    "reset_index",
                )
                if key in restore
            },
            "expected_state": dict(expected),
            "actual_first_state": actual,
            "restored_state_verification": verification,
        }

    def _raise_if_snapshot_restore_unverified(self) -> None:
        if self._last_reset_context.get("source") != "snapshot_restore":
            return
        if not self.config.snapshot_verify_restored_state:
            return
        verification = self._last_reset_context.get("restored_state_verification", {})
        if isinstance(verification, dict) and verification.get("valid"):
            return
        errors = []
        if isinstance(verification, dict):
            errors = list(verification.get("errors", []))
        message = "; ".join(str(error) for error in errors) or "restored state mismatch"
        raise RuntimeError(f"snapshot restored-state verification failed: {message}")

    async def _reset_from_snapshot(self) -> Any:
        """Reconnects and observes the current restored state without fresh-resetting Doom."""
        if self._owns_client:
            if self.client is not None:
                await self.client.close()
            self.client = None
        else:
            if self._action_queue is not None:
                try:
                    self._action_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
        self._state_stream = None
        self._action_queue = None

        slot = _snapshot_slot(self.config.snapshot or {})
        if slot is not None:
            await self._load_server_snapshot(slot)
        await self._ensure_stream()
        return await self._next_live_state(self.config.episode, self.config.map)

    async def _load_server_snapshot(self, slot: int) -> None:
        await self._ensure_client()
        assert self.client is not None
        start = time.monotonic()
        response = await self.client.load_snapshot(
            slot=slot,
            run_id=f"{self.config.run_id}-{self._episode_index}-load-snapshot",
        )
        elapsed = round(time.monotonic() - start, 4)
        if not response.accepted:
            raise RuntimeError(f"snapshot load rejected: {response.message}")
        stage = dict(self.config.curriculum_stage or {})
        restore = dict(stage.get("snapshot_restore", {}))
        restore.update(
            {
                "schema": "restfuldoom.snapshot_restore.v1",
                "api_method": "grpc_load_snapshot",
                "restore_command_configured": False,
                "returncode": 0,
                "elapsed_seconds": elapsed,
                "slot": int(response.slot),
                "accepted": bool(response.accepted),
                "message": response.message,
            }
        )
        stage["snapshot_restore"] = restore
        self.config = DoomEnvConfig(
            **{
                **self.config.__dict__,
                "curriculum_stage": stage,
            }
        )

    async def _reset_with_retries(self, reset_seed: int) -> Any:
        attempts = max(1, int(self.config.reset_attempts))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            reset = await self.client.reset_episode(
                skill=self.config.skill,
                episode=self.config.episode,
                map=self.config.map,
                seed=reset_seed,
                run_id=f"{self.config.run_id}-{self._episode_index}-attempt-{attempt}",
                start=self._episode_start(),
            )
            if not reset.accepted:
                raise RuntimeError(f"reset rejected: {reset.message}")
            try:
                return await self._next_reset_state(reset.episode, reset.map)
            except TimeoutError as error:
                last_error = error
        assert last_error is not None
        raise last_error

    def _episode_start(self) -> ClientEpisodeStart | None:
        if (
            self.config.reset_start_x_fp is None
            and self.config.reset_start_y_fp is None
            and not self.config.reset_start_face_nearest_enemy
            and self.config.reset_start_health is None
            and self.config.reset_start_armor is None
            and self.config.reset_start_ammo_bullets is None
        ):
            return None
        return ClientEpisodeStart(
            x_fp=self.config.reset_start_x_fp,
            y_fp=self.config.reset_start_y_fp,
            angle_degrees=self.config.reset_start_angle_degrees,
            face_nearest_enemy=self.config.reset_start_face_nearest_enemy,
            health=self.config.reset_start_health,
            armor=self.config.reset_start_armor,
            ammo_bullets=self.config.reset_start_ammo_bullets,
        )

    async def step(self, action_index: int) -> EnvStep:
        """Applies one high-level PPO skill and returns a transition."""
        if self._current_state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._action_queue is None or self._state_stream is None:
            raise RuntimeError("environment stream is not open")

        previous = self._current_state
        action, decision = self.controller.action_for(action_index, previous)
        await self._action_queue.put(action)
        action_tics = max(
            1,
            min(
                self.config.max_action_tics,
                int(getattr(action, "duration_tics", 0) or 1),
            ),
        )
        current = previous
        total_reward = 0.0
        action_reward = 0.0
        done = False
        reason = None
        had_shootable_target = _has_shootable_enemy(previous)
        had_visible_enemy = _has_visible_enemy(previous)
        first_visible_contact = False
        first_shootable_contact = False
        contact_reward = 0.0
        visible_contact_distance_delta = 0.0
        visible_contact_progress_reward = 0.0
        transition_summaries: list[dict[str, Any]] = []
        for _ in range(action_tics):
            tick_previous = current
            current = await self._state_stream.__anext__()
            had_shootable_target = had_shootable_target or _has_shootable_enemy(current)
            had_visible_enemy = had_visible_enemy or _has_visible_enemy(current)
            self._steps += 1
            transition = self._reward_engine.score(tick_previous, current)
            done, reason, reward = self._terminal_reward(tick_previous, current, transition)
            total_reward += reward
            transition_summaries.append(_transition_summary(transition))
            contact_delta = _visible_contact_distance_delta(tick_previous, current)
            if contact_delta:
                visible_contact_distance_delta += contact_delta
                progress_reward = self._visible_contact_progress_reward(contact_delta)
                visible_contact_progress_reward += progress_reward
                total_reward += progress_reward
            if done:
                break
            contact = self._contact_reward(current)
            if contact["reward"]:
                contact_reward += float(contact["reward"])
                total_reward += float(contact["reward"])
            first_visible_contact = first_visible_contact or bool(contact["first_visible"])
            first_shootable_contact = first_shootable_contact or bool(contact["first_shootable"])
            if contact["done"]:
                done = True
                reason = str(contact["reason"])
                break
        skill = SKILL_ACTIONS[action_index]
        route_outcome = _route_outcome(skill, previous, current, decision=decision)
        combat_action_reward = self._combat_action_reward(skill, had_shootable_target)
        route_action_reward = self._route_action_reward(route_outcome)
        action_reward = combat_action_reward + route_action_reward
        total_reward += action_reward
        self._current_state = current
        if hasattr(self.controller, "record_action_history"):
            self.controller.record_action_history(
                action_index=action_index,
                had_shootable_target=had_shootable_target,
                route_outcome=route_outcome,
            )
        info = {
            "skill": skill,
            "action_index": action_index,
            "decision_cycle": {
                "schema": DECISION_CYCLE_SCHEMA["schema"],
                "observation_schema": OBSERVATION_SCHEMA["schema"],
                "action_schema": ACTION_SCHEMA["schema"],
                "memory_contract": MEMORY_CONTRACT["schema"],
                "input_tick": int(getattr(previous, "tick", 0)),
                "output_tick": int(getattr(current, "tick", 0)),
                "macro_tics": len(transition_summaries),
            },
            "decision": decision,
            "transition": _combine_transition_summaries(transition_summaries),
            "macro_tics": len(transition_summaries),
            "action_reward": action_reward,
            "combat_action_reward": combat_action_reward,
            "route_action_reward": route_action_reward,
            "contact_reward": contact_reward,
            "visible_contact_distance_delta": round(visible_contact_distance_delta, 4),
            "visible_contact_progress_reward": round(visible_contact_progress_reward, 4),
            "had_visible_enemy": had_visible_enemy,
            "route_outcome": route_outcome,
            "had_shootable_target": had_shootable_target,
            "first_visible_contact": first_visible_contact,
            "first_shootable_contact": first_shootable_contact,
            "action_mask_filter": dict(self._last_action_mask_filter),
            "reset_warmup": dict(self._last_reset_warmup),
            "reset_context": dict(self._last_reset_context),
            "state": summarize_state(current),
            "done_reason": reason,
        }
        if self.config.curriculum is not None:
            info["curriculum"] = dict(self.config.curriculum)
        if self.config.curriculum_stage is not None:
            info["curriculum_stage"] = dict(self.config.curriculum_stage)
        return EnvStep(
            observation=self.controller.observation(current),
            reward=total_reward,
            done=done,
            info=info,
        )

    def action_mask(self) -> list[bool]:
        """Returns feasible PPO actions for the current state."""
        if self._current_state is None:
            return self._filter_allowed_skills([True for _ in SKILL_ACTIONS])
        if hasattr(self.controller, "action_mask"):
            return self._filter_allowed_skills(
                list(self.controller.action_mask(self._current_state))
            )
        return self._filter_allowed_skills([True for _ in SKILL_ACTIONS])

    def _filter_allowed_skills(self, mask: list[bool]) -> list[bool]:
        allowed_skills = tuple(self.config.allowed_skills or ())
        self._last_action_mask_filter = {}
        if not allowed_skills:
            return mask
        allowed = set(allowed_skills)
        filtered = [
            bool(value) and skill in allowed
            for skill, value in zip(SKILL_ACTIONS, mask, strict=False)
        ]
        info: dict[str, Any] = {
            "schema": "restfuldoom.allowed_skill_filter.v1",
            "configured": True,
            "strict": bool(self.config.strict_allowed_skills),
            "allowed_skills": list(allowed_skills),
            "raw_allowed_count": sum(1 for value in mask if value),
            "filtered_allowed_count": sum(1 for value in filtered if value),
            "fallback_applied": False,
            "fallback_skill": None,
        }
        if any(filtered):
            self._last_action_mask_filter = info
            return filtered
        if not self.config.strict_allowed_skills:
            info["fallback_applied"] = True
            info["fallback_skill"] = "unfiltered_mask"
            self._last_action_mask_filter = info
            return mask
        for skill in allowed_skills:
            if skill in SKILL_ACTIONS:
                fallback = [False for _ in SKILL_ACTIONS]
                fallback[SKILL_ACTIONS.index(skill)] = True
                info["fallback_applied"] = True
                info["fallback_skill"] = skill
                info["filtered_allowed_count"] = 1
                self._last_action_mask_filter = info
                return fallback
        self._last_action_mask_filter = info
        return filtered

    def _combat_action_reward(self, skill: str, had_shootable_target: bool) -> float:
        if had_shootable_target and skill == "fire":
            return self.config.shootable_fire_reward
        if had_shootable_target and skill != "fire":
            return -self.config.missed_fire_penalty
        if skill == "fire":
            return -self.config.blind_fire_penalty
        return 0.0

    def _route_action_reward(self, route_outcome: dict[str, Any]) -> float:
        if not route_outcome.get("attempted"):
            return 0.0
        progress_units = float(route_outcome.get("progress_units") or 0.0)
        reward = progress_units * self.config.route_progress_reward
        reward = max(-1.0, min(1.0, reward))
        if route_outcome.get("exit"):
            exit_reward = progress_units * self.config.exit_route_progress_reward
            reward += max(-1.0, min(1.0, exit_reward))
        if route_outcome.get("reached"):
            reward += self.config.route_reached_reward
            if route_outcome.get("exit"):
                reward += self.config.exit_route_reached_reward
        if route_outcome.get("failed"):
            reward -= self.config.route_failure_penalty
            if route_outcome.get("exit"):
                reward -= self.config.exit_route_failure_penalty
        return reward

    def _contact_reward(self, current: Any) -> dict[str, Any]:
        visible_now = _has_visible_enemy(current)
        shootable_now = _has_shootable_enemy(current)
        first_visible = visible_now and not self._episode_seen_visible_enemy
        first_shootable = shootable_now and not self._episode_seen_shootable_enemy
        reward = 0.0
        done = False
        reason = None
        if first_visible:
            reward += self.config.first_visible_bonus
            self._episode_seen_visible_enemy = True
            if self.config.terminate_on_first_visible:
                done = True
                reason = "first_visible_enemy"
        if first_shootable:
            reward += self.config.first_shootable_bonus
            self._episode_seen_shootable_enemy = True
            if self.config.terminate_on_first_shootable:
                done = True
                reason = "first_shootable_target"
        return {
            "reward": reward,
            "done": done,
            "reason": reason,
            "first_visible": first_visible,
            "first_shootable": first_shootable,
        }

    def _visible_contact_progress_reward(self, distance_delta: float) -> float:
        if self.config.visible_contact_progress_reward == 0.0:
            return 0.0
        reward = distance_delta * self.config.visible_contact_progress_reward
        return max(-1.0, min(1.0, reward))

    async def _ensure_stream(self) -> None:
        await self._ensure_client()
        assert self.client is not None
        if self._state_stream is None:
            self._action_queue = asyncio.Queue(maxsize=16)
            self._state_stream = self.client.session(self._action_iter()).__aiter__()

    async def _ensure_client(self) -> None:
        if self.client is None:
            self.client = DoomAgentClient(
                self.config.endpoint,
                token=self.config.token,
                agent_port=self.config.agent_port,
                tls=self.config.tls,
                authority=self.config.authority,
            )

    async def _action_iter(self) -> AsyncIterator[Any]:
        yield agent_pb2.PlayerAction()
        assert self._action_queue is not None
        while True:
            action = await self._action_queue.get()
            if action is None:
                break
            yield action

    async def _next_reset_state(self, episode: int, map_number: int) -> Any:
        assert self._state_stream is not None
        deadline = time.monotonic() + self.config.reset_timeout_seconds
        last_state = None
        while time.monotonic() < deadline:
            state = await asyncio.wait_for(
                self._state_stream.__anext__(),
                timeout=max(0.1, deadline - time.monotonic()),
            )
            last_state = state
            if (
                state.level.episode == episode
                and state.level.map == map_number
                and state.level.level_time <= 5
                and state.player.health > 0
            ):
                return state
        raise TimeoutError(
            "timed out waiting for reset state"
            + (f"; last_state={summarize_state(last_state)}" if last_state else "")
        )

    async def _next_live_state(self, episode: int, map_number: int) -> Any:
        assert self._state_stream is not None
        deadline = time.monotonic() + self.config.reset_timeout_seconds
        last_state = None
        while time.monotonic() < deadline:
            state = await asyncio.wait_for(
                self._state_stream.__anext__(),
                timeout=max(0.1, deadline - time.monotonic()),
            )
            last_state = state
            if (
                state.level.episode == episode
                and state.level.map == map_number
                and state.player.health > 0
            ):
                return state
        raise TimeoutError(
            "timed out waiting for snapshot-restored state"
            + (f"; last_state={summarize_state(last_state)}" if last_state else "")
        )

    async def _run_reset_warmup(self, state: Any) -> Any:
        """Runs heuristic-only curriculum steps before PPO starts collecting."""
        if self._action_queue is None or self._state_stream is None:
            return state
        current = state
        warmup_info: dict[str, Any] = {
            "enabled": True,
            "steps": 0,
            "tics": 0,
            "stop_reason": "limit",
            "episode_index": self._episode_index,
        }
        start_level = (state.level.episode, state.level.map)
        for _ in range(self.config.reset_warmup_steps):
            if self.config.reset_warmup_until_shootable and _has_shootable_enemy(current):
                warmup_info["stop_reason"] = "shootable"
                self._last_reset_warmup = warmup_info
                return current
            if self.config.reset_warmup_until_visible and _has_visible_enemy(current):
                warmup_info["stop_reason"] = "visible"
                self._last_reset_warmup = warmup_info
                break
            action_index = self.controller.heuristic_action_index(current)
            action, _decision = self.controller.action_for(action_index, current)
            await self._action_queue.put(action)
            warmup_info["steps"] = int(warmup_info["steps"]) + 1
            action_tics = max(
                1,
                min(
                    self.config.max_action_tics,
                    int(getattr(action, "duration_tics", 0) or 1),
                ),
            )
            for _tick in range(action_tics):
                current = await self._state_stream.__anext__()
                warmup_info["tics"] = int(warmup_info["tics"]) + 1
                if (
                    self.config.reset_warmup_max_tics > 0
                    and int(warmup_info["tics"]) >= self.config.reset_warmup_max_tics
                ):
                    warmup_info["stop_reason"] = "tic_limit"
                    self._last_reset_warmup = warmup_info
                    return current
                if current.player.health <= 0:
                    warmup_info["stop_reason"] = "death"
                    self._last_reset_warmup = warmup_info
                    return current
                if (current.level.episode, current.level.map) != start_level:
                    warmup_info["stop_reason"] = "level_changed"
                    self._last_reset_warmup = warmup_info
                    return current
                if self.config.reset_warmup_until_shootable and _has_shootable_enemy(current):
                    warmup_info["stop_reason"] = "shootable"
                    self._last_reset_warmup = warmup_info
                    return current
                if self.config.reset_warmup_until_visible and _has_visible_enemy(current):
                    warmup_info["stop_reason"] = "visible"
                    self._last_reset_warmup = warmup_info
                    return current
        self._last_reset_warmup = warmup_info
        return current

    def _terminal_reward(
        self,
        previous: Any,
        current: Any,
        transition: TransitionReward,
    ) -> tuple[bool, str | None, float]:
        reward = float(transition.reward)
        start_level = self._start_level or (previous.level.episode, previous.level.map)
        current_level = (current.level.episode, current.level.map)
        if current.player.health <= 0:
            return True, "death", reward
        if current_level != start_level:
            reward += self.config.level_complete_bonus
            if current.player.kills - self._start_kills >= self.config.required_kills:
                reward += self.config.kill_goal_bonus
            return True, "level_complete", reward
        if (
            self.config.terminate_on_required_kills
            and current.player.kills - self._start_kills >= self.config.required_kills
        ):
            reward += self.config.kill_goal_bonus
            return True, "required_kills", reward
        if self._steps >= self.config.max_steps:
            return True, "max_steps", reward
        return False, None, reward


def _transition_summary(transition: TransitionReward) -> dict[str, Any]:
    return {
        "reward": transition.reward,
        "kill_delta": transition.kill_delta,
        "damage_delta": transition.damage_delta,
        "enemy_distance_delta": transition.enemy_distance_delta,
        "item_delta": transition.item_delta,
        "secret_delta": transition.secret_delta,
        "health_delta": transition.health_delta,
        "progress_delta": transition.progress_delta,
        "done": transition.done,
    }


def _combine_transition_summaries(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not transitions:
        return {
            "reward": 0.0,
            "kill_delta": 0,
            "damage_delta": 0,
            "enemy_distance_delta": 0.0,
            "item_delta": 0,
            "secret_delta": 0,
            "health_delta": 0,
            "progress_delta": 0.0,
            "done": False,
        }
    return {
        "reward": sum(float(transition["reward"]) for transition in transitions),
        "kill_delta": sum(int(transition["kill_delta"]) for transition in transitions),
        "damage_delta": sum(int(transition["damage_delta"]) for transition in transitions),
        "enemy_distance_delta": sum(
            float(transition["enemy_distance_delta"]) for transition in transitions
        ),
        "item_delta": sum(int(transition["item_delta"]) for transition in transitions),
        "secret_delta": sum(int(transition["secret_delta"]) for transition in transitions),
        "health_delta": sum(int(transition["health_delta"]) for transition in transitions),
        "progress_delta": sum(float(transition["progress_delta"]) for transition in transitions),
        "done": any(bool(transition["done"]) for transition in transitions),
    }


def _has_shootable_enemy(state: Any) -> bool:
    combat = getattr(state, "combat", None)
    if combat is None:
        return False
    return bool(
        getattr(combat, "has_shootable_target", False)
        and getattr(combat, "target_is_enemy", False)
    )


def _snapshot_slot(snapshot: dict[str, Any]) -> int | None:
    value = snapshot.get("slot")
    if value is None:
        ref = snapshot.get("ref")
        if isinstance(ref, str) and ref.startswith("save_slot:"):
            value = ref.removeprefix("save_slot:")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _has_visible_enemy(state: Any) -> bool:
    for enemy in getattr(state, "enemies", []) or []:
        if bool(getattr(enemy, "line_of_sight", False)):
            return True
    return False


def _visible_contact_distance_delta(previous: Any, current: Any) -> float:
    """Returns positive Doom-unit distance gained toward a visible, non-shootable enemy."""
    if _has_shootable_enemy(previous) or not _has_visible_enemy(previous):
        return 0.0
    target = _nearest_visible_enemy(previous)
    if target is None:
        return 0.0
    target_id, previous_distance = target
    if target_id <= 0 or previous_distance <= 0.0:
        return 0.0
    current_distance = _enemy_distance_by_id(current, target_id)
    if current_distance <= 0.0:
        return 0.0
    return previous_distance - current_distance


def _nearest_visible_enemy(state: Any) -> tuple[int, float] | None:
    candidates: list[tuple[float, int]] = []
    for enemy in getattr(state, "enemies", []) or []:
        if not bool(getattr(enemy, "line_of_sight", False)):
            continue
        obj = getattr(enemy, "object", None)
        if obj is None or int(getattr(obj, "health", 0)) <= 0:
            continue
        enemy_id = int(getattr(obj, "id", 0))
        distance = _enemy_distance_units(state, enemy)
        if enemy_id > 0 and distance > 0.0:
            candidates.append((distance, enemy_id))
    if not candidates:
        return None
    distance, enemy_id = min(candidates, key=lambda item: item[0])
    return enemy_id, distance


def _enemy_distance_by_id(state: Any, enemy_id: int) -> float:
    for enemy in getattr(state, "enemies", []) or []:
        obj = getattr(enemy, "object", None)
        if obj is None or int(getattr(obj, "id", 0)) != enemy_id:
            continue
        if int(getattr(obj, "health", 0)) <= 0:
            return 0.0
        return _enemy_distance_units(state, enemy)
    return 0.0


def _enemy_distance_units(state: Any, enemy: Any) -> float:
    obj = getattr(enemy, "object", None)
    if obj is None:
        return 0.0
    distance_fp = float(getattr(obj, "distance_fp", 0) or 0)
    if distance_fp > 0.0:
        return distance_fp / 65536.0
    player_x, player_y = _player_xy_units(state)
    position = getattr(obj, "position", None)
    if position is None:
        return 0.0
    enemy_x = float(getattr(position, "x_fp", 0)) / 65536.0
    enemy_y = float(getattr(position, "y_fp", 0)) / 65536.0
    return ((player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2) ** 0.5


def _route_outcome(
    skill: str,
    previous: Any,
    current: Any,
    *,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarizes whether a route-progression macro-step helped."""
    route, line = _route_target_for_outcome(previous, decision)
    line_id = int(_line_value(line, "line_id", 0)) if line is not None else 0
    outcome: dict[str, Any] = {
        "attempted": skill == "route_progression" and line is not None,
        "line_id": line_id,
        "previous_distance_units": 0.0,
        "current_distance_units": 0.0,
        "progress_units": 0.0,
        "reached": False,
        "failed": False,
        "priority": int(_line_value(route, "priority", 0)) if route is not None else 0,
        "exit": bool(_line_value(route, "exit", False)) if route is not None else False,
        "walk_trigger": (
            bool(_line_value(route, "walk_trigger", False)) if route is not None else False
        ),
        "target_source": _line_value(route, "source", "route_waypoint")
        if route is not None
        else "",
    }
    if not outcome["attempted"]:
        return outcome

    previous_x, previous_y = _player_xy_units(previous)
    current_x, current_y = _player_xy_units(current)
    previous_distance = _distance_to_line_units(
        line,
        previous_x,
        previous_y,
        fallback_fp=_line_value(line, "nearest_distance_fp", 0),
        fallback_units=_line_value(line, "distance", 0.0),
    )
    current_distance = _distance_to_line_units(
        line,
        current_x,
        current_y,
        fallback_units=_line_value(line, "distance", 0.0),
    )
    progress = previous_distance - current_distance
    current_line_id = _route_waypoint_line_id(current)
    line_changed = current_line_id not in (0, line_id)
    reached = current_distance <= 48.0 or (line_changed and previous_distance <= 128.0)
    failed = not reached and progress <= 4.0
    outcome.update(
        {
            "previous_distance_units": round(previous_distance, 4),
            "current_distance_units": round(current_distance, 4),
            "progress_units": round(progress, 4),
            "reached": reached,
            "failed": failed,
            "line_changed": line_changed,
            "current_line_id": current_line_id,
        }
    )
    return outcome


def _route_target_for_outcome(
    state: Any,
    decision: dict[str, Any] | None,
) -> tuple[Any | None, Any | None]:
    if isinstance(decision, dict):
        selected_line = decision.get("use_line")
        if isinstance(selected_line, dict) and int(selected_line.get("line_id", 0)) > 0:
            selected_line_id = int(selected_line["line_id"])
            waypoint_route, waypoint_line = _route_waypoint(state)
            full_line = _find_navigation_line(state, selected_line_id)
            if (
                waypoint_line is not None
                and int(_line_value(waypoint_line, "line_id", 0)) == selected_line_id
            ):
                return waypoint_route, full_line if full_line is not None else waypoint_line
            route = {
                "priority": 0,
                "exit": _line_is_exit(selected_line),
                "walk_trigger": int(selected_line.get("special", 0)) not in EXIT_LINE_SPECIALS,
                "source": "decision_line",
            }
            return route, full_line if full_line is not None else selected_line
    return _route_waypoint(state)


def _find_navigation_line(state: Any, line_id: int) -> Any | None:
    navigation = getattr(state, "navigation", None)
    for line in getattr(navigation, "use_lines", []) or []:
        if int(_line_value(line, "line_id", 0)) == int(line_id):
            return line
    return None


def _route_waypoint(state: Any) -> tuple[Any | None, Any | None]:
    navigation = getattr(state, "navigation", None)
    route = getattr(navigation, "route_waypoint", None) if navigation is not None else None
    line = getattr(route, "line", None) if route is not None else None
    if line is None or int(getattr(line, "line_id", 0)) <= 0:
        return None, None
    return route, line


def _route_waypoint_line_id(state: Any) -> int:
    _route, line = _route_waypoint(state)
    if line is None:
        return 0
    return int(getattr(line, "line_id", 0))


def _player_xy_units(state: Any) -> tuple[float, float]:
    player = getattr(state, "player", None)
    obj = getattr(player, "object", None) if player is not None else None
    position = getattr(obj, "position", None) if obj is not None else None
    if position is None:
        return 0.0, 0.0
    return (
        float(getattr(position, "x_fp", 0)) / 65536.0,
        float(getattr(position, "y_fp", 0)) / 65536.0,
    )


def _verify_snapshot_restored_state(
    *,
    actual: dict[str, Any],
    expected: dict[str, Any],
    raw_state: Any,
    enabled: bool,
    tick_tolerance: int,
    verify_stream_tick: bool,
    position_tolerance_fp: int,
) -> dict[str, Any]:
    """Verifies that a restored snapshot produced the expected first state."""
    verification: dict[str, Any] = {
        "schema": "restfuldoom.snapshot_restored_state_verification.v1",
        "enabled": bool(enabled),
        "valid": True,
        "skipped": False,
        "errors": [],
        "compared_fields": [],
        "tolerances": {
            "tick": int(tick_tolerance),
            "position_fp": int(position_tolerance_fp),
        },
        "stream_tick_checked": bool(verify_stream_tick),
    }
    if not enabled:
        verification["skipped"] = True
        return verification
    if not expected:
        verification["skipped"] = True
        return verification

    errors: list[str] = []
    compared: list[str] = []
    for key in ("episode", "map", "health", "armor", "kills", "items", "secrets"):
        if key not in expected:
            continue
        compared.append(key)
        actual_value = actual.get(key)
        expected_value = expected.get(key)
        if _optional_int(actual_value) != _optional_int(expected_value):
            errors.append(
                f"{key} expected {expected_value!r} got {actual_value!r}"
            )

    if "level_time" in expected:
        compared.append("level_time")
        actual_level_time = _optional_int(actual.get("level_time"))
        expected_level_time = _optional_int(expected.get("level_time"))
        if actual_level_time is None or expected_level_time is None:
            errors.append(
                "level_time expected "
                f"{expected.get('level_time')!r} got {actual.get('level_time')!r}"
            )
        elif abs(actual_level_time - expected_level_time) > tick_tolerance:
            errors.append(
                "level_time drift "
                f"{abs(actual_level_time - expected_level_time)} exceeds tolerance "
                f"{tick_tolerance} (expected {expected_level_time}, got {actual_level_time})"
            )

    if verify_stream_tick and "tick" in expected:
        compared.append("tick")
        actual_tick = _optional_int(actual.get("tick"))
        expected_tick = _optional_int(expected.get("tick"))
        if actual_tick is None or expected_tick is None:
            errors.append(
                f"tick expected {expected.get('tick')!r} got {actual.get('tick')!r}"
            )
        elif abs(actual_tick - expected_tick) > tick_tolerance:
            errors.append(
                "tick drift "
                f"{abs(actual_tick - expected_tick)} exceeds tolerance {tick_tolerance} "
                f"(expected {expected_tick}, got {actual_tick})"
            )

    expected_position = expected.get("position_fp")
    actual_position = actual.get("position_fp")
    if isinstance(expected_position, list) and len(expected_position) >= 2:
        compared.append("position_fp")
        if not isinstance(actual_position, list) or len(actual_position) < 2:
            errors.append(
                f"position_fp expected {expected_position!r} got {actual_position!r}"
            )
        else:
            dx = abs(int(actual_position[0]) - int(expected_position[0]))
            dy = abs(int(actual_position[1]) - int(expected_position[1]))
            if dx > position_tolerance_fp or dy > position_tolerance_fp:
                errors.append(
                    "position_fp drift exceeds tolerance "
                    f"{position_tolerance_fp} (dx={dx}, dy={dy})"
                )

    if "visible_enemy" in expected:
        compared.append("visible_enemy")
        actual_visible = _has_visible_enemy(raw_state)
        if actual_visible != bool(expected.get("visible_enemy")):
            errors.append(
                "visible_enemy expected "
                f"{bool(expected.get('visible_enemy'))!r} got {actual_visible!r}"
            )

    if "shootable_target" in expected:
        compared.append("shootable_target")
        actual_shootable = _combat_flag(actual, raw_state, "has_shootable_target")
        if actual_shootable is None:
            actual_shootable = _has_shootable_enemy(raw_state)
        if actual_shootable != bool(expected.get("shootable_target")):
            errors.append(
                "shootable_target expected "
                f"{bool(expected.get('shootable_target'))!r} got {actual_shootable!r}"
            )

    if "target_is_enemy" in expected:
        compared.append("target_is_enemy")
        actual_target_is_enemy = _combat_flag(actual, raw_state, "target_is_enemy")
        if actual_target_is_enemy is None:
            errors.append(
                "target_is_enemy expected "
                f"{bool(expected.get('target_is_enemy'))!r} got None"
            )
        elif actual_target_is_enemy != bool(expected.get("target_is_enemy")):
            errors.append(
                "target_is_enemy expected "
                f"{bool(expected.get('target_is_enemy'))!r} got {actual_target_is_enemy!r}"
            )

    if "route_waypoint_exit" in expected:
        compared.append("route_waypoint_exit")
        actual_route_exit = _route_waypoint_flag(actual, raw_state, "exit")
        if actual_route_exit is None:
            errors.append(
                "route_waypoint_exit expected "
                f"{bool(expected.get('route_waypoint_exit'))!r} got None"
            )
        elif actual_route_exit != bool(expected.get("route_waypoint_exit")):
            errors.append(
                "route_waypoint_exit expected "
                f"{bool(expected.get('route_waypoint_exit'))!r} got {actual_route_exit!r}"
            )

    if "route_waypoint_line_id" in expected:
        compared.append("route_waypoint_line_id")
        actual_line_id = _snapshot_route_waypoint_line_id(actual, raw_state)
        expected_line_id = _optional_int(expected.get("route_waypoint_line_id"))
        if actual_line_id != expected_line_id:
            errors.append(
                "route_waypoint_line_id expected "
                f"{expected_line_id!r} got {actual_line_id!r}"
            )

    verification["compared_fields"] = compared
    verification["errors"] = errors
    verification["valid"] = not errors
    return verification


def _combat_flag(actual: dict[str, Any], raw_state: Any, key: str) -> bool | None:
    combat = actual.get("combat")
    if isinstance(combat, dict) and key in combat:
        return bool(combat.get(key))
    raw_combat = getattr(raw_state, "combat", None)
    if raw_combat is not None and hasattr(raw_combat, key):
        return bool(getattr(raw_combat, key))
    return None


def _route_waypoint_flag(actual: dict[str, Any], raw_state: Any, key: str) -> bool | None:
    route = _route_waypoint_dict(actual)
    if key in route:
        return bool(route.get(key))
    raw_route = getattr(getattr(raw_state, "navigation", None), "route_waypoint", None)
    if raw_route is not None and hasattr(raw_route, key):
        return bool(getattr(raw_route, key))
    return None


def _snapshot_route_waypoint_line_id(actual: dict[str, Any], raw_state: Any) -> int | None:
    route = _route_waypoint_dict(actual)
    line = route.get("line")
    if isinstance(line, dict) and "line_id" in line:
        return _optional_int(line.get("line_id"))
    raw_route = getattr(getattr(raw_state, "navigation", None), "route_waypoint", None)
    raw_line = getattr(raw_route, "line", None)
    if raw_line is not None and hasattr(raw_line, "line_id"):
        return _optional_int(getattr(raw_line, "line_id"))
    return None


def _route_waypoint_dict(actual: dict[str, Any]) -> dict[str, Any]:
    navigation = actual.get("navigation")
    if isinstance(navigation, dict):
        route = navigation.get("route_waypoint")
        if isinstance(route, dict):
            return route
    return {}


def _optional_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _distance_to_line_units(
    line: Any,
    x_units: float,
    y_units: float,
    *,
    fallback_fp: int | float = 0,
    fallback_units: int | float = 0,
) -> float:
    start = _line_value(line, "start")
    end = _line_value(line, "end")
    if start is None or end is None:
        units = float(fallback_units or 0.0)
        if units > 0.0:
            return units
        fallback = float(fallback_fp or 0.0) / 65536.0
        if fallback > 0.0:
            return fallback
        midpoint = _line_value(line, "midpoint")
        if midpoint is None:
            return 0.0
        return (
            (x_units - float(_line_value(midpoint, "x_fp", 0)) / 65536.0) ** 2
            + (y_units - float(_line_value(midpoint, "y_fp", 0)) / 65536.0) ** 2
        ) ** 0.5

    x1 = float(_line_value(start, "x_fp", 0)) / 65536.0
    y1 = float(_line_value(start, "y_fp", 0)) / 65536.0
    x2 = float(_line_value(end, "x_fp", 0)) / 65536.0
    y2 = float(_line_value(end, "y_fp", 0)) / 65536.0
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return ((x_units - x1) ** 2 + (y_units - y1) ** 2) ** 0.5
    t = ((x_units - x1) * dx + (y_units - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return ((x_units - nearest_x) ** 2 + (y_units - nearest_y) ** 2) ** 0.5


def _line_value(line: Any, key: str, default: Any = None) -> Any:
    if isinstance(line, dict):
        return line.get(key, default)
    return getattr(line, key, default)
