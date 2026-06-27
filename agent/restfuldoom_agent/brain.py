"""Structured local Doom agent brain.

The brain is intentionally small and fast: it consumes protobuf `GameState`
messages, maintains persistent tactical memory, chooses a local skill every tic,
and records enough episode evidence to evolve policy parameters over time.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import random
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .client import BackoffConfig, DoomAgentClient, agent_pb2, semantic_action
from .reward import RewardEngine, goal_preset
from .rollout_config import safe_endpoint_host
from .schemas import ACTION_SCHEMA, OBSERVATION_SCHEMA
from .skill_policy import SkillPolicyModel, SkillPolicyTrainConfig, train_skill_policy

FP = 65536.0
CELL_UNITS = 128.0
MEMORY_SCHEMA = "restfuldoom.agent_memory.v1"
BT_ATTACK = 1
BT_USE = 2
USE_LINE_ACTIVATE_DISTANCE_UNITS = 160.0
EXIT_LINE_USE_DISTANCE_UNITS = 96.0
EXIT_ASSIST_DISTANCE_UNITS = 900.0
EXIT_ASSIST_DOOR_USE_DISTANCE_UNITS = 384.0
EXIT_ASSIST_DOOR_EXIT_DISTANCE_UNITS = 420.0
EXIT_ASSIST_DOOR_CLOSE_USE_DISTANCE_UNITS = 96.0
EXIT_ASSIST_DOOR_CLOSE_USE_ANGLE_DEGREES = 48.0
LINE_ATTEMPT_STALL_TICS = 45
LINE_ATTEMPT_BLOCK_TICS = 1200
# Manual door/switch specials handled by Doom's P_UseSpecialLine path.
MANUAL_USE_LINE_SPECIALS = frozenset(
    {
        1,
        7,
        9,
        11,
        14,
        15,
        18,
        20,
        21,
        23,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        41,
        42,
        43,
        45,
        49,
        50,
        51,
        55,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        101,
        102,
        103,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        122,
        123,
        127,
        131,
        132,
        133,
        135,
        137,
        140,
    }
)
PROGRESSION_LINE_PRIORITIES = {
    88: 0,  # Walk-trigger platform that opens progression on E1M1.
    36: 1,  # Walk-trigger turbo lower floor.
    11: 2,  # Exit switch.
    51: 2,  # Secret exit switch.
}
EXIT_LINE_SPECIALS = frozenset({11, 51})
WALK_TRIGGER_LINE_SPECIALS = frozenset({36, 88})
_NON_LOCOMOTION_SKILLS = {
    "aim_at_enemy",
    "dead",
    "defensive_fire",
    "fire_on_enemy",
    "hold_attack",
    "open_or_probe",
    "strafe_cooldown",
    "turn_from_block",
    "unstick_turn",
    "use_ahead",
    "use_blocking_line",
}


@dataclass(frozen=True)
class BrainConfig:
    """Configures one or more structured-brain episodes."""

    endpoint: str = "127.0.0.1:50051"
    token: str | None = None
    agent_port: int = 50051
    tls: bool = False
    authority: str | None = None
    goal_preset: str = "combat"
    mission: str = "survive, explore, and kill visible enemies"
    max_states: int = 700
    memory_path: Path = Path("agent_memory/e1m1.json")
    trajectory_jsonl: Path | None = Path("trajectories/brain.jsonl")
    skill_model_path: Path | None = None
    evolve_runs: int = 1
    seed: int = 7
    policy_id: str = "cautious_combat_v1"
    stop_on_death: bool = True
    required_kills: int = 1
    require_level_complete: bool = True
    reconnect: bool = True
    max_reconnects: int = 5


@dataclass(frozen=True)
class BrainPolicyParams:
    """Small tunable parameter set for skill selection and control."""

    aim_tolerance_degrees: float = 9.0
    close_enemy_units: float = 220.0
    retreat_health: int = 35
    move_amount: int = 28
    retreat_amount: int = 26
    turn_amount: int = 12
    fine_turn_amount: int = 5
    strafe_amount: int = 22
    shoot_tolerance_degrees: float = 3.0
    shoot_cooldown_tics: int = 10
    use_interval_tics: int = 35
    stuck_window_tics: int = 18
    stuck_distance_units: float = 4.0
    explore_turn_interval_tics: int = 24
    enemy_memory_tics: int = 350

    def bounded(self) -> "BrainPolicyParams":
        """Clamp parameters after mutation."""
        return BrainPolicyParams(
            aim_tolerance_degrees=_clamp_float(self.aim_tolerance_degrees, 3.0, 22.0),
            close_enemy_units=_clamp_float(self.close_enemy_units, 96.0, 420.0),
            retreat_health=_clamp_int(self.retreat_health, 10, 80),
            move_amount=_clamp_int(self.move_amount, 12, 50),
            retreat_amount=_clamp_int(self.retreat_amount, 10, 50),
            turn_amount=_clamp_int(self.turn_amount, 5, 32),
            fine_turn_amount=_clamp_int(self.fine_turn_amount, 2, 15),
            strafe_amount=_clamp_int(self.strafe_amount, 8, 45),
            shoot_tolerance_degrees=_clamp_float(self.shoot_tolerance_degrees, 1.0, 8.0),
            shoot_cooldown_tics=_clamp_int(self.shoot_cooldown_tics, 1, 16),
            use_interval_tics=_clamp_int(self.use_interval_tics, 12, 80),
            stuck_window_tics=_clamp_int(self.stuck_window_tics, 8, 45),
            stuck_distance_units=_clamp_float(self.stuck_distance_units, 1.0, 16.0),
            explore_turn_interval_tics=_clamp_int(self.explore_turn_interval_tics, 10, 70),
            enemy_memory_tics=_clamp_int(self.enemy_memory_tics, 80, 900),
        )

    def mutate(self, rng: random.Random, scale: float = 1.0) -> "BrainPolicyParams":
        """Return a nearby parameter set for evolutionary rollout search."""
        return BrainPolicyParams(
            aim_tolerance_degrees=self.aim_tolerance_degrees + rng.uniform(-4.0, 4.0) * scale,
            close_enemy_units=self.close_enemy_units + rng.uniform(-80.0, 80.0) * scale,
            retreat_health=self.retreat_health + rng.randint(-15, 15),
            move_amount=self.move_amount + rng.randint(-8, 8),
            retreat_amount=self.retreat_amount + rng.randint(-8, 8),
            turn_amount=self.turn_amount + rng.randint(-5, 5),
            fine_turn_amount=self.fine_turn_amount + rng.randint(-3, 3),
            strafe_amount=self.strafe_amount + rng.randint(-8, 8),
            shoot_tolerance_degrees=self.shoot_tolerance_degrees + rng.uniform(-1.5, 1.5) * scale,
            shoot_cooldown_tics=self.shoot_cooldown_tics + rng.randint(-4, 3),
            use_interval_tics=self.use_interval_tics + rng.randint(-12, 12),
            stuck_window_tics=self.stuck_window_tics + rng.randint(-6, 8),
            stuck_distance_units=self.stuck_distance_units + rng.uniform(-2.0, 3.0) * scale,
            explore_turn_interval_tics=self.explore_turn_interval_tics + rng.randint(-8, 12),
            enemy_memory_tics=self.enemy_memory_tics + rng.randint(-120, 160),
        ).bounded()


@dataclass(frozen=True)
class TacticalFeatures:
    """Compact current-state view used by the policy."""

    tick: int
    x_units: float
    y_units: float
    angle: float
    health: int
    ammo_bullets: int
    kills: int
    items: int
    secrets: int
    cell: str
    visible_enemies: list[dict[str, Any]]
    known_enemies: list[dict[str, Any]]
    remembered_enemies: list[dict[str, Any]]
    enemy_count: int
    navigation: dict[str, Any]
    combat: dict[str, Any]
    episode: int
    map: int


@dataclass
class EpisodeStats:
    """Mutable per-episode metrics."""

    run_id: str
    candidate_id: str
    policy_id: str
    goal: str
    states: int = 0
    total_reward: float = 0.0
    start_tick: int | None = None
    end_tick: int | None = None
    start_episode: int | None = None
    start_map: int | None = None
    end_episode: int | None = None
    end_map: int | None = None
    start_health: int | None = None
    end_health: int | None = None
    start_kills: int = 0
    end_kills: int = 0
    peak_kills: int = 0
    start_items: int = 0
    end_items: int = 0
    peak_items: int = 0
    deaths: int = 0
    stuck_events: int = 0
    enemy_damage: int = 0
    opened_or_used: int = 0
    visible_enemy_ticks: int = 0
    nearest_enemy_start: float | None = None
    nearest_enemy_end: float | None = None
    nearest_enemy_min: float | None = None
    visited_cells: set[str] = field(default_factory=set)
    level_completed: bool = False
    skill_counts: dict[str, int] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)

    def kill_delta(self) -> int:
        """Return the best observed kill delta, preserving progress across map resets."""
        return max(self.end_kills, self.peak_kills) - self.start_kills

    def item_delta(self) -> int:
        """Return the best observed item delta, preserving progress across map resets."""
        return max(self.end_items, self.peak_items) - self.start_items

    def succeeded(self, *, required_kills: int, require_level_complete: bool) -> bool:
        """Return whether this episode reached the configured good-state target."""
        kill_ok = self.kill_delta() >= required_kills
        level_ok = self.level_completed or not require_level_complete
        return kill_ok and level_ok

    def score(self) -> float:
        """Fitness score used for policy promotion."""
        kill_delta = self.kill_delta()
        item_delta = self.item_delta()
        health = self.end_health if self.end_health is not None else 0
        survival_bonus = 10.0 if self.deaths == 0 and self.states > 0 else 0.0
        return (
            self.total_reward
            + kill_delta * 20.0
            + item_delta * 3.0
            + len(self.visited_cells) * 2.0
            + self.enemy_damage * 0.35
            + self.visible_enemy_ticks * 0.2
            + self.enemy_distance_progress() * 0.05
            + max(0, health) * 0.05
            + survival_bonus
            - self.stuck_events * 1.5
            - self.deaths * 50.0
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe episode summary."""
        return {
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "policy_id": self.policy_id,
            "goal": self.goal,
            "states": self.states,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "start_episode": self.start_episode,
            "start_map": self.start_map,
            "end_episode": self.end_episode,
            "end_map": self.end_map,
            "total_reward": round(self.total_reward, 4),
            "fitness": round(self.score(), 4),
            "start_health": self.start_health,
            "end_health": self.end_health,
            "kill_delta": self.kill_delta(),
            "item_delta": self.item_delta(),
            "peak_kills": self.peak_kills,
            "peak_items": self.peak_items,
            "deaths": self.deaths,
            "stuck_events": self.stuck_events,
            "enemy_damage": self.enemy_damage,
            "visited_cells": len(self.visited_cells),
            "visible_enemy_ticks": self.visible_enemy_ticks,
            "nearest_enemy_start": self.nearest_enemy_start,
            "nearest_enemy_end": self.nearest_enemy_end,
            "nearest_enemy_min": self.nearest_enemy_min,
            "enemy_distance_progress": round(self.enemy_distance_progress(), 4),
            "opened_or_used": self.opened_or_used,
            "level_completed": self.level_completed,
            "skill_counts": dict(sorted(self.skill_counts.items())),
            "lessons": list(self.lessons),
        }

    def observe_nearest_enemy(self, distance: float | None) -> None:
        """Track whether the policy is making progress toward enemies."""
        if distance is None:
            return
        if self.nearest_enemy_start is None:
            self.nearest_enemy_start = distance
        self.nearest_enemy_end = distance
        if self.nearest_enemy_min is None or distance < self.nearest_enemy_min:
            self.nearest_enemy_min = distance

    def enemy_distance_progress(self) -> float:
        """Return positive progress toward the nearest known enemy."""
        if self.nearest_enemy_start is None or self.nearest_enemy_min is None:
            return 0.0
        return max(0.0, self.nearest_enemy_start - self.nearest_enemy_min)


class AgentMemory:
    """Persistent cross-run world and policy memory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data = self._load()

    @classmethod
    def load(cls, path: str | Path) -> "AgentMemory":
        """Load or create memory."""
        return cls(Path(path))

    def best_params(self) -> BrainPolicyParams:
        """Return the currently promoted policy parameters."""
        params = self.data.get("policy", {}).get("best_params")
        if isinstance(params, dict):
            try:
                merged = asdict(BrainPolicyParams())
                merged.update(params)
                return BrainPolicyParams(**merged).bounded()
            except TypeError:
                pass
        return BrainPolicyParams()

    def record_step(
        self,
        features: TacticalFeatures,
        decision: dict[str, Any],
        reward: Any,
        stats: EpisodeStats,
    ) -> None:
        """Persist compact observations that help future runs."""
        now = _iso_now()
        cells = self.data.setdefault("cells", {})
        cell = cells.setdefault(
            features.cell,
            {
                "first_seen_tick": features.tick,
                "visits": 0,
                "enemy_sightings": 0,
                "damage_events": 0,
                "last_seen_at": now,
            },
        )
        cell["visits"] = int(cell.get("visits", 0)) + 1
        cell["last_seen_tick"] = features.tick
        cell["last_seen_at"] = now
        if features.visible_enemies:
            cell["enemy_sightings"] = int(cell.get("enemy_sightings", 0)) + 1
            stats.visible_enemy_ticks += 1
        if reward.health_delta < 0:
            cell["damage_events"] = int(cell.get("damage_events", 0)) + 1
        stats.visited_cells.add(features.cell)

        enemies = self.data.setdefault("enemies", {})
        for enemy in features.known_enemies:
            enemy_id = str(enemy["id"])
            entry = enemies.setdefault(
                enemy_id,
                {
                    "first_seen_tick": features.tick,
                    "visible_count": 0,
                    "max_threat": 0.0,
                },
            )
            entry["last_seen_tick"] = features.tick
            entry["last_position"] = [round(enemy["x"], 2), round(enemy["y"], 2)]
            entry["last_distance"] = round(enemy["distance"], 2)
            entry["last_health"] = enemy["health"]
            entry["line_of_sight"] = bool(enemy["line_of_sight"])
            entry["visible_count"] = int(entry.get("visible_count", 0)) + 1
            entry["max_threat"] = max(float(entry.get("max_threat", 0.0)), enemy["threat"])

        if decision.get("stuck"):
            stats.stuck_events += 1
        if decision.get("skill") == "open_or_probe":
            stats.opened_or_used += 1

    def finish_episode(
        self,
        *,
        stats: EpisodeStats,
        params: BrainPolicyParams,
        promoted: bool,
    ) -> dict[str, Any]:
        """Append a summarized episode and update policy memory."""
        summary = stats.summary()
        summary["params"] = asdict(params)
        summary["promoted"] = promoted

        episodes = self.data.setdefault("episodes", [])
        episodes.append(summary)
        del episodes[:-50]

        policy = self.data.setdefault("policy", {})
        previous_best = float(policy.get("best_score", -1e12))
        if promoted:
            policy["best_score"] = summary["fitness"]
            policy["best_params"] = asdict(params)
            policy["best_run_id"] = stats.run_id
            policy["promoted_at"] = _iso_now()
        else:
            policy["best_score"] = previous_best
        policy["last_score"] = summary["fitness"]
        policy["last_success"] = summary.get("success", False)
        policy["last_params"] = asdict(params)
        policy["generations"] = int(policy.get("generations", 0)) + 1
        policy["updated_at"] = _iso_now()

        lessons = self.data.setdefault("lessons", [])
        for lesson in stats.lessons:
            lessons.append({"run_id": stats.run_id, "lesson": lesson, "at": _iso_now()})
        del lessons[:-100]

        self.data["updated_at"] = _iso_now()
        self.save()
        return summary

    def should_promote(self, score: float) -> bool:
        """Return whether an episode score beats known policy memory."""
        best = float(self.data.get("policy", {}).get("best_score", -1e12))
        return score >= best

    def summary(self) -> dict[str, Any]:
        """Return compact memory diagnostics."""
        policy = self.data.get("policy", {})
        episodes = self.data.get("episodes", [])
        return {
            "schema": self.data.get("schema"),
            "path": str(self.path),
            "episodes": len(episodes),
            "cells": len(self.data.get("cells", {})),
            "enemies": len(self.data.get("enemies", {})),
            "lessons": len(self.data.get("lessons", [])),
            "best_score": policy.get("best_score"),
            "best_run_id": policy.get("best_run_id"),
            "generations": policy.get("generations", 0),
            "last_episode": episodes[-1] if episodes else None,
        }

    def save(self) -> None:
        """Persist memory to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            now = _iso_now()
            return {
                "schema": MEMORY_SCHEMA,
                "created_at": now,
                "updated_at": now,
                "policy": {
                    "best_score": -1e12,
                    "best_params": asdict(BrainPolicyParams()),
                    "generations": 0,
                },
                "cells": {},
                "enemies": {},
                "episodes": [],
                "lessons": [],
            }

        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if loaded.get("schema") != MEMORY_SCHEMA:
            raise ValueError(
                f"expected memory schema {MEMORY_SCHEMA}, got {loaded.get('schema')!r}"
            )
        return loaded


class BrainPolicy:
    """Fast local skill-selection policy backed by persistent memory."""

    def __init__(
        self,
        *,
        memory: AgentMemory,
        params: BrainPolicyParams,
        policy_id: str,
        skill_model: SkillPolicyModel | None = None,
    ) -> None:
        self.memory = memory
        self.params = params
        self.policy_id = policy_id
        self.skill_model = skill_model
        self.last_features: TacticalFeatures | None = None
        self.last_decision: dict[str, Any] = {}
        self._last_position: tuple[float, float] | None = None
        self._last_progress_tick = 0
        self._last_shot_tick = -9999
        self._last_use_tick = -9999
        self._last_stuck_phase = -1
        self._explore_bias = 1
        self._blocked_enemy_cells: dict[str, int] = {}
        self._blocked_use_lines: dict[str, int] = {}
        self._line_attempts: dict[str, dict[str, Any]] = {}
        self._exit_push_attempts: dict[str, dict[str, Any]] = {}
        self._episode_cell_visits: dict[str, int] = {}
        self._start_kills: int | None = None
        self._last_visible_enemy_tick = -9999
        self._last_visible_enemy_id: int | None = None
        self._last_contact_ray: dict[str, Any] | None = None
        self._hazard_escape: dict[str, Any] | None = None

    async def next_action(self, state: Any) -> Any:
        """Return a fast local action for the current protobuf state."""
        features = extract_features(state, self.memory, self.params)
        self.last_features = features
        action, decision = self._decide(features)
        if self.skill_model is not None:
            try:
                decision = dict(decision)
                decision["learned_skill_prediction"] = self.skill_model.predict_tactical(features)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                decision = dict(decision)
                decision["learned_skill_error"] = str(exc)
        self.last_decision = decision
        return action

    def _decide(self, features: TacticalFeatures) -> tuple[Any, dict[str, Any]]:
        if self._start_kills is None:
            self._start_kills = features.kills
        self._episode_cell_visits[features.cell] = (
            self._episode_cell_visits.get(features.cell, 0) + 1
        )
        if features.visible_enemies:
            self._last_visible_enemy_tick = features.tick
            self._last_visible_enemy_id = int(features.visible_enemies[0]["id"])

        stuck = self._is_stuck(features)
        visible = features.visible_enemies
        close_enemy = visible[0] if visible else None

        if features.health <= 0:
            return agent_pb2.PlayerAction(), self._decision("dead", features, stuck=stuck)

        shootable_enemy = self._shootable_enemy(features)
        if features.health <= 15 and (close_enemy is not None or shootable_enemy is not None):
            return self._retreat_or_fire(
                features,
                close_enemy if close_enemy is not None else shootable_enemy,
                stuck,
            )

        if shootable_enemy is not None and self._can_shoot(features, shootable_enemy):
            self._last_shot_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_ATTACK,
                    angle_turn=raw_turn_for_delta(shootable_enemy["angle_delta"]),
                    duration_tics=1,
                    tick=features.tick,
                ),
                self._decision(
                    "fire_on_shootable_target",
                    features,
                    enemy=shootable_enemy,
                    stuck=stuck,
                ),
            )

        if close_enemy and (
            features.health <= self.params.retreat_health
            or close_enemy["distance"] <= self.params.close_enemy_units
        ):
            return self._retreat_or_fire(features, close_enemy, stuck)

        if close_enemy:
            return self._engage(features, close_enemy, stuck)

        local_exit_line = self._select_local_exit_line(features)
        if (
            local_exit_line is not None
            and (
                not stuck
                or self._line_control_distance(local_exit_line) <= 224.0
                or not features.navigation["forward_open"]
            )
        ):
            if not features.navigation["forward_open"]:
                assist_door = self._select_exit_assist_door(features, local_exit_line)
                if assist_door is not None:
                    return self._use_nearby_line(features, assist_door, stuck)
                blocking_line = self._select_stuck_manual_line(features)
                if (
                    blocking_line is not None
                    and int(blocking_line.get("special", 0)) not in EXIT_LINE_SPECIALS
                ):
                    return self._use_nearby_line(features, blocking_line, stuck)
            return self._advance_progression_line(features, local_exit_line, stuck)

        hazard_escape = self._escape_hazard_cell(features, stuck)
        if hazard_escape is not None:
            return hazard_escape

        if stuck:
            stuck_line = self._select_stuck_manual_line(features)
            if stuck_line is not None:
                return self._use_nearby_line(features, stuck_line, stuck)
            progression_line = self._select_progression_line(features)
            if progression_line is not None:
                return self._advance_progression_line(features, progression_line, stuck)
            self._mark_blocked_target(features)
            return self._recover_from_stuck(features)

        use_ray = self._select_use_ray(features)
        if use_ray is not None:
            return self._use_directional_line(features, use_ray, stuck)

        progression_line = self._select_progression_line(features)
        if progression_line is not None:
            return self._advance_progression_line(features, progression_line, stuck)

        use_line = self._select_nearby_use_line(features)
        if use_line is not None:
            return self._use_nearby_line(features, use_line, stuck)

        if self._should_use_ahead(features):
            self._last_use_tick = features.tick
            return (
                semantic_action(agent_pb2.ACTION_USE, duration_tics=2, tick=features.tick),
                self._decision("use_ahead", features, stuck=stuck),
            )

        if self._should_probe(features):
            self._last_use_tick = features.tick
            return (
                semantic_action(agent_pb2.ACTION_USE, duration_tics=2, tick=features.tick),
                self._decision("open_or_probe", features, stuck=stuck),
            )

        contact = self._continue_last_contact_corridor(features, stuck)
        if contact is not None:
            return contact

        enemy = self._select_known_enemy(features)
        if enemy is not None and self._should_seek_known_enemy(features, enemy):
            return self._turn_toward_or_move(
                features,
                enemy["angle_delta"],
                "seek_known_enemy",
                stuck,
                enemy=enemy,
            )

        if enemy is not None and self._should_hunt_known_enemy(features, enemy):
            return self._turn_toward_or_move(
                features,
                enemy["angle_delta"],
                "hunt_known_enemy",
                stuck,
                enemy=enemy,
            )

        return self._explore(features, stuck)

    def _recover_from_stuck(self, features: TacticalFeatures) -> tuple[Any, dict[str, Any]]:
        elapsed = max(0, features.tick - self._last_progress_tick)
        phase = (elapsed // 8) % 5
        self._last_stuck_phase = phase

        if phase == 0 and features.tick - self._last_use_tick >= self.params.use_interval_tics:
            self._last_use_tick = features.tick
            return (
                semantic_action(agent_pb2.ACTION_USE, duration_tics=2, tick=features.tick),
                self._decision("unstick_use", features, stuck=True, stuck_phase=phase),
            )

        if phase == 1:
            return (
                semantic_action(
                    agent_pb2.ACTION_BACKWARD,
                    amount=self.params.retreat_amount,
                    duration_tics=4,
                    tick=features.tick,
                ),
                self._decision("unstick_backtrack", features, stuck=True, stuck_phase=phase),
            )

        if phase == 2:
            return (
                raw_ticcmd_action(
                    forward_move=8,
                    angle_turn=-640 if self._explore_bias > 0 else 640,
                    duration_tics=4,
                    tick=features.tick,
                ),
                self._decision("unstick_turn", features, stuck=True, stuck_phase=phase),
            )

        if phase == 3:
            return (
                semantic_action(
                    agent_pb2.ACTION_STRAFE_RIGHT if self._explore_bias > 0 else agent_pb2.ACTION_STRAFE_LEFT,
                    amount=self.params.strafe_amount,
                    duration_tics=4,
                    tick=features.tick,
                ),
                self._decision("unstick_strafe", features, stuck=True, stuck_phase=phase),
            )

        self._explore_bias *= -1
        return (
            raw_ticcmd_action(
                forward_move=max(12, self.params.move_amount - 6),
                side_move=self._explore_bias * max(10, self.params.strafe_amount // 2),
                duration_tics=4,
                tick=features.tick,
            ),
            self._decision("unstick_forward", features, stuck=True, stuck_phase=phase),
        )

    def _escape_hazard_cell(
        self,
        features: TacticalFeatures,
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]] | None:
        if not self._is_hazard_cell(features):
            self._hazard_escape = None
            return None

        target_line = self._select_local_exit_line(features)
        if target_line is None and features.health > 20:
            target_line = self._select_hazard_progression_line(features)
        if target_line is not None:
            angle_delta = self._line_control_angle_delta(target_line)
            ray = self._best_navigation_ray(features, angle_delta)
            line_record = self._line_record(target_line)
            if ray is not None:
                return self._move_on_ray_toward_line(
                    features,
                    angle_delta,
                    ray,
                    "escape_hazard_toward_progression",
                    stuck,
                    line_record,
                )
            if abs(angle_delta) <= 18 and features.navigation["forward_open"]:
                return (
                    raw_ticcmd_action(
                        forward_move=self.params.move_amount,
                        angle_turn=raw_turn_for_delta(angle_delta),
                        duration_tics=4,
                        tick=features.tick,
                    ),
                    self._decision(
                        "escape_hazard_toward_progression",
                        features,
                        stuck=stuck,
                        use_line=line_record,
                    ),
                )

        active = self._hazard_escape
        if (
            active is None
            or active.get("cell") != features.cell
            or int(active.get("until_tick", -1)) < features.tick
        ):
            ray = self._select_hazard_escape_ray(features)
            if ray is None:
                return self._recover_from_stuck(features)
            self._hazard_escape = {
                "cell": features.cell,
                "offset": float(ray["angle_offset_degrees"]),
                "until_tick": features.tick + 48,
            }
        else:
            ray = self._ray_for_offset(features, float(active["offset"]))
            if ray is None:
                ray = self._select_hazard_escape_ray(features)
                if ray is None:
                    return self._recover_from_stuck(features)
                self._hazard_escape = {
                    "cell": features.cell,
                    "offset": float(ray["angle_offset_degrees"]),
                    "until_tick": features.tick + 48,
                }

        return self._move_on_ray(
            features,
            float(ray["angle_offset_degrees"]),
            ray,
            "escape_hazard_cell",
            stuck,
        )

    def _is_hazard_cell(self, features: TacticalFeatures) -> bool:
        if features.visible_enemies or features.known_enemies:
            return False
        cell = self.memory.data.get("cells", {}).get(features.cell, {})
        damage_events = int(cell.get("damage_events", 0))
        if damage_events <= 0:
            return False
        return features.health <= 55

    def _select_hazard_progression_line(
        self,
        features: TacticalFeatures,
    ) -> dict[str, Any] | None:
        candidates = [
            line
            for line in features.navigation.get("use_lines", [])
            if int(line.get("special", 0)) in PROGRESSION_LINE_PRIORITIES
            and float(line.get("distance", 999999)) <= 2600.0
            and abs(float(line.get("angle_delta", 999))) <= 150
            and self._is_progression_line_ready(features, line)
            and not self._is_line_blocked(features, line)
        ]
        if not candidates:
            return None
        advanced = [
            line
            for line in candidates
            if not (
                int(line.get("special", 0)) in WALK_TRIGGER_LINE_SPECIALS
                and float(line.get("distance", 999999)) <= 128.0
            )
        ]
        if advanced:
            candidates = advanced
        return min(
            candidates,
            key=lambda line: (
                self._progression_line_priority(line),
                float(line["distance"]) / 384.0,
                abs(float(line["angle_delta"])) / 45.0,
            ),
        )

    def _select_hazard_escape_ray(self, features: TacticalFeatures) -> dict[str, Any] | None:
        probes = features.navigation.get("direction_probes") or []
        open_probes = [probe for probe in probes if probe.get("open")]
        if not open_probes:
            return None
        return max(
            open_probes,
            key=lambda probe: (
                not bool(probe.get("use_line_ahead")),
                int(probe.get("block_distance_fp", 0)),
                -abs(float(probe.get("angle_offset_degrees", 0))),
            ),
        )

    def _ray_for_offset(
        self,
        features: TacticalFeatures,
        offset: float,
    ) -> dict[str, Any] | None:
        for probe in features.navigation.get("direction_probes") or []:
            if probe.get("open") and float(probe.get("angle_offset_degrees", 0)) == offset:
                return probe
        return None

    def _engage(
        self,
        features: TacticalFeatures,
        enemy: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        angle_delta = enemy["angle_delta"]
        if enemy["distance"] > 1800:
            ray = self._best_navigation_ray(features, angle_delta)
            if ray is not None and abs(angle_delta) <= 90:
                return self._move_on_ray(
                    features,
                    angle_delta,
                    ray,
                    "close_visible_enemy",
                    stuck,
                    enemy=enemy,
                )
            if not features.navigation["forward_open"] and abs(angle_delta) <= 55:
                return self._skirt_visible_enemy(features, enemy, stuck)
            if abs(angle_delta) > 30:
                amount = (
                    self.params.fine_turn_amount
                    if abs(angle_delta) < 25
                    else self.params.turn_amount
                )
                return (
                    semantic_action(
                        turn_action_for_delta(angle_delta),
                        amount=amount,
                        duration_tics=1,
                        tick=features.tick,
                    ),
                    self._decision("aim_at_enemy", features, enemy=enemy, stuck=stuck),
                )
            if not features.navigation["forward_open"]:
                return self._avoid_blocked_front(features, "close_visible_enemy")
            return (
                raw_ticcmd_action(
                    forward_move=self.params.move_amount,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision("close_visible_enemy", features, enemy=enemy, stuck=stuck),
            )

        shoot_tolerance = min(
            self.params.aim_tolerance_degrees,
            self.params.shoot_tolerance_degrees,
        )
        if abs(angle_delta) > shoot_tolerance:
            amount = self.params.fine_turn_amount if abs(angle_delta) < 25 else self.params.turn_amount
            action_type = turn_action_for_delta(angle_delta)
            return (
                semantic_action(action_type, amount=amount, duration_tics=1, tick=features.tick),
                self._decision("aim_at_enemy", features, enemy=enemy, stuck=stuck),
            )

        if self._can_shoot(features, enemy):
            self._last_shot_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_ATTACK,
                    forward_move=8
                    if enemy["distance"] > 420 and features.navigation["forward_open"]
                    else 0,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=1,
                    tick=features.tick,
                ),
                self._decision("fire_on_enemy", features, enemy=enemy, stuck=stuck),
            )

        return (
            raw_ticcmd_action(
                buttons=BT_ATTACK,
                side_move=(
                    self.params.strafe_amount
                    if (features.tick // 12) % 2
                    else -self.params.strafe_amount
                ),
                angle_turn=raw_turn_for_delta(angle_delta),
                duration_tics=1,
                tick=features.tick,
            ),
            self._decision("hold_attack", features, enemy=enemy, stuck=stuck),
            )

    def _best_navigation_ray(
        self,
        features: TacticalFeatures,
        preferred_angle_delta: float,
    ) -> dict[str, Any] | None:
        probes = features.navigation.get("direction_probes") or []
        open_probes = [probe for probe in probes if probe.get("open")]
        if not open_probes:
            return None
        return min(
            open_probes,
            key=lambda probe: (
                abs(float(probe["angle_offset_degrees"]) - preferred_angle_delta),
                abs(float(probe["angle_offset_degrees"])),
            ),
        )

    def _move_on_ray(
        self,
        features: TacticalFeatures,
        aim_angle_delta: float,
        ray: dict[str, Any],
        skill: str,
        stuck: bool,
        *,
        enemy: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        offset = float(ray["angle_offset_degrees"])
        radians = math.radians(offset)
        forward = _clamp_int(
            round(self.params.move_amount * max(0.2, math.cos(radians))),
            4,
            self.params.move_amount,
        )
        side = _clamp_int(
            round(-math.sin(radians) * self.params.move_amount),
            -self.params.move_amount,
            self.params.move_amount,
        )
        if skill == "close_visible_enemy" or (enemy is not None and enemy.get("line_of_sight")):
            self._remember_contact_ray(features, enemy, offset)
        return (
            raw_ticcmd_action(
                forward_move=forward,
                side_move=side,
                angle_turn=raw_turn_for_delta(aim_angle_delta),
                duration_tics=3,
                tick=features.tick,
            ),
            self._decision(
                skill,
                features,
                enemy=enemy,
                stuck=stuck,
                direction_probe={
                    "angle_offset_degrees": int(offset),
                    "block_distance_fp": int(ray.get("block_distance_fp", 0)),
                },
            ),
        )

    def _continue_last_contact_corridor(
        self,
        features: TacticalFeatures,
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]] | None:
        contact = self._last_contact_ray
        if contact is None:
            return None
        age = features.tick - int(contact["tick"])
        if age < 0 or age > 90:
            return None
        if self._episode_cell_visits.get(features.cell, 0) > max(
            16, self.params.stuck_window_tics * 2
        ):
            return None

        preferred = float(contact["ray_offset"])
        ray = self._best_navigation_ray(features, preferred)
        if ray is None:
            return None

        enemy = None
        enemy_id = int(contact["enemy_id"])
        for known_enemy in features.known_enemies:
            if int(known_enemy["id"]) == enemy_id:
                enemy = known_enemy
                break

        aim_angle = (
            enemy["angle_delta"]
            if enemy is not None and abs(enemy["angle_delta"]) <= 90
            else preferred
        )
        return self._move_on_ray(
            features,
            aim_angle,
            ray,
            "pursue_last_contact_corridor",
            stuck,
            enemy=enemy,
        )

    def _skirt_visible_enemy(
        self,
        features: TacticalFeatures,
        enemy: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        nav = features.navigation
        angle_delta = enemy["angle_delta"]
        if nav["right_open"] and not nav["left_open"]:
            side = self.params.strafe_amount
            ray_offset = -90
        elif nav["left_open"] and not nav["right_open"]:
            side = -self.params.strafe_amount
            ray_offset = 90
        else:
            side = self._explore_bias * max(8, self.params.strafe_amount // 2)
            ray_offset = -45 if side > 0 else 45
        self._remember_contact_ray(features, enemy, ray_offset)
        return (
            raw_ticcmd_action(
                forward_move=max(6, self.params.move_amount // 3),
                side_move=side,
                angle_turn=raw_turn_for_delta(angle_delta),
                duration_tics=3,
                tick=features.tick,
            ),
            self._decision("skirt_visible_enemy", features, enemy=enemy, stuck=stuck),
        )

    def _remember_contact_ray(
        self,
        features: TacticalFeatures,
        enemy: dict[str, Any] | None,
        ray_offset: float,
    ) -> None:
        if enemy is None:
            return
        self._last_contact_ray = {
            "tick": features.tick,
            "enemy_id": int(enemy["id"]),
            "ray_offset": float(ray_offset),
        }

    def _retreat_or_fire(
        self,
        features: TacticalFeatures,
        enemy: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        if features.health <= 15:
            if (
                abs(enemy["angle_delta"]) <= self.params.shoot_tolerance_degrees
                and self._can_shoot(features, enemy)
            ):
                self._last_shot_tick = features.tick
                return (
                    raw_ticcmd_action(
                        buttons=BT_ATTACK,
                        forward_move=-max(8, self.params.retreat_amount // 2),
                        side_move=(
                            self.params.strafe_amount
                            if (features.tick // 10) % 2
                            else -self.params.strafe_amount
                        ),
                        angle_turn=raw_turn_for_delta(enemy["angle_delta"]),
                        duration_tics=1,
                        tick=features.tick,
                    ),
                    self._decision(
                        "critical_defensive_fire",
                        features,
                        enemy=enemy,
                        stuck=stuck,
                    ),
                )
            return (
                raw_ticcmd_action(
                    forward_move=-self.params.retreat_amount,
                    side_move=(
                        self.params.strafe_amount
                        if (features.tick // 10) % 2
                        else -self.params.strafe_amount
                    ),
                    angle_turn=raw_turn_for_delta(enemy["angle_delta"]),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision("critical_retreat", features, enemy=enemy, stuck=stuck),
            )
        if (
            abs(enemy["angle_delta"])
            <= min(self.params.aim_tolerance_degrees, self.params.shoot_tolerance_degrees)
            and self._can_shoot(features, enemy)
        ):
            self._last_shot_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_ATTACK,
                    angle_turn=raw_turn_for_delta(enemy["angle_delta"]),
                    duration_tics=1,
                    tick=features.tick,
                ),
                self._decision("defensive_fire", features, enemy=enemy, stuck=stuck),
            )
        return (
            semantic_action(
                agent_pb2.ACTION_BACKWARD,
                amount=self.params.retreat_amount,
                duration_tics=3,
                tick=features.tick,
            ),
            self._decision("retreat", features, enemy=enemy, stuck=stuck),
        )

    def _turn_toward_or_move(
        self,
        features: TacticalFeatures,
        angle_delta: float,
        skill: str,
        stuck: bool,
        enemy: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if abs(angle_delta) > 18:
            return (
                semantic_action(
                    turn_action_for_delta(angle_delta),
                    amount=self.params.turn_amount,
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    skill,
                    features,
                    angle_delta=round(angle_delta, 2),
                    stuck=stuck,
                    enemy=enemy,
                ),
            )
        if not features.navigation["forward_open"]:
            return self._avoid_blocked_front(features, skill)

        return (
            semantic_action(
                agent_pb2.ACTION_FORWARD,
                amount=self.params.move_amount,
                duration_tics=3,
                tick=features.tick,
            ),
            self._decision(
                skill,
                features,
                angle_delta=round(angle_delta, 2),
                stuck=stuck,
                enemy=enemy,
            ),
        )

    def _explore(self, features: TacticalFeatures, stuck: bool) -> tuple[Any, dict[str, Any]]:
        if not features.navigation["forward_open"]:
            return self._avoid_blocked_front(features, "explore_blocked_front")

        visits = self._episode_cell_visits.get(features.cell, 0)
        nav = features.navigation
        turn = 0
        side = 0
        skill = "explore_frontier"

        if visits > max(20, self.params.stuck_window_tics * 2):
            self._explore_bias *= -1
            turn = -768 if self._explore_bias > 0 else 768
            side = self._explore_bias * max(8, self.params.strafe_amount // 2)
            skill = "break_cell_loop"
        elif not nav["left_open"] and nav["right_open"]:
            turn = -384
            side = max(8, self.params.strafe_amount // 2)
            skill = "wall_follow_right"
        elif not nav["right_open"] and nav["left_open"]:
            turn = 384
            side = -max(8, self.params.strafe_amount // 2)
            skill = "wall_follow_left"
        elif (features.tick // self.params.explore_turn_interval_tics) % 5 == 0:
            turn = -256 if self._explore_bias > 0 else 256
            skill = "sweep_frontier"

        if turn or side:
            return (
                raw_ticcmd_action(
                    forward_move=self.params.move_amount,
                    side_move=side,
                    angle_turn=turn,
                    duration_tics=4,
                    tick=features.tick,
                ),
                self._decision(skill, features, stuck=stuck, cell_visits=visits),
            )

        return (
            raw_ticcmd_action(
                forward_move=self.params.move_amount,
                duration_tics=4,
                tick=features.tick,
            ),
            self._decision("explore_frontier", features, stuck=stuck, cell_visits=visits),
        )

    def _is_stuck(self, features: TacticalFeatures) -> bool:
        current = (features.x_units, features.y_units)
        if self._last_position is None:
            self._last_position = current
            self._last_progress_tick = features.tick
            return False

        if self.last_decision.get("skill") in _NON_LOCOMOTION_SKILLS:
            self._last_position = current
            self._last_progress_tick = features.tick
            return False

        moved = math.dist(current, self._last_position)
        if moved >= self.params.stuck_distance_units:
            self._last_position = current
            self._last_progress_tick = features.tick
            return False

        return features.tick - self._last_progress_tick >= self.params.stuck_window_tics

    def _should_probe(self, features: TacticalFeatures) -> bool:
        if features.tick - self._last_use_tick < self.params.use_interval_tics:
            return False
        if features.navigation["use_line_ahead"] and not features.navigation["forward_open"]:
            return True
        cell = self.memory.data.get("cells", {}).get(features.cell, {})
        visits = int(cell.get("visits", 0))
        local_visits = self._episode_cell_visits.get(features.cell, 0)
        return (
            not features.navigation["forward_open"]
            and visits > 8
            and local_visits > self.params.stuck_window_tics
        )

    def _should_use_ahead(self, features: TacticalFeatures) -> bool:
        if features.tick - self._last_use_tick < self.params.use_interval_tics:
            return False
        return bool(features.navigation["use_line_ahead"])

    def _select_use_ray(self, features: TacticalFeatures) -> dict[str, Any] | None:
        if features.tick - self._last_use_tick < max(8, self.params.use_interval_tics // 2):
            return None
        probes = [
            probe
            for probe in features.navigation.get("direction_probes", [])
            if probe.get("use_line_ahead")
        ]
        if not probes:
            return None
        return min(
            probes,
            key=lambda probe: (
                abs(float(probe["angle_offset_degrees"])),
                -int(probe.get("block_distance_fp", 0)),
            ),
        )

    def _select_exit_side_use_ray(self, features: TacticalFeatures) -> dict[str, Any] | None:
        if features.tick - self._last_use_tick < max(8, self.params.use_interval_tics // 2):
            return None
        probes = [
            probe
            for probe in features.navigation.get("direction_probes", [])
            if probe.get("use_line_ahead")
            and int(probe.get("blocking_line_special", 0)) in MANUAL_USE_LINE_SPECIALS
        ]
        if not probes:
            return None
        return min(
            probes,
            key=lambda probe: (
                abs(float(probe["angle_offset_degrees"])),
                -int(probe.get("block_distance_fp", 0)),
            ),
        )

    def _select_nearby_use_line(self, features: TacticalFeatures) -> dict[str, Any] | None:
        if features.tick - self._last_use_tick < max(8, self.params.use_interval_tics // 2):
            return None
        max_distance = 512
        has_local_exit = self._has_local_exit_line(features)
        candidates = []
        for line in features.navigation.get("use_lines", []):
            special = int(line.get("special", 0))
            distance = float(line.get("distance", 999999))
            if special not in MANUAL_USE_LINE_SPECIALS:
                continue
            if has_local_exit and special == 1 and distance > EXIT_ASSIST_DOOR_USE_DISTANCE_UNITS:
                continue
            if distance > max_distance:
                continue
            if abs(float(line.get("angle_delta", 999))) > 135:
                continue
            if self._is_line_blocked(features, line):
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                abs(float(line["angle_delta"])),
                float(line["distance"]),
            ),
        )

    def _select_stuck_manual_line(self, features: TacticalFeatures) -> dict[str, Any] | None:
        if features.tick - self._last_use_tick < max(8, self.params.use_interval_tics // 2):
            return None
        candidates = [
            line
            for line in features.navigation.get("use_lines", [])
            if int(line.get("special", 0)) in MANUAL_USE_LINE_SPECIALS
            and float(line.get("distance", 999999)) <= 192.0
            and abs(float(line.get("angle_delta", 999))) <= 60.0
            and not self._is_line_blocked(features, line)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                0 if bool(features.navigation.get("use_line_ahead")) else 1,
                float(line["distance"]),
                abs(float(line["angle_delta"])),
            ),
        )

    def _select_local_exit_line(self, features: TacticalFeatures) -> dict[str, Any] | None:
        if not self._has_episode_kill(features):
            return None
        candidates = [
            line
            for line in features.navigation.get("use_lines", [])
            if int(line.get("special", 0)) in EXIT_LINE_SPECIALS
            and float(line.get("distance", 999999)) <= EXIT_ASSIST_DISTANCE_UNITS
            and abs(float(line.get("angle_delta", 999))) <= 150
            and not self._is_line_blocked(features, line)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                float(line["distance"]),
                abs(float(line["angle_delta"])),
            ),
        )

    def _select_progression_line(
        self,
        features: TacticalFeatures,
    ) -> dict[str, Any] | None:
        if not self._has_episode_kill(features):
            return None
        max_distance = 2600.0
        candidates = [
            line
            for line in features.navigation.get("use_lines", [])
            if int(line.get("special", 0)) in PROGRESSION_LINE_PRIORITIES
            and float(line.get("distance", 999999)) <= max_distance
            and abs(float(line.get("angle_delta", 999))) <= 150
            and self._is_progression_line_ready(features, line)
            and not self._is_line_blocked(features, line)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                self._progression_line_priority(line),
                float(line["distance"]) / 384.0,
                abs(float(line["angle_delta"])) / 45.0,
            ),
        )

    def _is_progression_line_ready(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
    ) -> bool:
        special = int(line.get("special", 0))
        start_kills = self._start_kills if self._start_kills is not None else features.kills
        kill_delta = features.kills - start_kills
        if (
            special in WALK_TRIGGER_LINE_SPECIALS
            and float(line.get("distance", 999999)) > 1200.0
        ):
            return kill_delta >= 5
        if special not in EXIT_LINE_SPECIALS:
            return True
        if float(line.get("distance", 999999)) <= EXIT_ASSIST_DISTANCE_UNITS:
            return True
        return kill_delta >= 5

    def _advance_progression_line(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._record_line_attempt(features, line):
            return self._explore(features, stuck)

        angle_delta = self._line_control_angle_delta(line)
        distance = self._line_control_distance(line)
        line_record = self._line_record(line)
        special = int(line["special"])
        if special in EXIT_LINE_SPECIALS and distance > 224.0:
            assist_door = self._select_exit_assist_door(features, line)
            if assist_door is not None:
                return self._use_nearby_line(features, assist_door, stuck)

        activate_distance = self._line_activate_distance(features, line)
        if (
            special in EXIT_LINE_SPECIALS
            and int(line.get("side", 0)) == 0
            and distance > activate_distance
            and (
                distance <= 224.0
                or float(line.get("front_distance", 999999)) <= 160.0
            )
        ):
            exit_push_stalled = self._exit_push_stalled(features, line, distance)
            if exit_push_stalled:
                retry_door = self._select_retry_exit_assist_door(features, line)
                if retry_door is not None:
                    return self._use_exit_assist_retry(features, retry_door, stuck)
            if not features.navigation["forward_open"] and not exit_push_stalled:
                side_use_ray = self._select_exit_side_use_ray(features)
                if side_use_ray is not None:
                    return self._use_directional_line(features, side_use_ray, stuck)
            if exit_push_stalled and features.navigation["back_open"]:
                return (
                    raw_ticcmd_action(
                        forward_move=-self.params.retreat_amount,
                        angle_turn=raw_turn_for_delta(angle_delta),
                        duration_tics=4,
                        tick=features.tick,
                    ),
                    self._decision(
                        "backtrack_from_exit_switch",
                        features,
                        stuck=stuck,
                        use_line=line_record,
                    ),
                )
            if abs(angle_delta) > 30:
                return (
                    semantic_action(
                        turn_action_for_delta(angle_delta),
                        amount=self.params.turn_amount,
                        duration_tics=2,
                        tick=features.tick,
                    ),
                    self._decision(
                        "turn_to_exit_switch",
                        features,
                        stuck=stuck,
                        use_line=line_record,
                    ),
                )
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    forward_move=self.params.move_amount,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision(
                    "push_exit_switch",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )

        if (
            special in EXIT_LINE_SPECIALS
            and int(line.get("side", 0)) == 0
            and distance <= activate_distance
        ):
            if abs(angle_delta) > 18:
                return (
                    semantic_action(
                        turn_action_for_delta(angle_delta),
                        amount=self.params.turn_amount,
                        duration_tics=2,
                        tick=features.tick,
                    ),
                    self._decision(
                        "turn_to_exit_switch",
                        features,
                        stuck=stuck,
                        use_line=line_record,
                    ),
                )
            self._last_use_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    forward_move=max(4, self.params.move_amount // 2),
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision(
                    "press_exit_switch",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )

        if not features.navigation["forward_open"]:
            ray = self._best_navigation_ray(features, angle_delta)
            if ray is not None:
                return self._move_on_ray_toward_line(
                    features,
                    angle_delta,
                    ray,
                    "route_to_progression_line",
                    stuck,
                    line_record,
                )
            if special not in WALK_TRIGGER_LINE_SPECIALS:
                return self._avoid_blocked_front(features, "approach_progression_line")

        if abs(angle_delta) > 18:
            return (
                semantic_action(
                    turn_action_for_delta(angle_delta),
                    amount=self.params.turn_amount,
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    "turn_to_progression_line",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )

        if (
            special in EXIT_LINE_SPECIALS
            and int(line.get("side", 0)) == 1
            and abs(angle_delta) <= 18
        ):
            return (
                raw_ticcmd_action(
                    forward_move=self.params.move_amount,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=4,
                    tick=features.tick,
                ),
                self._decision(
                    "approach_progression_line_front",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        if (
            special in MANUAL_USE_LINE_SPECIALS
            and (special not in EXIT_LINE_SPECIALS or int(line.get("side", 0)) == 0)
            and distance <= activate_distance
            and abs(angle_delta) <= 12
        ):
            self._last_use_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    forward_move=max(4, self.params.move_amount // 2),
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    "use_progression_line",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )

        return (
            raw_ticcmd_action(
                forward_move=self.params.move_amount,
                angle_turn=raw_turn_for_delta(angle_delta),
                duration_tics=4,
                tick=features.tick,
            ),
            self._decision(
                "cross_progression_line"
                if special in WALK_TRIGGER_LINE_SPECIALS
                else "approach_progression_line",
                features,
                stuck=stuck,
                use_line=line_record,
            ),
        )

    def _select_retry_exit_assist_door(
        self,
        features: TacticalFeatures,
        exit_line: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidates = []
        for line in features.navigation.get("use_lines", []):
            if int(line.get("line_id", -1)) == int(exit_line.get("line_id", -2)):
                continue
            if int(line.get("special", 0)) != 1:
                continue
            if float(line.get("distance", 999999)) > (
                EXIT_ASSIST_DOOR_CLOSE_USE_DISTANCE_UNITS + 24.0
            ):
                continue
            if self._line_center_distance(line, exit_line) > EXIT_ASSIST_DOOR_EXIT_DISTANCE_UNITS:
                continue
            if abs(float(line.get("angle_delta", 999))) > EXIT_ASSIST_DOOR_CLOSE_USE_ANGLE_DEGREES:
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                int(line.get("side", 0)),
                float(line["distance"]),
                abs(float(line["angle_delta"])),
                self._line_center_distance(line, exit_line),
            ),
        )

    def _select_exit_assist_door(
        self,
        features: TacticalFeatures,
        exit_line: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidates = []
        for line in features.navigation.get("use_lines", []):
            special = int(line.get("special", 0))
            if int(line.get("line_id", -1)) == int(exit_line.get("line_id", -2)):
                continue
            if special not in MANUAL_USE_LINE_SPECIALS:
                continue
            if float(line.get("distance", 999999)) > EXIT_ASSIST_DOOR_USE_DISTANCE_UNITS:
                continue
            if self._line_center_distance(line, exit_line) > EXIT_ASSIST_DOOR_EXIT_DISTANCE_UNITS:
                continue
            if abs(float(line.get("angle_delta", 999))) > 120:
                continue
            if self._is_line_blocked(features, line):
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                0 if int(line.get("special", 0)) == 1 else 1,
                self._line_center_distance(line, exit_line),
                abs(float(line["angle_delta"])),
                float(line["distance"]),
            ),
        )

    @staticmethod
    def _line_center_distance(line: dict[str, Any], other: dict[str, Any]) -> float:
        return math.dist(
            (
                float(line.get("x_units", line.get("nearest_x_units", 0.0))),
                float(line.get("y_units", line.get("nearest_y_units", 0.0))),
            ),
            (
                float(other.get("x_units", other.get("nearest_x_units", 0.0))),
                float(other.get("y_units", other.get("nearest_y_units", 0.0))),
            ),
        )

    def _move_on_ray_toward_line(
        self,
        features: TacticalFeatures,
        aim_angle_delta: float,
        ray: dict[str, Any],
        skill: str,
        stuck: bool,
        line_record: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        offset = float(ray["angle_offset_degrees"])
        radians = math.radians(offset)
        forward = _clamp_int(
            round(self.params.move_amount * max(0.2, math.cos(radians))),
            4,
            self.params.move_amount,
        )
        side = _clamp_int(
            round(-math.sin(radians) * self.params.move_amount),
            -self.params.move_amount,
            self.params.move_amount,
        )
        return (
            raw_ticcmd_action(
                forward_move=forward,
                side_move=side,
                angle_turn=raw_turn_for_delta(aim_angle_delta),
                duration_tics=4,
                tick=features.tick,
            ),
            self._decision(
                skill,
                features,
                stuck=stuck,
                use_line=line_record,
                direction_probe={
                    "angle_offset_degrees": int(offset),
                    "block_distance_fp": int(ray.get("block_distance_fp", 0)),
                },
            ),
        )

    def _use_exit_assist_retry(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        angle_delta = self._line_control_angle_delta(line)
        line_record = self._line_record(line)
        if abs(angle_delta) > 15:
            return (
                semantic_action(
                    turn_action_for_delta(angle_delta),
                    amount=self.params.turn_amount,
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    "turn_to_retry_exit_assist_door",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )

        self._last_use_tick = features.tick
        return (
            raw_ticcmd_action(
                buttons=BT_USE,
                forward_move=max(4, self.params.move_amount // 2),
                angle_turn=raw_turn_for_delta(angle_delta),
                duration_tics=2,
                tick=features.tick,
            ),
            self._decision(
                "retry_exit_assist_door",
                features,
                stuck=stuck,
                use_line=line_record,
            ),
        )

    def _use_nearby_line(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._record_line_attempt(features, line):
            return self._explore(features, stuck)

        angle_delta = self._line_control_angle_delta(line)
        distance = self._line_control_distance(line)
        line_record = self._line_record(line)
        special = int(line["special"])
        activate_distance = self._line_activate_distance(features, line)
        if (
            special == 1
            and self._has_local_exit_line(features)
            and distance <= EXIT_ASSIST_DOOR_CLOSE_USE_DISTANCE_UNITS
            and abs(angle_delta) <= EXIT_ASSIST_DOOR_CLOSE_USE_ANGLE_DEGREES
        ):
            if abs(angle_delta) > 12:
                return (
                    semantic_action(
                        turn_action_for_delta(angle_delta),
                        amount=self.params.turn_amount,
                        duration_tics=2,
                        tick=features.tick,
                    ),
                    self._decision(
                        "turn_to_exit_assist_door",
                        features,
                        stuck=stuck,
                        use_line=line_record,
                    ),
                )
            self._last_use_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    "use_exit_assist_door",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        if (
            special in EXIT_LINE_SPECIALS
            and int(line.get("side", 0)) == 1
            and abs(angle_delta) <= 18
        ):
            return (
                raw_ticcmd_action(
                    forward_move=self.params.move_amount,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision(
                    "approach_nearby_use_line_front",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        if (
            distance > activate_distance
            and abs(angle_delta) <= 18
        ):
            return (
                raw_ticcmd_action(
                    forward_move=self.params.move_amount,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision(
                    "approach_nearby_use_line",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        if (
            abs(angle_delta) <= 12
            and (special not in EXIT_LINE_SPECIALS or int(line.get("side", 0)) == 0)
        ):
            self._last_use_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    angle_turn=raw_turn_for_delta(angle_delta),
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    "use_nearby_line",
                    features,
                    stuck=stuck,
                    use_line=line_record,
                ),
            )
        return (
            semantic_action(
                turn_action_for_delta(angle_delta),
                amount=self.params.turn_amount,
                duration_tics=2,
                tick=features.tick,
            ),
            self._decision(
                "turn_to_nearby_use_line",
                features,
                stuck=stuck,
                use_line=line_record,
            ),
        )

    def _use_directional_line(
        self,
        features: TacticalFeatures,
        ray: dict[str, Any],
        stuck: bool,
    ) -> tuple[Any, dict[str, Any]]:
        offset = float(ray["angle_offset_degrees"])
        if abs(offset) <= 15:
            self._last_use_tick = features.tick
            return (
                raw_ticcmd_action(
                    buttons=BT_USE,
                    forward_move=max(4, self.params.move_amount // 2)
                    if self._has_local_exit_line(features)
                    else 0,
                    angle_turn=raw_turn_for_delta(offset),
                    duration_tics=2,
                    tick=features.tick,
                ),
                self._decision(
                    "use_directional_line",
                    features,
                    stuck=stuck,
                    direction_probe={
                        "angle_offset_degrees": int(offset),
                        "block_distance_fp": int(ray.get("block_distance_fp", 0)),
                    },
                ),
            )
        return (
            semantic_action(
                turn_action_for_delta(offset),
                amount=self.params.turn_amount,
                duration_tics=2,
                tick=features.tick,
            ),
            self._decision(
                "turn_to_use_line",
                features,
                stuck=stuck,
                direction_probe={
                    "angle_offset_degrees": int(offset),
                    "block_distance_fp": int(ray.get("block_distance_fp", 0)),
                },
            ),
        )

    def _has_episode_kill(self, features: TacticalFeatures) -> bool:
        start_kills = self._start_kills if self._start_kills is not None else features.kills
        return features.kills > start_kills

    def _line_record(self, line: dict[str, Any]) -> dict[str, Any]:
        record = {
            "line_id": int(line["line_id"]),
            "special": int(line["special"]),
            "tag": int(line["tag"]),
            "distance": round(float(line["distance"]), 2),
            "angle_delta": round(float(line["angle_delta"]), 2),
            "side": int(line.get("side", 0)),
        }
        if "front_distance" in line:
            record["front_distance"] = round(float(line["front_distance"]), 2)
            record["front_angle_delta"] = round(float(line["front_angle_delta"]), 2)
        return record

    def _line_control_angle_delta(self, line: dict[str, Any]) -> float:
        if (
            int(line.get("special", 0)) in EXIT_LINE_SPECIALS
            and int(line.get("side", 0)) == 1
            and float(line.get("distance", 999999)) <= EXIT_ASSIST_DISTANCE_UNITS
            and "front_angle_delta" in line
        ):
            return float(line["front_angle_delta"])
        return float(line["angle_delta"])

    def _line_control_distance(self, line: dict[str, Any]) -> float:
        if (
            int(line.get("special", 0)) in EXIT_LINE_SPECIALS
            and int(line.get("side", 0)) == 1
            and float(line.get("distance", 999999)) <= EXIT_ASSIST_DISTANCE_UNITS
            and "front_distance" in line
        ):
            return float(line["front_distance"])
        return float(line["distance"])

    def _progression_line_priority(self, line: dict[str, Any]) -> int:
        special = int(line["special"])
        if special in EXIT_LINE_SPECIALS and float(line["distance"]) <= EXIT_ASSIST_DISTANCE_UNITS:
            return -1
        return PROGRESSION_LINE_PRIORITIES[special]

    def _line_activate_distance(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
    ) -> float:
        special = int(line["special"])
        if special in EXIT_LINE_SPECIALS:
            return EXIT_LINE_USE_DISTANCE_UNITS
        if special == 1 and self._has_local_exit_line(features):
            return EXIT_ASSIST_DOOR_CLOSE_USE_DISTANCE_UNITS
        return USE_LINE_ACTIVATE_DISTANCE_UNITS

    def _has_local_exit_line(self, features: TacticalFeatures) -> bool:
        return any(
            int(line.get("special", 0)) in EXIT_LINE_SPECIALS
            and float(line.get("distance", 999999)) <= EXIT_ASSIST_DISTANCE_UNITS
            for line in features.navigation.get("use_lines", [])
        )

    def _record_line_attempt(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
    ) -> bool:
        if (
            int(line.get("special", 0)) in EXIT_LINE_SPECIALS
            and float(line.get("distance", 999999)) <= EXIT_ASSIST_DISTANCE_UNITS
        ):
            return True
        if (
            bool(features.navigation.get("use_line_ahead"))
            and int(line.get("special", 0)) in MANUAL_USE_LINE_SPECIALS
            and float(line.get("distance", 999999)) <= 192.0
            and not (
                int(line.get("special", 0)) == 1
                and self._has_local_exit_line(features)
            )
        ):
            return True

        key = self._line_key(features.cell, line)
        blocked_until = self._blocked_use_lines.get(key)
        if blocked_until is not None and blocked_until > features.tick:
            return False

        signature = {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "items": features.items,
            "kills": features.kills,
        }
        previous = self._line_attempts.get(key)
        if previous is None or previous.get("signature") != signature:
            self._line_attempts[key] = {
                "first_tick": features.tick,
                "last_tick": features.tick,
                "best_distance": float(line["distance"]),
                "signature": signature,
            }
            return True

        distance = float(line["distance"])
        if distance < float(previous.get("best_distance", distance)) - 8.0:
            previous["first_tick"] = features.tick
            previous["best_distance"] = distance

        previous["last_tick"] = features.tick
        if features.tick - int(previous["first_tick"]) >= LINE_ATTEMPT_STALL_TICS:
            self._blocked_use_lines[key] = features.tick + LINE_ATTEMPT_BLOCK_TICS
            self._line_attempts.pop(key, None)
            return False
        return True

    def _exit_push_stalled(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
        distance: float,
    ) -> bool:
        key = self._line_key(features.cell, line)
        signature = {
            "cell": features.cell,
            "episode": features.episode,
            "map": features.map,
            "kills": features.kills,
            "items": features.items,
        }
        previous = self._exit_push_attempts.get(key)
        if previous is None or previous.get("signature") != signature:
            self._exit_push_attempts[key] = {
                "first_tick": features.tick,
                "best_distance": distance,
                "signature": signature,
            }
            return False

        if distance < float(previous.get("best_distance", distance)) - 8.0:
            previous["first_tick"] = features.tick
            previous["best_distance"] = distance
            return False

        return features.tick - int(previous["first_tick"]) >= LINE_ATTEMPT_STALL_TICS

    def _is_line_blocked(
        self,
        features: TacticalFeatures,
        line: dict[str, Any],
    ) -> bool:
        key = self._line_key(features.cell, line)
        blocked_until = self._blocked_use_lines.get(key)
        if blocked_until is None:
            return False
        if blocked_until <= features.tick:
            del self._blocked_use_lines[key]
            return False
        return True

    @staticmethod
    def _line_key(cell: str, line: dict[str, Any]) -> str:
        return f"{cell}:{int(line['line_id'])}"

    def _can_shoot(
        self,
        features: TacticalFeatures,
        enemy: dict[str, Any] | None = None,
    ) -> bool:
        cooldown = self.params.shoot_cooldown_tics
        if enemy is not None and enemy.get("line_of_sight"):
            cooldown = min(cooldown, 1)
        return (
            features.ammo_bullets > 0
            and features.tick - self._last_shot_tick >= cooldown
        )

    def _avoid_blocked_front(
        self,
        features: TacticalFeatures,
        source_skill: str,
    ) -> tuple[Any, dict[str, Any]]:
        nav = features.navigation
        if nav["use_line_ahead"] and features.tick - self._last_use_tick >= max(8, self.params.use_interval_tics // 2):
            self._last_use_tick = features.tick
            return (
                semantic_action(agent_pb2.ACTION_USE, duration_tics=2, tick=features.tick),
                self._decision(
                    "use_blocking_line",
                    features,
                    source_skill=source_skill,
                    front_block_distance_fp=nav["front_block_distance_fp"],
                    front_blocking_line_special=nav["front_blocking_line_special"],
                    stuck=False,
                ),
            )

        if nav["left_open"] and not nav["right_open"]:
            return (
                semantic_action(
                    agent_pb2.ACTION_STRAFE_LEFT,
                    amount=self.params.strafe_amount,
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision("sidestep_left", features, source_skill=source_skill, stuck=False),
            )

        if nav["right_open"] and not nav["left_open"]:
            return (
                semantic_action(
                    agent_pb2.ACTION_STRAFE_RIGHT,
                    amount=self.params.strafe_amount,
                    duration_tics=3,
                    tick=features.tick,
                ),
                self._decision("sidestep_right", features, source_skill=source_skill, stuck=False),
            )

        turn_right = nav["right_open"] or not nav["left_open"]
        return (
            raw_ticcmd_action(
                forward_move=4,
                side_move=(self.params.strafe_amount // 2) if turn_right else -(self.params.strafe_amount // 2),
                angle_turn=-768 if turn_right else 768,
                duration_tics=3,
                tick=features.tick,
            ),
            self._decision("turn_from_block", features, source_skill=source_skill, stuck=False),
        )

    def _select_known_enemy(self, features: TacticalFeatures) -> dict[str, Any] | None:
        for enemy in features.known_enemies:
            if self._is_blocked_target(features, enemy):
                continue
            return enemy
        return None

    def _should_hunt_known_enemy(self, features: TacticalFeatures, enemy: dict[str, Any]) -> bool:
        """Use global enemy coordinates only as a local hint, not wall GPS."""
        if enemy.get("line_of_sight"):
            return True
        if int(enemy["id"]) != self._last_visible_enemy_id:
            return False
        if features.tick - self._last_visible_enemy_tick > min(
            self.params.enemy_memory_tics, 80
        ):
            return False
        if enemy["distance"] > 900:
            return False
        if abs(enemy["angle_delta"]) > 35:
            return False
        if features.navigation["forward_open"]:
            return True
        return features.navigation["left_open"] or features.navigation["right_open"]

    def _should_seek_known_enemy(self, features: TacticalFeatures, enemy: dict[str, Any]) -> bool:
        """Use known enemy coordinates as a bounded exploration objective."""
        start_kills = self._start_kills if self._start_kills is not None else features.kills
        if features.kills - start_kills >= 5:
            return False
        if enemy.get("line_of_sight"):
            return True
        if enemy["distance"] > 2600:
            return False
        if abs(enemy["angle_delta"]) > 150:
            return False
        if self._episode_cell_visits.get(features.cell, 0) > max(
            36, self.params.stuck_window_tics * 3
        ):
            return False
        return (
            features.navigation["forward_open"]
            or features.navigation["left_open"]
            or features.navigation["right_open"]
        )

    def _mark_blocked_target(self, features: TacticalFeatures) -> None:
        if not features.known_enemies:
            return
        if self.last_decision.get("skill") not in {"hunt_known_enemy", "seek_known_enemy"}:
            return
        enemy = self.last_decision.get("enemy")
        enemy_id = enemy.get("id") if isinstance(enemy, dict) else features.known_enemies[0]["id"]
        key = self._blocked_target_key(features.cell, int(enemy_id))
        self._blocked_enemy_cells[key] = features.tick + max(120, self.params.enemy_memory_tics)

    def _is_blocked_target(self, features: TacticalFeatures, enemy: dict[str, Any]) -> bool:
        key = self._blocked_target_key(features.cell, int(enemy["id"]))
        blocked_until = self._blocked_enemy_cells.get(key)
        if blocked_until is None:
            return False
        if blocked_until <= features.tick:
            del self._blocked_enemy_cells[key]
            return False
        return True

    @staticmethod
    def _blocked_target_key(cell: str, enemy_id: int) -> str:
        return f"{cell}:{enemy_id}"

    def _decision(self, skill: str, features: TacticalFeatures, **extra: Any) -> dict[str, Any]:
        enemy = extra.pop("enemy", None)
        decision = {
            "policy_id": self.policy_id,
            "skill": skill,
            "tick": features.tick,
            "cell": features.cell,
            "health": features.health,
            "ammo_bullets": features.ammo_bullets,
            "visible_enemies": len(features.visible_enemies),
            "known_enemies": len(features.known_enemies),
            "remembered_enemies": len(features.remembered_enemies),
            "blocked_targets": len(self._blocked_enemy_cells),
            "navigation": features.navigation,
            "combat": features.combat,
            **extra,
        }
        if enemy is not None:
            decision["enemy"] = {
                "id": enemy["id"],
                "distance": round(enemy["distance"], 2),
                "angle_delta": round(enemy["angle_delta"], 2),
                "health": enemy["health"],
                "threat": round(enemy["threat"], 3),
            }
        return decision

    def _shootable_enemy(self, features: TacticalFeatures) -> dict[str, Any] | None:
        combat = features.combat
        if not combat["has_shootable_target"] or not combat["target_is_enemy"]:
            return None
        target_id = int(combat["target_id"])
        for enemy in features.known_enemies:
            if int(enemy["id"]) == target_id:
                return {
                    **enemy,
                    "distance": combat["target_distance_fp"] / FP,
                    "angle_delta": 0.0,
                    "health": combat["target_health"],
                    "line_of_sight": True,
                }
        return {
            "id": target_id,
            "x": features.x_units,
            "y": features.y_units,
            "distance": combat["target_distance_fp"] / FP,
            "angle_delta": 0.0,
            "health": combat["target_health"],
            "threat": 1.0,
            "line_of_sight": True,
            "last_seen_tick": features.tick,
        }


async def run_brain(config: BrainConfig) -> dict[str, Any]:
    """Run one or more brain episodes and return aggregate telemetry."""
    memory = AgentMemory.load(config.memory_path)
    rng = random.Random(config.seed)
    base_params = memory.best_params()
    summaries: list[dict[str, Any]] = []

    for index in range(config.evolve_runs):
        candidate_id = f"candidate-{index + 1}"
        params = base_params if index == 0 else base_params.mutate(rng, scale=1.0)
        summary = await run_brain_episode(config, memory, params, candidate_id)
        summaries.append(summary)
        if summary["promoted"]:
            base_params = BrainPolicyParams(**summary["params"]).bounded()
        if summary.get("success"):
            break

    return {
        "schema": "restfuldoom.brain_run.v1",
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "goal_preset": config.goal_preset,
        "mission": config.mission,
        "success": any(summary.get("success") for summary in summaries),
        "episodes": summaries,
        "memory": memory.summary(),
    }


async def run_brain_episode(
    config: BrainConfig,
    memory: AgentMemory,
    params: BrainPolicyParams,
    candidate_id: str,
) -> dict[str, Any]:
    """Run one structured-brain episode."""
    run_id = f"brain-{uuid.uuid4().hex[:12]}"
    skill_model_path = _resolve_skill_model_path(config, memory)
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
    stats = EpisodeStats(
        run_id=run_id,
        candidate_id=candidate_id,
        policy_id=config.policy_id,
        goal=config.goal_preset,
    )
    metadata = {
        "source": "agent-brain",
        "run_id": run_id,
        "candidate_id": candidate_id,
        "policy_id": config.policy_id,
        "goal_preset": config.goal_preset,
        "mission": config.mission,
        "endpoint_host": safe_endpoint_host(config.endpoint),
        "memory_path": str(config.memory_path),
        "params": asdict(params),
    }
    if skill_model_path is not None:
        metadata["skill_model_path"] = str(skill_model_path)
    trajectory_path = _candidate_trajectory_path(config.trajectory_jsonl, candidate_id)
    enemy_health_by_id: dict[int, int] = {}

    try:
        async for step in client.stream_rollout(
            policy,
            reward_engine=reward,
            max_states=config.max_states,
            trajectory_jsonl=trajectory_path,
            reconnect=config.reconnect,
            backoff=BackoffConfig(max_attempts=config.max_reconnects),
            rollout_metadata=metadata,
        ):
            features = policy.last_features
            if features is None:
                continue
            if stats.states == 0:
                stats.start_tick = features.tick
                stats.start_episode = features.episode
                stats.start_map = features.map
                stats.start_health = features.health
                stats.start_kills = features.kills
                stats.peak_kills = features.kills
                stats.start_items = features.items
                stats.peak_items = features.items
            stats.observe_nearest_enemy(
                features.known_enemies[0]["distance"] if features.known_enemies else None
            )
            stats.states += 1
            stats.end_tick = features.tick
            stats.end_episode = features.episode
            stats.end_map = features.map
            stats.level_completed = stats.level_completed or (
                stats.start_episode is not None
                and stats.start_map is not None
                and (features.episode, features.map) != (stats.start_episode, stats.start_map)
            )
            stats.end_health = features.health
            stats.end_kills = features.kills
            stats.peak_kills = max(stats.peak_kills, features.kills)
            stats.end_items = features.items
            stats.peak_items = max(stats.peak_items, features.items)
            stats.total_reward += step.reward.reward
            damage = _record_enemy_damage(enemy_health_by_id, features)
            if damage > 0:
                stats.enemy_damage += damage
            skill = policy.last_decision.get("skill", "unknown")
            stats.skill_counts[skill] = stats.skill_counts.get(skill, 0) + 1
            memory.record_step(features, policy.last_decision, step.reward, stats)

            if step.reward.kill_delta > 0:
                stats.lessons.append(
                    f"kill gained in cell {features.cell} using {skill}"
                )
            if damage > 0:
                stats.lessons.append(
                    f"dealt {damage} damage in cell {features.cell} using {skill}"
                )
            if step.reward.done:
                stats.deaths += 1
                if features.visible_enemies:
                    stats.lessons.append(
                        f"died in cell {features.cell} with {len(features.visible_enemies)} visible enemies"
                    )
                if config.stop_on_death:
                    break
            if stats.succeeded(
                required_kills=config.required_kills,
                require_level_complete=config.require_level_complete,
            ):
                stats.lessons.append(
                    f"good state reached: level_completed={stats.level_completed}, kills={stats.kill_delta()}"
                )
                break
    finally:
        await client.close()

    success = stats.succeeded(
        required_kills=config.required_kills,
        require_level_complete=config.require_level_complete,
    )
    promoted = memory.should_promote(stats.score())
    summary = memory.finish_episode(stats=stats, params=params, promoted=promoted)
    summary["trajectory_jsonl"] = str(trajectory_path) if trajectory_path else None
    summary["success"] = success
    episodes = memory.data.setdefault("episodes", [])
    if episodes:
        episodes[-1]["trajectory_jsonl"] = summary["trajectory_jsonl"]
        episodes[-1]["success"] = success
    memory.data.setdefault("policy", {})["last_success"] = success
    if success:
        memory.data.setdefault("successes", []).append(summary)
        memory.data["policy"]["last_success_run_id"] = stats.run_id
    memory.save()
    return summary


def extract_features(
    state: Any,
    memory: AgentMemory,
    params: BrainPolicyParams,
) -> TacticalFeatures:
    """Convert a protobuf `GameState` into policy features."""
    player = state.player
    obj = player.object
    position = obj.position
    x = position.x_fp / FP
    y = position.y_fp / FP
    angle = float(obj.angle_degrees % 360)
    navigation = navigation_from_state(getattr(state, "navigation", None))
    combat = combat_from_state(getattr(state, "combat", None))
    for line in navigation.get("use_lines", []):
        line["distance"] = line["nearest_distance_fp"] / FP
        line["angle_delta"] = angle_to_target_units(
            x,
            y,
            angle,
            float(line["nearest_x_units"]),
            float(line["nearest_y_units"]),
        )
        line["side"] = doom_line_side(x, y, line)
        front_point = line_front_point_units(line, offset_units=96.0)
        if front_point is not None:
            front_x, front_y = front_point
            line["front_x_units"] = front_x
            line["front_y_units"] = front_y
            line["front_distance"] = math.dist((x, y), (front_x, front_y))
            line["front_angle_delta"] = angle_to_target_units(
                x,
                y,
                angle,
                front_x,
                front_y,
            )
    navigation["use_lines"].sort(
        key=lambda line: (float(line["distance"]), abs(float(line["angle_delta"])))
    )
    visible = []
    known = []
    remembered = []

    for enemy in state.enemies:
        enemy_obj = enemy.object
        ex = enemy_obj.position.x_fp / FP
        ey = enemy_obj.position.y_fp / FP
        distance = enemy_obj.distance_fp / FP
        delta = angle_to_target_units(x, y, angle, ex, ey)
        threat = _threat(distance, enemy_obj.health, enemy.line_of_sight, delta)
        known_enemy = {
            "id": enemy_obj.id,
            "x": ex,
            "y": ey,
            "distance": distance,
            "angle_delta": delta,
            "health": enemy_obj.health,
            "threat": threat,
            "line_of_sight": bool(enemy.line_of_sight),
            "last_seen_tick": int(state.tick),
        }
        known.append(known_enemy)
        if enemy.line_of_sight:
            visible.append(known_enemy)

    visible.sort(key=lambda item: (-item["threat"], item["distance"]))
    known.sort(key=lambda item: (item["distance"], abs(item["angle_delta"])))

    enemies = memory.data.get("enemies", {})
    for enemy_id, entry in enemies.items():
        last_tick = int(entry.get("last_seen_tick", -999999))
        tick_age = int(state.tick) - last_tick
        if tick_age < 0 or tick_age > params.enemy_memory_tics:
            continue
        last_position = entry.get("last_position")
        if not isinstance(last_position, list) or len(last_position) != 2:
            continue
        ex, ey = float(last_position[0]), float(last_position[1])
        distance = math.dist((x, y), (ex, ey))
        remembered.append(
            {
                "id": int(enemy_id),
                "x": ex,
                "y": ey,
                "distance": distance,
                "last_seen_tick": last_tick,
            }
        )
    remembered.sort(key=lambda item: (item["distance"], -item["last_seen_tick"]))

    return TacticalFeatures(
        tick=int(state.tick),
        x_units=x,
        y_units=y,
        angle=angle,
        health=player.health,
        ammo_bullets=player.ammo.bullets,
        kills=player.kills,
        items=player.items,
        secrets=player.secrets,
        cell=cell_key(x, y),
        visible_enemies=visible,
        known_enemies=known,
        remembered_enemies=remembered,
        enemy_count=len(state.enemies),
        navigation=navigation,
        combat=combat,
        episode=state.level.episode,
        map=state.level.map,
    )


def navigation_from_state(navigation: Any | None) -> dict[str, Any]:
    """Return JSON-safe navigation affordances with conservative defaults."""
    if navigation is None:
        return {
            "forward_open": True,
            "back_open": True,
            "left_open": True,
            "right_open": True,
            "use_line_ahead": False,
            "front_blocking_line_special": 0,
            "front_block_distance_fp": 0,
            "probe_distance_fp": 0,
            "direction_probes": [],
            "use_lines": [],
        }

    direction_probes = [
        {
            "angle_offset_degrees": int(getattr(probe, "angle_offset_degrees", 0)),
            "open": bool(getattr(probe, "open", False)),
            "block_distance_fp": int(getattr(probe, "block_distance_fp", 0)),
            "blocking_line_special": int(getattr(probe, "blocking_line_special", 0)),
            "use_line_ahead": bool(getattr(probe, "use_line_ahead", False)),
        }
        for probe in getattr(navigation, "direction_probes", [])
    ]
    if not direction_probes:
        probe_distance = int(getattr(navigation, "probe_distance_fp", 0))
        direction_probes = [
            {
                "angle_offset_degrees": 0,
                "open": bool(getattr(navigation, "forward_open", True)),
                "block_distance_fp": int(
                    getattr(navigation, "front_block_distance_fp", probe_distance)
                ),
                "blocking_line_special": int(
                    getattr(navigation, "front_blocking_line_special", 0)
                ),
                "use_line_ahead": bool(getattr(navigation, "use_line_ahead", False)),
            },
            {
                "angle_offset_degrees": 90,
                "open": bool(getattr(navigation, "left_open", True)),
                "block_distance_fp": probe_distance,
                "blocking_line_special": 0,
                "use_line_ahead": False,
            },
            {
                "angle_offset_degrees": -90,
                "open": bool(getattr(navigation, "right_open", True)),
                "block_distance_fp": probe_distance,
                "blocking_line_special": 0,
                "use_line_ahead": False,
            },
        ]
    use_lines = []
    for line in getattr(navigation, "use_lines", []):
        midpoint = getattr(line, "midpoint", None)
        nearest = getattr(line, "nearest_point", None)
        start = getattr(line, "start", None)
        end = getattr(line, "end", None)
        x_fp = int(getattr(midpoint, "x_fp", 0))
        y_fp = int(getattr(midpoint, "y_fp", 0))
        z_fp = int(getattr(midpoint, "z_fp", 0))
        nearest_x_fp = int(getattr(nearest, "x_fp", x_fp))
        nearest_y_fp = int(getattr(nearest, "y_fp", y_fp))
        start_x_fp = int(getattr(start, "x_fp", x_fp))
        start_y_fp = int(getattr(start, "y_fp", y_fp))
        end_x_fp = int(getattr(end, "x_fp", x_fp))
        end_y_fp = int(getattr(end, "y_fp", y_fp))
        use_lines.append(
            {
                "line_id": int(getattr(line, "line_id", 0)),
                "x_units": x_fp / FP,
                "y_units": y_fp / FP,
                "z_units": z_fp / FP,
                "nearest_x_units": nearest_x_fp / FP,
                "nearest_y_units": nearest_y_fp / FP,
                "start_x_units": start_x_fp / FP,
                "start_y_units": start_y_fp / FP,
                "end_x_units": end_x_fp / FP,
                "end_y_units": end_y_fp / FP,
                "special": int(getattr(line, "special", 0)),
                "tag": int(getattr(line, "tag", 0)),
                "distance_fp": int(getattr(line, "distance_fp", 0)),
                "nearest_distance_fp": int(
                    getattr(line, "nearest_distance_fp", getattr(line, "distance_fp", 0))
                ),
            }
        )

    return {
        "forward_open": bool(getattr(navigation, "forward_open", True)),
        "back_open": bool(getattr(navigation, "back_open", True)),
        "left_open": bool(getattr(navigation, "left_open", True)),
        "right_open": bool(getattr(navigation, "right_open", True)),
        "use_line_ahead": bool(getattr(navigation, "use_line_ahead", False)),
        "front_blocking_line_special": int(
            getattr(navigation, "front_blocking_line_special", 0)
        ),
        "front_block_distance_fp": int(getattr(navigation, "front_block_distance_fp", 0)),
        "probe_distance_fp": int(getattr(navigation, "probe_distance_fp", 0)),
        "direction_probes": direction_probes,
        "use_lines": use_lines,
    }


def combat_from_state(combat: Any | None) -> dict[str, Any]:
    """Return JSON-safe combat affordances with conservative defaults."""
    if combat is None:
        return {
            "has_shootable_target": False,
            "target_id": 0,
            "target_health": 0,
            "target_distance_fp": 0,
            "aim_slope_fp": 0,
            "range_fp": 0,
            "target_is_enemy": False,
        }

    return {
        "has_shootable_target": bool(getattr(combat, "has_shootable_target", False)),
        "target_id": int(getattr(combat, "target_id", 0)),
        "target_health": int(getattr(combat, "target_health", 0)),
        "target_distance_fp": int(getattr(combat, "target_distance_fp", 0)),
        "aim_slope_fp": int(getattr(combat, "aim_slope_fp", 0)),
        "range_fp": int(getattr(combat, "range_fp", 0)),
        "target_is_enemy": bool(getattr(combat, "target_is_enemy", False)),
    }


def raw_ticcmd_action(
    *,
    forward_move: int = 0,
    side_move: int = 0,
    angle_turn: int = 0,
    buttons: int = 0,
    duration_tics: int = 1,
    tick: int = 0,
) -> Any:
    """Create a low-level ticcmd action for combined move/turn control."""
    return agent_pb2.PlayerAction(
        tick=tick,
        duration_tics=duration_tics,
        raw=agent_pb2.RawTiccmd(
            forward_move=int(forward_move),
            side_move=int(side_move),
            angle_turn=int(angle_turn),
            buttons=int(buttons),
        ),
    )


def _record_enemy_damage(
    enemy_health_by_id: dict[int, int],
    features: TacticalFeatures,
) -> int:
    """Update per-run enemy health memory and return observed damage."""
    damage = 0
    seen: set[int] = set()
    for enemy in features.known_enemies:
        enemy_id = int(enemy["id"])
        health = int(enemy["health"])
        seen.add(enemy_id)
        prior = enemy_health_by_id.get(enemy_id)
        if prior is not None and health < prior:
            damage += prior - health
        enemy_health_by_id[enemy_id] = health
    for enemy_id in list(enemy_health_by_id):
        if enemy_id not in seen:
            del enemy_health_by_id[enemy_id]
    return damage


def export_training_job(
    output_path: str | Path,
    *,
    memory_path: str | Path = "agent_memory/e1m1.json",
    notes_path: str | Path = "agent-notes.md",
) -> dict[str, Any]:
    """Export a portable training checkpoint for Docker-to-cloud resume."""
    output = Path(output_path)
    memory = AgentMemory.load(memory_path)
    trajectory_paths = [
        path for path in _episode_trajectory_paths(memory) if path.exists()
    ]
    skill_model_path = _memory_skill_model_path(memory)
    model_checkpoints = []
    if skill_model_path is not None and skill_model_path.exists():
        model_checkpoints.append(skill_model_path)
    ppo_checkpoints = _memory_ppo_checkpoint_paths(memory)
    manifest = {
        "schema": "restfuldoom.training_job.v1",
        "created_at": _iso_now(),
        "memory_path": "agent_memory/e1m1.json",
        "notes_path": "agent-notes.md",
        "memory_summary": memory.summary(),
        "best_params": memory.data.get("policy", {}).get("best_params"),
        "best_score": memory.data.get("policy", {}).get("best_score"),
        "best_run_id": memory.data.get("policy", {}).get("best_run_id"),
        "learned_policy": memory.data.get("learned_policy"),
        "model_checkpoints": [
            f"agent_models/{path.name}" for path in model_checkpoints
        ],
        "ppo_policy": memory.data.get("ppo_policy"),
        "ppo_checkpoints": [
            f"agent_models/ppo/{path.name}" for path in ppo_checkpoints
        ],
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "reward_config": memory.data.get("ppo_policy", {}).get("reward_config")
        if isinstance(memory.data.get("ppo_policy"), dict)
        else None,
        "eval_history": memory.data.get("ppo_policy", {}).get("eval_history", [])
        if isinstance(memory.data.get("ppo_policy"), dict)
        else [],
        "successes": memory.data.get("successes", []),
        "trajectories": [f"trajectories/{path.name}" for path in trajectory_paths],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        _tar_add_json(archive, "manifest.json", manifest)
        _tar_add_path(archive, Path(memory_path), "agent_memory/e1m1.json")
        if Path(notes_path).exists():
            _tar_add_path(archive, Path(notes_path), "agent-notes.md")
        for checkpoint in model_checkpoints:
            archive.add(checkpoint, arcname=f"agent_models/{checkpoint.name}")
        for checkpoint in ppo_checkpoints:
            archive.add(checkpoint, arcname=f"agent_models/ppo/{checkpoint.name}")
        for trajectory in trajectory_paths:
            archive.add(trajectory, arcname=f"trajectories/{trajectory.name}")

    return {
        "schema": manifest["schema"],
        "output_path": str(output),
        "memory_summary": manifest["memory_summary"],
        "trajectory_count": len(manifest["trajectories"]),
        "model_checkpoint_count": len(manifest["model_checkpoints"]),
        "ppo_checkpoint_count": len(manifest["ppo_checkpoints"]),
        "success_count": len(manifest["successes"]),
        "best_score": manifest["best_score"],
        "best_run_id": manifest["best_run_id"],
    }


def train_skill_policy_from_memory(
    output_path: str | Path,
    *,
    memory_path: str | Path = "agent_memory/e1m1.json",
    trajectory_paths: list[str | Path] | None = None,
    config: SkillPolicyTrainConfig | None = None,
) -> dict[str, Any]:
    """Train a learned skill selector from trajectory JSONL and save it in memory."""
    memory = AgentMemory.load(memory_path)
    paths = [Path(path) for path in trajectory_paths] if trajectory_paths else _episode_trajectory_paths(memory)
    summary = train_skill_policy(paths, output_path, config=config)
    memory.data["learned_policy"] = {
        "schema": summary["schema"],
        "checkpoint_path": summary["checkpoint_path"],
        "trained_at": _iso_now(),
        "sample_count": summary["sample_count"],
        "class_count": summary["class_count"],
        "train_accuracy": summary["train_accuracy"],
        "eval_accuracy": summary["eval_accuracy"],
        "classes": summary["classes"],
    }
    memory.data["updated_at"] = _iso_now()
    memory.save()
    return {
        **summary,
        "memory_path": str(memory.path),
    }


def import_training_job(
    bundle_path: str | Path,
    *,
    destination: str | Path = ".",
) -> dict[str, Any]:
    """Import a training checkpoint into a repo or cloud worker directory."""
    bundle = Path(bundle_path)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise ValueError("training job bundle is missing manifest.json")
        manifest = json.loads(manifest_file.read().decode("utf-8"))
        if manifest.get("schema") != "restfuldoom.training_job.v1":
            raise ValueError(
                f"expected restfuldoom.training_job.v1, got {manifest.get('schema')!r}"
            )
        for member in archive.getmembers():
            if member.name == "manifest.json":
                continue
            _safe_extract_member(archive, member, dest)
    return {
        "schema": manifest["schema"],
        "destination": str(dest),
        "memory_path": str(dest / manifest["memory_path"]),
        "trajectory_count": len(manifest.get("trajectories", [])),
        "model_checkpoint_count": len(manifest.get("model_checkpoints", [])),
        "ppo_checkpoint_count": len(manifest.get("ppo_checkpoints", [])),
        "best_score": manifest.get("best_score"),
        "best_run_id": manifest.get("best_run_id"),
        "success_count": len(manifest.get("successes", [])),
    }


def _resolve_skill_model_path(
    config: BrainConfig,
    memory: AgentMemory,
) -> Path | None:
    if config.skill_model_path is not None:
        path = Path(config.skill_model_path)
        return path if path.exists() else None
    return _memory_skill_model_path(memory)


def _memory_skill_model_path(memory: AgentMemory) -> Path | None:
    learned = memory.data.get("learned_policy")
    if not isinstance(learned, dict):
        return None
    checkpoint = learned.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        return None
    path = Path(checkpoint)
    return path if path.exists() else None


def _memory_ppo_checkpoint_paths(memory: AgentMemory) -> list[Path]:
    paths: list[Path] = []
    policy = memory.data.get("ppo_policy")
    if isinstance(policy, dict):
        checkpoint = policy.get("checkpoint_path")
        if isinstance(checkpoint, str) and checkpoint:
            path = Path(checkpoint)
            if path.exists():
                paths.append(path)
    for entry in memory.data.get("ppo_checkpoints", []):
        if isinstance(entry, str):
            path = Path(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("checkpoint_path"), str):
            path = Path(entry["checkpoint_path"])
        else:
            continue
        if path.exists() and path not in paths:
            paths.append(path)
    return paths


def cell_key(x_units: float, y_units: float) -> str:
    """Return a stable map-memory cell key."""
    return f"{math.floor(x_units / CELL_UNITS)}:{math.floor(y_units / CELL_UNITS)}"


def angle_to_target(features: TacticalFeatures, target_x: float, target_y: float) -> float:
    """Return signed angle delta from player view to target."""
    return angle_to_target_units(features.x_units, features.y_units, features.angle, target_x, target_y)


def angle_to_target_units(
    x: float,
    y: float,
    player_angle: float,
    target_x: float,
    target_y: float,
) -> float:
    """Return signed angle delta in degrees, positive meaning turn right."""
    target_angle = math.degrees(math.atan2(target_y - y, target_x - x)) % 360
    return normalize_angle_delta(target_angle - player_angle)


def doom_line_side(x: float, y: float, line: dict[str, Any]) -> int:
    """Return Doom line side 0 for front, 1 for back."""
    start_x = float(line.get("start_x_units", line.get("x_units", 0.0)))
    start_y = float(line.get("start_y_units", line.get("y_units", 0.0)))
    end_x = float(line.get("end_x_units", line.get("x_units", 0.0)))
    end_y = float(line.get("end_y_units", line.get("y_units", 0.0)))
    dx = end_x - start_x
    dy = end_y - start_y
    if abs(dx) < 1e-6:
        if x <= start_x:
            return 1 if dy > 0 else 0
        return 1 if dy < 0 else 0
    if abs(dy) < 1e-6:
        if y <= start_y:
            return 1 if dx < 0 else 0
        return 1 if dx > 0 else 0
    left = dy * (x - start_x)
    right = (y - start_y) * dx
    return 0 if right < left else 1


def line_front_point_units(
    line: dict[str, Any],
    *,
    offset_units: float,
) -> tuple[float, float] | None:
    """Return a point slightly in front of a Doom line."""
    start_x = float(line.get("start_x_units", line.get("x_units", 0.0)))
    start_y = float(line.get("start_y_units", line.get("y_units", 0.0)))
    end_x = float(line.get("end_x_units", line.get("x_units", 0.0)))
    end_y = float(line.get("end_y_units", line.get("y_units", 0.0)))
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    front_normal_x = dy / length
    front_normal_y = -dx / length
    nearest_x = float(line.get("nearest_x_units", line.get("x_units", 0.0)))
    nearest_y = float(line.get("nearest_y_units", line.get("y_units", 0.0)))
    return (
        nearest_x + front_normal_x * offset_units,
        nearest_y + front_normal_y * offset_units,
    )


def normalize_angle_delta(delta: float) -> float:
    """Normalize angle delta into [-180, 180]."""
    while delta <= -180:
        delta += 360
    while delta > 180:
        delta -= 360
    return delta


def turn_action_for_delta(angle_delta: float) -> int:
    """Return the semantic turn action that reduces a signed angle delta."""
    return agent_pb2.ACTION_TURN_LEFT if angle_delta > 0 else agent_pb2.ACTION_TURN_RIGHT


def raw_turn_for_delta(angle_delta: float) -> int:
    """Return a small raw ticcmd turn that nudges aim toward a signed delta."""
    return _clamp_int(angle_delta * 32.0, -512, 512)


def _candidate_trajectory_path(path: Path | None, candidate_id: str) -> Path | None:
    if path is None:
        return None
    if "%CANDIDATE%" in str(path):
        return Path(str(path).replace("%CANDIDATE%", candidate_id))
    if candidate_id == "candidate-1":
        return path
    return path.with_name(f"{path.stem}-{candidate_id}{path.suffix}")


def _episode_trajectory_paths(memory: AgentMemory) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for episode in memory.data.get("episodes", []):
        trajectory = episode.get("trajectory_jsonl")
        if not trajectory:
            continue
        path = Path(trajectory)
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _tar_add_json(archive: tarfile.TarFile, name: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(data))


def _tar_add_path(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.exists():
        archive.add(path, arcname=arcname)


def _safe_extract_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    target = (destination / member.name).resolve()
    root = destination.resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"refusing to extract unsafe path {member.name!r}")
    archive.extract(member, destination)


def _threat(distance: float, health: int, line_of_sight: bool, angle_delta: float) -> float:
    if not line_of_sight:
        return 0.0
    distance_score = max(0.0, 1.0 - min(distance, 800.0) / 800.0)
    aim_score = max(0.0, 1.0 - min(abs(angle_delta), 90.0) / 90.0)
    health_score = max(0.2, min(health, 100) / 100.0)
    return distance_score * 0.6 + aim_score * 0.3 + health_score * 0.1


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clamp_int(value: int | float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def _clamp_float(value: int | float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
