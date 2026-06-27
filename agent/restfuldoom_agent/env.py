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
    AgentMemory,
    BrainPolicy,
    BrainPolicyParams,
    MANUAL_USE_LINE_SPECIALS,
    extract_features,
    raw_ticcmd_action,
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
    encode_temporal_context_features,
)
from .skill_policy import features_from_tactical

SKILL_ACTIONS = PPO_SKILL_ACTIONS


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
    first_visible_bonus: float = 0.0
    first_shootable_bonus: float = 0.0
    visible_contact_progress_reward: float = 0.0
    terminate_on_first_visible: bool = False
    terminate_on_first_shootable: bool = False

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
        ]

    def reset_episode_context(self) -> None:
        """Clears stateful observation features at episode boundaries."""
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

    def action_for(self, action_index: int, state: Any) -> tuple[Any, dict[str, Any]]:
        """Returns the PlayerAction for a PPO skill index."""
        if action_index < 0 or action_index >= len(SKILL_ACTIONS):
            raise ValueError(f"action_index must be in [0, {len(SKILL_ACTIONS) - 1}]")

        features = extract_features(state, self.memory, self.params)
        self.policy.last_features = features
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
            return SKILL_ACTIONS.index("engage")
        if self.policy._select_local_exit_line(features) is not None:
            return SKILL_ACTIONS.index("press_exit")
        if (
            self.policy._select_nearby_use_line(features) is not None
            or self.policy._select_use_ray(features) is not None
            or self.policy._should_use_ahead(features)
        ):
            return SKILL_ACTIONS.index("open_use_line")
        if self.policy._select_progression_line(features) is not None:
            return SKILL_ACTIONS.index("route_progression")
        if self.policy._select_known_enemy(features) is not None:
            return SKILL_ACTIONS.index("seek_enemy")
        return SKILL_ACTIONS.index("route_progression")

    def action_mask(self, state: Any) -> list[bool]:
        """Returns currently feasible PPO skills from protobuf affordances."""
        features = extract_features(state, self.memory, self.params)
        stuck = self.policy._is_stuck(features)
        mask = {skill: False for skill in SKILL_ACTIONS}

        shootable = self.policy._shootable_enemy(features)
        if shootable is not None and self.policy._can_shoot(features, shootable):
            mask["fire"] = True

        if features.visible_enemies:
            mask["engage"] = True
            if shootable is None and self.policy._select_known_enemy(features) is not None:
                mask["seek_enemy"] = True
            if shootable is None and self._contact_route_waypoint(features) is not None:
                mask["route_progression"] = True
            if shootable is None and self._contact_use_line(features) is not None:
                mask["open_use_line"] = True
            nearest_visible = min(
                (float(enemy["distance"]) for enemy in features.visible_enemies),
                default=99999.0,
            )
            if (
                features.health <= self.params.retreat_health
                or nearest_visible <= self.params.close_enemy_units
            ):
                mask["retreat"] = True

        if not features.visible_enemies and self.policy._select_known_enemy(features) is not None:
            mask["seek_enemy"] = True

        if (
            self.policy._select_nearby_use_line(features) is not None
            or self.policy._select_use_ray(features) is not None
            or self.policy._should_use_ahead(features)
        ):
            mask["open_use_line"] = True

        if self.policy._select_progression_line(features) is not None or not features.visible_enemies:
            mask["route_progression"] = True

        if stuck:
            mask["recover_stuck"] = True

        if self.policy._select_local_exit_line(features) is not None:
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
                return self.policy._turn_toward_or_move(
                    features,
                    enemy["angle_delta"],
                    "ppo_seek_visible_enemy",
                    stuck,
                    enemy=enemy,
                )
            enemy = self.policy._select_known_enemy(features)
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
            if line is None:
                line = self._contact_use_line(features)
            if line is not None:
                return self.policy._use_nearby_line(features, line, stuck)
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

    def _contact_use_line(self, features: Any) -> dict[str, Any] | None:
        """Returns a manual line worth approaching during visible contact."""
        if not features.visible_enemies or self.policy._shootable_enemy(features) is not None:
            return None
        candidates = []
        for line in features.navigation.get("use_lines", []):
            if int(line.get("special", 0)) not in MANUAL_USE_LINE_SPECIALS:
                continue
            if float(line.get("distance", 999999.0)) > 800.0:
                continue
            if abs(float(line.get("angle_delta", 999.0))) > 75.0:
                continue
            if self.policy._is_line_blocked(features, line):
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda line: (
                abs(float(line.get("angle_delta", 999.0))),
                float(line.get("distance", 999999.0)),
            ),
        )


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
        await self._ensure_stream()
        self._steps = 0
        self._episode_index += 1
        if hasattr(self.controller, "reset_episode_context"):
            self.controller.reset_episode_context()
        reset_seed = self.config.seed if seed is None else seed
        state = await self._reset_with_retries(reset_seed)
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
        self._reward_engine = RewardEngine(self.config.reward_goal())
        self._current_state = state
        self._start_level = (state.level.episode, state.level.map)
        self._start_kills = state.player.kills
        self._episode_seen_visible_enemy = _has_visible_enemy(state)
        self._episode_seen_shootable_enemy = _has_shootable_enemy(state)
        return self.controller.observation(state)

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
        route_outcome = _route_outcome(skill, previous, current)
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
        return EnvStep(
            observation=self.controller.observation(current),
            reward=total_reward,
            done=done,
            info={
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
                "reset_warmup": dict(self._last_reset_warmup),
                "state": summarize_state(current),
                "done_reason": reason,
            },
        )

    def action_mask(self) -> list[bool]:
        """Returns feasible PPO actions for the current state."""
        if self._current_state is None:
            return [True for _ in SKILL_ACTIONS]
        if hasattr(self.controller, "action_mask"):
            return list(self.controller.action_mask(self._current_state))
        return [True for _ in SKILL_ACTIONS]

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
        reward = float(route_outcome.get("progress_units") or 0.0) * self.config.route_progress_reward
        reward = max(-1.0, min(1.0, reward))
        if route_outcome.get("reached"):
            reward += self.config.route_reached_reward
        if route_outcome.get("failed"):
            reward -= self.config.route_failure_penalty
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
        if self.client is None:
            self.client = DoomAgentClient(
                self.config.endpoint,
                token=self.config.token,
                agent_port=self.config.agent_port,
                tls=self.config.tls,
                authority=self.config.authority,
            )
        if self._state_stream is None:
            self._action_queue = asyncio.Queue(maxsize=16)
            self._state_stream = self.client.session(self._action_iter()).__aiter__()

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


def _route_outcome(skill: str, previous: Any, current: Any) -> dict[str, Any]:
    """Summarizes whether a route-progression macro-step helped."""
    route, line = _route_waypoint(previous)
    line_id = int(getattr(line, "line_id", 0)) if line is not None else 0
    outcome: dict[str, Any] = {
        "attempted": skill == "route_progression" and line is not None,
        "line_id": line_id,
        "previous_distance_units": 0.0,
        "current_distance_units": 0.0,
        "progress_units": 0.0,
        "reached": False,
        "failed": False,
        "priority": int(getattr(route, "priority", 0)) if route is not None else 0,
        "exit": bool(getattr(route, "exit", False)) if route is not None else False,
        "walk_trigger": bool(getattr(route, "walk_trigger", False)) if route is not None else False,
    }
    if not outcome["attempted"]:
        return outcome

    previous_x, previous_y = _player_xy_units(previous)
    current_x, current_y = _player_xy_units(current)
    previous_distance = _distance_to_line_units(
        line,
        previous_x,
        previous_y,
        fallback_fp=getattr(line, "nearest_distance_fp", 0),
    )
    current_distance = _distance_to_line_units(line, current_x, current_y)
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


def _distance_to_line_units(
    line: Any,
    x_units: float,
    y_units: float,
    *,
    fallback_fp: int | float = 0,
) -> float:
    start = getattr(line, "start", None)
    end = getattr(line, "end", None)
    if start is None or end is None:
        fallback = float(fallback_fp or 0.0) / 65536.0
        if fallback > 0.0:
            return fallback
        midpoint = getattr(line, "midpoint", None)
        if midpoint is None:
            return 0.0
        return (
            (x_units - float(getattr(midpoint, "x_fp", 0)) / 65536.0) ** 2
            + (y_units - float(getattr(midpoint, "y_fp", 0)) / 65536.0) ** 2
        ) ** 0.5

    x1 = float(getattr(start, "x_fp", 0)) / 65536.0
    y1 = float(getattr(start, "y_fp", 0)) / 65536.0
    x2 = float(getattr(end, "x_fp", 0)) / 65536.0
    y2 = float(getattr(end, "y_fp", 0)) / 65536.0
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
