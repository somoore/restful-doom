"""Gym-style protobuf environment for Doom skill learning."""

from __future__ import annotations

import asyncio
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
    extract_features,
    raw_ticcmd_action,
)
from .client import DoomAgentClient, agent_pb2, semantic_action, summarize_state
from .reward import Goal, RewardEngine, TransitionReward, goal_preset
from .schemas import ACTION_SCHEMA, OBSERVATION_SCHEMA, PPO_SKILL_ACTIONS
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
    max_action_tics: int = 8
    reset_warmup_steps: int = 0
    reset_warmup_max_tics: int = 0
    reset_warmup_until_visible: bool = False
    reset_warmup_until_shootable: bool = False
    shootable_fire_reward: float = 0.5
    missed_fire_penalty: float = 0.05
    blind_fire_penalty: float = 0.02

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

    def observation(self, state: Any) -> list[float]:
        """Encodes a protobuf state as the stable PPO feature vector."""
        features = extract_features(state, self.memory, self.params)
        return features_from_tactical(features)

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
        reset_seed = self.config.seed if seed is None else seed
        reset = await self.client.reset_episode(
            skill=self.config.skill,
            episode=self.config.episode,
            map=self.config.map,
            seed=reset_seed,
            run_id=f"{self.config.run_id}-{self._episode_index}",
        )
        state = await self._next_reset_state(reset.episode, reset.map)
        self._last_reset_warmup = {
            "enabled": False,
            "steps": 0,
            "tics": 0,
            "episode_index": self._episode_index,
        }
        if self.config.reset_warmup_steps > 0:
            state = await self._run_reset_warmup(state)
        self._reward_engine = RewardEngine(self.config.reward_goal())
        self._current_state = state
        self._start_level = (state.level.episode, state.level.map)
        self._start_kills = state.player.kills
        return self.controller.observation(state)

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
        transition_summaries: list[dict[str, Any]] = []
        for _ in range(action_tics):
            tick_previous = current
            current = await self._state_stream.__anext__()
            had_shootable_target = had_shootable_target or _has_shootable_enemy(current)
            self._steps += 1
            transition = self._reward_engine.score(tick_previous, current)
            done, reason, reward = self._terminal_reward(tick_previous, current, transition)
            total_reward += reward
            transition_summaries.append(_transition_summary(transition))
            if done:
                break
        skill = SKILL_ACTIONS[action_index]
        action_reward = self._combat_action_reward(skill, had_shootable_target)
        total_reward += action_reward
        self._current_state = current
        return EnvStep(
            observation=self.controller.observation(current),
            reward=total_reward,
            done=done,
            info={
                "skill": skill,
                "action_index": action_index,
                "decision": decision,
                "transition": _combine_transition_summaries(transition_summaries),
                "macro_tics": len(transition_summaries),
                "action_reward": action_reward,
                "had_shootable_target": had_shootable_target,
                "reset_warmup": dict(self._last_reset_warmup),
                "state": summarize_state(current),
                "done_reason": reason,
            },
        )

    def _combat_action_reward(self, skill: str, had_shootable_target: bool) -> float:
        if had_shootable_target and skill == "fire":
            return self.config.shootable_fire_reward
        if had_shootable_target and skill != "fire":
            return -self.config.missed_fire_penalty
        if skill == "fire":
            return -self.config.blind_fire_penalty
        return 0.0

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
