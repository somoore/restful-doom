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
        current = await self._state_stream.__anext__()
        self._steps += 1
        transition = self._reward_engine.score(previous, current)
        done, reason, reward = self._terminal_reward(previous, current, transition)
        self._current_state = current
        return EnvStep(
            observation=self.controller.observation(current),
            reward=reward,
            done=done,
            info={
                "skill": SKILL_ACTIONS[action_index],
                "action_index": action_index,
                "decision": decision,
                "transition": _transition_summary(transition),
                "state": summarize_state(current),
                "done_reason": reason,
            },
        )

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
