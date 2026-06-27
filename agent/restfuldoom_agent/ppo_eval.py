"""Evaluation helpers for PPO skill checkpoints."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .env import ACTION_SCHEMA, DoomAgentEnv, DoomEnvConfig
from .ppo import EvaluationResult, PPOTrainer, PromotionDecision, PromotionGate


@dataclass(frozen=True)
class EpisodeEval:
    """One policy evaluation episode."""

    seed: int
    total_reward: float
    level_completed: bool
    death: bool
    max_kills: int
    min_health: int
    steps: int
    steps_to_exit: int
    stuck_events: int
    done_reason: str | None
    start_kills: int = 0
    kill_delta: int = 0
    max_kill_gain: int = 0
    max_items: int = 0
    start_items: int = 0
    item_delta: int = 0
    max_item_gain: int = 0
    max_secrets: int = 0
    start_secrets: int = 0
    secret_delta: int = 0
    max_secret_gain: int = 0
    start_episode: int = 0
    start_map: int = 0
    end_episode: int = 0
    end_map: int = 0
    level_transition_delta: int = 0
    reset_source: str = ""


@dataclass(frozen=True)
class PolicyEval:
    """Detailed policy evaluation with aggregate gate fields."""

    result: EvaluationResult
    episodes: list[EpisodeEval]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-safe evaluation dictionary."""
        return {
            "result": asdict(self.result),
            "episodes": [asdict(episode) for episode in self.episodes],
        }


async def evaluate_checkpoint(
    checkpoint_path: str,
    env_config: DoomEnvConfig,
    *,
    episodes: int = 1,
    max_steps: int | None = None,
    seed: int = 7,
    device: str = "cpu",
    deterministic: bool = True,
    before_reset: Callable[[DoomAgentEnv, int], None] | None = None,
) -> PolicyEval:
    """Evaluates a PPO checkpoint in the Doom skill environment."""
    trainer = PPOTrainer.load_checkpoint(checkpoint_path, device=device)

    def choose(obs: list[float], _env: DoomAgentEnv) -> int:
        action, _logprob, _value = trainer.model.act(
            obs,
            deterministic=deterministic,
            action_mask=_env.action_mask(),
        )
        return action

    return await evaluate_skill_policy(
        policy_id=f"ppo:{checkpoint_path}",
        env_config=env_config,
        choose_action=choose,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        before_reset=before_reset,
    )


async def evaluate_random_policy(
    env_config: DoomEnvConfig,
    *,
    episodes: int = 1,
    max_steps: int | None = None,
    seed: int = 7,
    before_reset: Callable[[DoomAgentEnv, int], None] | None = None,
) -> PolicyEval:
    """Evaluates a uniform random high-level skill policy."""
    rng = random.Random(seed)
    action_count = len(ACTION_SCHEMA["actions"])

    def choose(_obs: list[float], _env: DoomAgentEnv) -> int:
        mask = _env.action_mask()
        choices = [index for index, allowed in enumerate(mask) if allowed]
        if not choices:
            return rng.randrange(action_count)
        return rng.choice(choices)

    return await evaluate_skill_policy(
        policy_id="random_skill",
        env_config=env_config,
        choose_action=choose,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        before_reset=before_reset,
    )


async def evaluate_heuristic_policy(
    env_config: DoomEnvConfig,
    *,
    episodes: int = 1,
    max_steps: int | None = None,
    seed: int = 7,
    before_reset: Callable[[DoomAgentEnv, int], None] | None = None,
) -> PolicyEval:
    """Evaluates the deterministic skill heuristic baseline."""

    def choose(_obs: list[float], env: DoomAgentEnv) -> int:
        if env._current_state is None:
            return 0
        return env.controller.heuristic_action_index(env._current_state)

    return await evaluate_skill_policy(
        policy_id="heuristic_skill",
        env_config=env_config,
        choose_action=choose,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        before_reset=before_reset,
    )


async def evaluate_skill_policy(
    *,
    policy_id: str,
    env_config: DoomEnvConfig,
    choose_action: Callable[[list[float], DoomAgentEnv], int],
    episodes: int,
    max_steps: int | None,
    seed: int,
    before_reset: Callable[[DoomAgentEnv, int], None] | None = None,
) -> PolicyEval:
    """Evaluates a high-level skill policy in the resettable environment."""
    env = DoomAgentEnv(env_config)
    episode_results: list[EpisodeEval] = []
    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            if before_reset is not None:
                before_reset(env, episode_index)
            obs = await env.reset(seed=episode_seed)
            start_tick = int(env._current_state.tick) if env._current_state is not None else 0
            total_reward = 0.0
            start_kills = (
                int(env._current_state.player.kills) if env._current_state is not None else 0
            )
            start_items = _state_player_int(env._current_state, "items")
            start_secrets = _state_player_int(env._current_state, "secrets")
            start_episode = _state_level_int(env._current_state, "episode")
            start_map = _state_level_int(env._current_state, "map")
            max_kills = start_kills
            max_items = start_items
            max_secrets = start_secrets
            kill_delta = 0
            item_delta = 0
            secret_delta = 0
            min_health = (
                int(env._current_state.player.health) if env._current_state is not None else 0
            )
            reset_context = getattr(env, "_last_reset_context", {})
            reset_source = (
                str(reset_context.get("source", ""))
                if isinstance(reset_context, dict)
                else ""
            )
            stuck_events = 0
            done_reason = None
            steps = 0
            step_limit = max_steps or env_config.max_steps
            for steps in range(1, step_limit + 1):
                action_index = choose_action(obs, env)
                step = await env.step(action_index)
                obs = step.observation
                total_reward += step.reward
                state = step.info.get("state", {})
                decision = step.info.get("decision", {})
                max_kills = max(max_kills, int(state.get("kills", 0)))
                max_items = max(max_items, _int_field(state, "items"))
                max_secrets = max(max_secrets, _int_field(state, "secrets"))
                transition = step.info.get("transition", {})
                if isinstance(transition, dict):
                    kill_delta += _int_field(transition, "kill_delta")
                    item_delta += _int_field(transition, "item_delta")
                    secret_delta += _int_field(transition, "secret_delta")
                min_health = min(min_health, int(state.get("health", 0)))
                if decision.get("stuck") and step.info.get("skill") == "recover_stuck":
                    stuck_events += 1
                if step.done:
                    done_reason = step.info.get("done_reason")
                    break
            end_tick = int(env._current_state.tick) if env._current_state is not None else start_tick
            end_episode = _state_level_int(env._current_state, "episode")
            end_map = _state_level_int(env._current_state, "map")
            level_completed = done_reason == "level_complete"
            death = done_reason == "death" or min_health <= 0
            max_kill_gain = max(0, max_kills - start_kills)
            max_item_gain = max(0, max_items - start_items)
            max_secret_gain = max(0, max_secrets - start_secrets)
            level_transition_delta = int(
                (end_episode, end_map) != (start_episode, start_map)
            )
            episode_results.append(
                EpisodeEval(
                    seed=episode_seed,
                    total_reward=round(total_reward, 4),
                    level_completed=level_completed,
                    death=death,
                    max_kills=max_kills,
                    min_health=min_health,
                    steps=steps,
                    steps_to_exit=end_tick - start_tick if level_completed else step_limit,
                    stuck_events=stuck_events,
                    done_reason=done_reason,
                    start_kills=start_kills,
                    kill_delta=kill_delta,
                    max_kill_gain=max_kill_gain,
                    max_items=max_items,
                    start_items=start_items,
                    item_delta=item_delta,
                    max_item_gain=max_item_gain,
                    max_secrets=max_secrets,
                    start_secrets=start_secrets,
                    secret_delta=secret_delta,
                    max_secret_gain=max_secret_gain,
                    start_episode=start_episode,
                    start_map=start_map,
                    end_episode=end_episode,
                    end_map=end_map,
                    level_transition_delta=level_transition_delta,
                    reset_source=reset_source,
                )
            )
    finally:
        await env.close()
    return _aggregate(policy_id, episode_results)


def decide_promotion(
    *,
    candidate: PolicyEval,
    baseline: PolicyEval,
    min_completion_delta: float = 0.0,
    min_kill_delta: float = 0.0,
    min_reward_delta: float = 0.0,
    min_completion_rate: float = 1.0,
    min_mean_kills: float = 1.0,
) -> PromotionDecision:
    """Runs the promotion gate over two evaluations."""
    return PromotionGate(
        min_completion_delta=min_completion_delta,
        min_kill_delta=min_kill_delta,
        min_reward_delta=min_reward_delta,
        min_completion_rate=min_completion_rate,
        min_mean_kills=min_mean_kills,
    ).decide(candidate=candidate.result, baseline=baseline.result)


def _aggregate(policy_id: str, episodes: list[EpisodeEval]) -> PolicyEval:
    count = max(1, len(episodes))
    completions = sum(1 for episode in episodes if episode.level_completed)
    survivors = sum(1 for episode in episodes if not episode.death)
    result = EvaluationResult(
        policy_id=policy_id,
        level_completion_rate=completions / count,
        mean_kills=sum(_episode_earned_kills(episode) for episode in episodes) / count,
        survival_rate=survivors / count,
        mean_steps_to_exit=sum(episode.steps_to_exit for episode in episodes) / count,
        mean_stuck_events=sum(episode.stuck_events for episode in episodes) / count,
        episode_count=len(episodes),
        mean_reward=sum(episode.total_reward for episode in episodes) / count,
    )
    return PolicyEval(result=result, episodes=episodes)


def _episode_earned_kills(episode: EpisodeEval) -> int:
    """Returns kills earned after episode reset, excluding restored snapshot state."""
    absolute_gain = max(0, int(episode.max_kills) - int(getattr(episode, "start_kills", 0)))
    return max(int(getattr(episode, "kill_delta", 0)), int(episode.max_kill_gain), absolute_gain)


def _int_field(values: dict[str, Any], key: str) -> int:
    try:
        return int(values.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _state_player_int(state: Any, key: str) -> int:
    if state is None:
        return 0
    player = getattr(state, "player", None)
    if player is None:
        return 0
    try:
        return int(getattr(player, key, 0))
    except (TypeError, ValueError):
        return 0


def _state_level_int(state: Any, key: str) -> int:
    if state is None:
        return 0
    level = getattr(state, "level", None)
    if level is None:
        return 0
    try:
        return int(getattr(level, key, 0))
    except (TypeError, ValueError):
        return 0
