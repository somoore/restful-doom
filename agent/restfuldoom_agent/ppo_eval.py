"""Evaluation helpers for PPO skill checkpoints."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .env import ACTION_SCHEMA, OBSERVATION_SCHEMA, DoomAgentEnv, DoomEnvConfig
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
    steps_to_required_kills: int = 0
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
    seed_applied: bool = False
    skill_counts: dict[str, int] = field(default_factory=dict)
    visible_enemy_steps: int = 0
    first_visible_contacts: int = 0
    first_shootable_contacts: int = 0
    shootable_target_steps: int = 0
    fire_on_shootable_steps: int = 0
    missed_shootable_fire_steps: int = 0
    damage_delta: int = 0
    invalid_action_steps: int = 0
    selected_disallowed_steps: int = 0
    action_mask_fallback_steps: int = 0
    allowed_skill_filter_steps: int = 0
    allowed_skill_filter_fallback_steps: int = 0
    strict_allowed_skill_filter_steps: int = 0
    strict_allowed_skill_fallback_steps: int = 0
    allowed_skill_filter_fallback_skills: dict[str, int] = field(default_factory=dict)
    snapshot_verification_failures: int = 0
    route_action_reward: float = 0.0
    route_attempt_steps: int = 0
    route_reached_steps: int = 0
    route_failed_steps: int = 0
    route_progress_units: float = 0.0
    exit_route_attempt_steps: int = 0
    exit_route_reached_steps: int = 0
    exit_route_failed_steps: int = 0
    exit_route_progress_units: float = 0.0


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
    trace_path: str | Path | None = None,
) -> PolicyEval:
    """Evaluates a PPO checkpoint in the Doom skill environment."""
    trainer = PPOTrainer.load_checkpoint(
        checkpoint_path,
        device=device,
        target_obs_dim=len(OBSERVATION_SCHEMA["feature_names"]),
        target_action_dim=len(ACTION_SCHEMA["actions"]),
    )

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
        trace_path=trace_path,
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
    trace_path: str | Path | None = None,
) -> PolicyEval:
    """Evaluates a high-level skill policy in the resettable environment."""
    env = DoomAgentEnv(env_config)
    episode_results: list[EpisodeEval] = []
    trace_handle = _open_trace(trace_path, policy_id)
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
            seed_applied = bool(
                reset_context.get("seed_applied", False)
                if isinstance(reset_context, dict)
                else False
            )
            snapshot_verification_failures = int(
                _reset_context_snapshot_verification_failed(reset_context)
            )
            stuck_events = 0
            done_reason = None
            steps = 0
            skill_counts: dict[str, int] = {}
            visible_enemy_steps = 0
            first_visible_contacts = 0
            first_shootable_contacts = 0
            shootable_target_steps = 0
            fire_on_shootable_steps = 0
            missed_shootable_fire_steps = 0
            damage_delta = 0
            invalid_action_steps = 0
            selected_disallowed_steps = 0
            action_mask_fallback_steps = 0
            allowed_skill_filter_steps = 0
            allowed_skill_filter_fallback_steps = 0
            strict_allowed_skill_filter_steps = 0
            strict_allowed_skill_fallback_steps = 0
            allowed_skill_filter_fallback_skills: dict[str, int] = {}
            route_action_reward = 0.0
            route_attempt_steps = 0
            route_reached_steps = 0
            route_failed_steps = 0
            route_progress_units = 0.0
            exit_route_attempt_steps = 0
            exit_route_reached_steps = 0
            exit_route_failed_steps = 0
            exit_route_progress_units = 0.0
            step_limit = max_steps or env_config.max_steps
            for steps in range(1, step_limit + 1):
                action_mask = env.action_mask()
                action_index = choose_action(obs, env)
                if not _action_allowed(action_mask, action_index):
                    invalid_action_steps += 1
                step = await env.step(action_index)
                if (
                    step.info.get("action_mask_enforced")
                    and not bool(step.info.get("action_mask_requested_allowed", True))
                ):
                    selected_disallowed_steps += 1
                if step.info.get("action_mask_fallback_applied"):
                    action_mask_fallback_steps += 1
                _write_trace_step(
                    trace_handle,
                    policy_id=policy_id,
                    episode_index=episode_index,
                    seed=episode_seed,
                    step_index=steps,
                    action_index=action_index,
                    action_mask=action_mask,
                    reward=step.reward,
                    done=step.done,
                    observation=step.observation,
                    info=step.info,
                )
                obs = step.observation
                total_reward += step.reward
                state = step.info.get("state", {})
                decision = step.info.get("decision", {})
                skill = str(step.info.get("skill", "unknown"))
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
                had_visible_enemy = bool(step.info.get("had_visible_enemy"))
                had_shootable_target = bool(step.info.get("had_shootable_target"))
                if had_visible_enemy:
                    visible_enemy_steps += 1
                if had_shootable_target:
                    shootable_target_steps += 1
                    if skill == "fire":
                        fire_on_shootable_steps += 1
                    else:
                        missed_shootable_fire_steps += 1
                if step.info.get("first_visible_contact"):
                    first_visible_contacts += 1
                if step.info.get("first_shootable_contact"):
                    first_shootable_contacts += 1
                max_kills = max(max_kills, int(state.get("kills", 0)))
                max_items = max(max_items, _int_field(state, "items"))
                max_secrets = max(max_secrets, _int_field(state, "secrets"))
                transition = step.info.get("transition", {})
                if isinstance(transition, dict):
                    kill_delta += _int_field(transition, "kill_delta")
                    item_delta += _int_field(transition, "item_delta")
                    secret_delta += _int_field(transition, "secret_delta")
                    damage_delta += _int_field(transition, "damage_delta")
                route_action_reward += float(step.info.get("route_action_reward", 0.0))
                route_outcome = step.info.get("route_outcome", {})
                if isinstance(route_outcome, dict):
                    if route_outcome.get("attempted"):
                        route_attempt_steps += 1
                        if route_outcome.get("exit"):
                            exit_route_attempt_steps += 1
                    if route_outcome.get("reached"):
                        route_reached_steps += 1
                        if route_outcome.get("exit"):
                            exit_route_reached_steps += 1
                    if route_outcome.get("failed"):
                        route_failed_steps += 1
                        if route_outcome.get("exit"):
                            exit_route_failed_steps += 1
                    progress_units = float(route_outcome.get("progress_units", 0.0))
                    route_progress_units += progress_units
                    if route_outcome.get("exit"):
                        exit_route_progress_units += progress_units
                min_health = min(min_health, int(state.get("health", 0)))
                if decision.get("stuck") and step.info.get("skill") == "recover_stuck":
                    stuck_events += 1
                action_mask_filter = step.info.get("action_mask_filter", {})
                if isinstance(action_mask_filter, dict) and action_mask_filter:
                    allowed_skill_filter_steps += 1
                    if action_mask_filter.get("strict"):
                        strict_allowed_skill_filter_steps += 1
                    if action_mask_filter.get("fallback_applied"):
                        allowed_skill_filter_fallback_steps += 1
                        if action_mask_filter.get("strict"):
                            strict_allowed_skill_fallback_steps += 1
                        fallback_skill = str(
                            action_mask_filter.get("fallback_skill", "unknown")
                        )
                        allowed_skill_filter_fallback_skills[fallback_skill] = (
                            allowed_skill_filter_fallback_skills.get(fallback_skill, 0)
                            + 1
                        )
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
                    steps_to_required_kills=steps
                    if done_reason == "required_kills"
                    else 0,
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
                    seed_applied=seed_applied,
                    skill_counts=dict(sorted(skill_counts.items())),
                    visible_enemy_steps=visible_enemy_steps,
                    first_visible_contacts=first_visible_contacts,
                    first_shootable_contacts=first_shootable_contacts,
                    shootable_target_steps=shootable_target_steps,
                    fire_on_shootable_steps=fire_on_shootable_steps,
                    missed_shootable_fire_steps=missed_shootable_fire_steps,
                    damage_delta=damage_delta,
                    invalid_action_steps=invalid_action_steps,
                    selected_disallowed_steps=selected_disallowed_steps,
                    action_mask_fallback_steps=action_mask_fallback_steps,
                    allowed_skill_filter_steps=allowed_skill_filter_steps,
                    allowed_skill_filter_fallback_steps=allowed_skill_filter_fallback_steps,
                    strict_allowed_skill_filter_steps=strict_allowed_skill_filter_steps,
                    strict_allowed_skill_fallback_steps=strict_allowed_skill_fallback_steps,
                    allowed_skill_filter_fallback_skills=dict(
                        sorted(allowed_skill_filter_fallback_skills.items())
                    ),
                    snapshot_verification_failures=snapshot_verification_failures,
                    route_action_reward=round(route_action_reward, 4),
                    route_attempt_steps=route_attempt_steps,
                    route_reached_steps=route_reached_steps,
                    route_failed_steps=route_failed_steps,
                    route_progress_units=round(route_progress_units, 4),
                    exit_route_attempt_steps=exit_route_attempt_steps,
                    exit_route_reached_steps=exit_route_reached_steps,
                    exit_route_failed_steps=exit_route_failed_steps,
                    exit_route_progress_units=round(exit_route_progress_units, 4),
                )
            )
    finally:
        if trace_handle is not None:
            trace_handle.close()
        await env.close()
    return _aggregate(policy_id, episode_results)


def _open_trace(trace_path: str | Path | None, policy_id: str) -> Any | None:
    """Opens an eval trace JSONL file and writes its schema header."""
    if trace_path is None:
        return None
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    json.dump(
        {
            "schema": "restfuldoom.ppo_eval_trace.v1",
            "policy_id": policy_id,
        },
        handle,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    return handle


def _write_trace_step(
    handle: Any | None,
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
) -> None:
    """Writes one candidate eval step when tracing is enabled."""
    if handle is None:
        return
    json.dump(
        {
            "policy_id": policy_id,
            "episode_index": int(episode_index),
            "seed": int(seed),
            "step_index": int(step_index),
            "action_index": int(action_index),
            "action_mask": [bool(value) for value in action_mask],
            "reward": round(float(reward), 6),
            "done": bool(done),
            "done_reason": info.get("done_reason"),
            "skill": info.get("skill"),
            "decision": info.get("decision", {}),
            "state": info.get("state", {}),
            "transition": info.get("transition", {}),
            "had_visible_enemy": bool(info.get("had_visible_enemy")),
            "had_shootable_target": bool(info.get("had_shootable_target")),
            "first_visible_contact": bool(info.get("first_visible_contact")),
            "first_shootable_contact": bool(info.get("first_shootable_contact")),
            "route_outcome": info.get("route_outcome", {}),
            "learning_trace": info.get("learning_trace", {}),
            "observation": list(observation),
        },
        handle,
        default=str,
        sort_keys=True,
    )
    handle.write("\n")


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
        mean_items=sum(_episode_earned_items(episode) for episode in episodes) / count,
        mean_item_gain=sum(int(episode.max_item_gain) for episode in episodes) / count,
        mean_secrets=sum(_episode_earned_secrets(episode) for episode in episodes) / count,
        mean_secret_gain=sum(int(episode.max_secret_gain) for episode in episodes) / count,
        snapshot_verification_failures=sum(
            int(episode.snapshot_verification_failures) for episode in episodes
        ),
        seed_applied_episode_count=sum(
            1 for episode in episodes if bool(episode.seed_applied)
        ),
        invalid_action_steps=sum(int(episode.invalid_action_steps) for episode in episodes),
        selected_disallowed_steps=sum(
            int(episode.selected_disallowed_steps) for episode in episodes
        ),
        action_mask_fallback_steps=sum(
            int(episode.action_mask_fallback_steps) for episode in episodes
        ),
        allowed_skill_filter_steps=sum(
            int(episode.allowed_skill_filter_steps) for episode in episodes
        ),
        allowed_skill_filter_fallback_steps=sum(
            int(episode.allowed_skill_filter_fallback_steps) for episode in episodes
        ),
        strict_allowed_skill_filter_steps=sum(
            int(episode.strict_allowed_skill_filter_steps) for episode in episodes
        ),
        strict_allowed_skill_fallback_steps=sum(
            int(episode.strict_allowed_skill_fallback_steps) for episode in episodes
        ),
        route_action_reward=round(
            sum(float(episode.route_action_reward) for episode in episodes),
            4,
        ),
        route_attempt_steps=sum(int(episode.route_attempt_steps) for episode in episodes),
        route_reached_steps=sum(int(episode.route_reached_steps) for episode in episodes),
        route_failed_steps=sum(int(episode.route_failed_steps) for episode in episodes),
        route_progress_units=round(
            sum(float(episode.route_progress_units) for episode in episodes),
            4,
        ),
        exit_route_attempt_steps=sum(
            int(episode.exit_route_attempt_steps) for episode in episodes
        ),
        exit_route_reached_steps=sum(
            int(episode.exit_route_reached_steps) for episode in episodes
        ),
        exit_route_failed_steps=sum(
            int(episode.exit_route_failed_steps) for episode in episodes
        ),
        exit_route_progress_units=round(
            sum(float(episode.exit_route_progress_units) for episode in episodes),
            4,
        ),
        reset_source_breakdown=_reset_source_breakdown(episodes),
    )
    return PolicyEval(result=result, episodes=episodes)


def _reset_source_breakdown(episodes: list[EpisodeEval]) -> dict[str, dict[str, Any]]:
    """Aggregates eval metrics by reset source so snapshot gates remain visible."""
    grouped: dict[str, list[EpisodeEval]] = {}
    for episode in episodes:
        source = str(getattr(episode, "reset_source", "") or "unknown")
        grouped.setdefault(source, []).append(episode)
    breakdown: dict[str, dict[str, Any]] = {}
    for source, source_episodes in sorted(grouped.items()):
        count = max(1, len(source_episodes))
        breakdown[source] = {
            "episode_count": len(source_episodes),
            "level_completion_rate": sum(
                1 for episode in source_episodes if episode.level_completed
            )
            / count,
            "level_transition_rate": sum(
                int(episode.level_transition_delta) for episode in source_episodes
            )
            / count,
            "mean_steps_to_exit": sum(
                episode.steps_to_exit for episode in source_episodes
            )
            / count,
            "survival_rate": sum(1 for episode in source_episodes if not episode.death)
            / count,
            "mean_reward": sum(episode.total_reward for episode in source_episodes)
            / count,
            "mean_kills": sum(
                _episode_earned_kills(episode) for episode in source_episodes
            )
            / count,
            "mean_items": sum(
                _episode_earned_items(episode) for episode in source_episodes
            )
            / count,
            "mean_item_gain": sum(
                int(episode.max_item_gain) for episode in source_episodes
            )
            / count,
            "mean_secrets": sum(
                _episode_earned_secrets(episode) for episode in source_episodes
            )
            / count,
            "mean_secret_gain": sum(
                int(episode.max_secret_gain) for episode in source_episodes
            )
            / count,
            "mean_stuck_events": sum(
                episode.stuck_events for episode in source_episodes
            )
            / count,
            "snapshot_verification_failures": sum(
                int(episode.snapshot_verification_failures)
                for episode in source_episodes
            ),
            "seed_applied_episode_count": sum(
                1 for episode in source_episodes if bool(episode.seed_applied)
            ),
            "invalid_action_steps": sum(
                int(episode.invalid_action_steps) for episode in source_episodes
            ),
            "selected_disallowed_steps": sum(
                int(episode.selected_disallowed_steps) for episode in source_episodes
            ),
            "action_mask_fallback_steps": sum(
                int(episode.action_mask_fallback_steps) for episode in source_episodes
            ),
            "allowed_skill_filter_steps": sum(
                int(episode.allowed_skill_filter_steps) for episode in source_episodes
            ),
            "allowed_skill_filter_fallback_steps": sum(
                int(episode.allowed_skill_filter_fallback_steps)
                for episode in source_episodes
            ),
            "strict_allowed_skill_filter_steps": sum(
                int(episode.strict_allowed_skill_filter_steps)
                for episode in source_episodes
            ),
            "strict_allowed_skill_fallback_steps": sum(
                int(episode.strict_allowed_skill_fallback_steps)
                for episode in source_episodes
            ),
            "route_action_reward": round(
                sum(float(episode.route_action_reward) for episode in source_episodes),
                4,
            ),
            "route_attempt_steps": sum(
                int(episode.route_attempt_steps) for episode in source_episodes
            ),
            "route_reached_steps": sum(
                int(episode.route_reached_steps) for episode in source_episodes
            ),
            "route_failed_steps": sum(
                int(episode.route_failed_steps) for episode in source_episodes
            ),
            "route_progress_units": round(
                sum(float(episode.route_progress_units) for episode in source_episodes),
                4,
            ),
            "exit_route_attempt_steps": sum(
                int(episode.exit_route_attempt_steps) for episode in source_episodes
            ),
            "exit_route_reached_steps": sum(
                int(episode.exit_route_reached_steps) for episode in source_episodes
            ),
            "exit_route_failed_steps": sum(
                int(episode.exit_route_failed_steps) for episode in source_episodes
            ),
            "exit_route_progress_units": round(
                sum(
                    float(episode.exit_route_progress_units)
                    for episode in source_episodes
                ),
                4,
            ),
        }
    return breakdown


def _episode_earned_kills(episode: EpisodeEval) -> int:
    """Returns kills earned after episode reset, excluding restored snapshot state."""
    absolute_gain = max(0, int(episode.max_kills) - int(getattr(episode, "start_kills", 0)))
    return max(int(getattr(episode, "kill_delta", 0)), int(episode.max_kill_gain), absolute_gain)


def _episode_earned_items(episode: EpisodeEval) -> int:
    """Returns items earned after episode reset, excluding restored snapshot state."""
    absolute_gain = max(0, int(episode.max_items) - int(getattr(episode, "start_items", 0)))
    return max(int(getattr(episode, "item_delta", 0)), int(episode.max_item_gain), absolute_gain)


def _episode_earned_secrets(episode: EpisodeEval) -> int:
    """Returns secrets earned after episode reset, excluding restored snapshot state."""
    absolute_gain = max(
        0,
        int(episode.max_secrets) - int(getattr(episode, "start_secrets", 0)),
    )
    return max(
        int(getattr(episode, "secret_delta", 0)),
        int(episode.max_secret_gain),
        absolute_gain,
    )


def _action_allowed(action_mask: list[bool], action_index: int) -> bool:
    return 0 <= int(action_index) < len(action_mask) and bool(action_mask[int(action_index)])


def _reset_context_snapshot_verification_failed(reset_context: Any) -> bool:
    if not isinstance(reset_context, dict):
        return False
    verification = reset_context.get("verification")
    if not isinstance(verification, dict):
        verification = reset_context.get("restored_state_verification")
    if not isinstance(verification, dict):
        verification = reset_context.get("snapshot_verification")
    if not isinstance(verification, dict):
        verification = reset_context.get("snapshot_restored_state_verification")
    if not isinstance(verification, dict):
        return False
    if "valid" in verification:
        return not bool(verification.get("valid"))
    if "ok" in verification:
        return not bool(verification.get("ok"))
    if "verified" in verification:
        return not bool(verification.get("verified"))
    return bool(verification.get("failed"))


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
