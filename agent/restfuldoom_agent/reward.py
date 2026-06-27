"""Goal and reward helpers for Doom agent experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any


@dataclass(frozen=True)
class Goal:
    """Defines a simple map-space objective."""

    target_x_fp: int | None = None
    target_y_fp: int | None = None
    kill_weight: float = 5.0
    damage_weight: float = 0.0
    enemy_distance_weight: float = 0.0
    item_weight: float = 1.0
    secret_weight: float = 2.0
    health_weight: float = 0.05
    progress_weight: float = 0.001
    death_penalty: float = 25.0
    name: str = "custom"


GOAL_PRESETS: dict[str, Goal] = {
    "survival": Goal(
        kill_weight=1.0,
        damage_weight=0.02,
        enemy_distance_weight=0.0,
        item_weight=0.5,
        secret_weight=0.5,
        health_weight=0.25,
        progress_weight=0.0,
        death_penalty=75.0,
        name="survival",
    ),
    "navigation": Goal(
        kill_weight=0.5,
        damage_weight=0.03,
        enemy_distance_weight=0.002,
        item_weight=0.25,
        secret_weight=0.25,
        health_weight=0.05,
        progress_weight=0.002,
        death_penalty=35.0,
        name="navigation",
    ),
    "combat": Goal(
        kill_weight=10.0,
        damage_weight=0.2,
        enemy_distance_weight=0.01,
        item_weight=0.5,
        secret_weight=0.5,
        health_weight=0.03,
        progress_weight=0.0,
        death_penalty=30.0,
        name="combat",
    ),
    "item_collection": Goal(
        kill_weight=0.5,
        damage_weight=0.02,
        enemy_distance_weight=0.0,
        item_weight=6.0,
        secret_weight=4.0,
        health_weight=0.05,
        progress_weight=0.0005,
        death_penalty=25.0,
        name="item_collection",
    ),
    "exit_seeking": Goal(
        kill_weight=0.25,
        damage_weight=0.02,
        enemy_distance_weight=0.001,
        item_weight=0.5,
        secret_weight=1.0,
        health_weight=0.05,
        progress_weight=0.004,
        death_penalty=40.0,
        name="exit_seeking",
    ),
}


def goal_preset(
    name: str,
    *,
    target_x_fp: int | None = None,
    target_y_fp: int | None = None,
) -> Goal:
    """Returns a named reward preset, optionally bound to a map target."""
    key = name.replace("-", "_").lower()
    try:
        preset = GOAL_PRESETS[key]
    except KeyError as error:
        available = ", ".join(sorted(GOAL_PRESETS))
        raise ValueError(f"unknown goal preset {name!r}; choose one of: {available}") from error

    return Goal(
        target_x_fp=target_x_fp,
        target_y_fp=target_y_fp,
        kill_weight=preset.kill_weight,
        damage_weight=preset.damage_weight,
        enemy_distance_weight=preset.enemy_distance_weight,
        item_weight=preset.item_weight,
        secret_weight=preset.secret_weight,
        health_weight=preset.health_weight,
        progress_weight=preset.progress_weight,
        death_penalty=preset.death_penalty,
        name=preset.name,
    )


@dataclass(frozen=True)
class TransitionReward:
    """Captures reward details for one transition."""

    reward: float
    kill_delta: int
    damage_delta: int
    enemy_distance_delta: float
    item_delta: int
    secret_delta: int
    health_delta: int
    progress_delta: float
    done: bool


class RewardEngine:
    """Computes shaped rewards from streamed Doom states."""

    def __init__(self, goal: Goal | None = None) -> None:
        self.goal = goal or Goal()

    def score(self, previous: Any | None, current: Any) -> TransitionReward:
        """Scores the transition into `current`."""
        if previous is None:
            return TransitionReward(0.0, 0, 0, 0.0, 0, 0, 0, 0.0, False)

        kill_delta = current.player.kills - previous.player.kills
        damage_delta = _enemy_damage_delta(previous, current)
        enemy_distance_delta = _nearest_enemy_distance_delta(previous, current)
        item_delta = current.player.items - previous.player.items
        secret_delta = current.player.secrets - previous.player.secrets
        health_delta = current.player.health - previous.player.health
        progress_delta = self._progress_delta(previous, current)
        done = current.player.health <= 0

        reward = (
            kill_delta * self.goal.kill_weight
            + damage_delta * self.goal.damage_weight
            + enemy_distance_delta * self.goal.enemy_distance_weight
            + item_delta * self.goal.item_weight
            + secret_delta * self.goal.secret_weight
            + health_delta * self.goal.health_weight
            + progress_delta * self.goal.progress_weight
        )
        if done:
            reward -= self.goal.death_penalty

        return TransitionReward(
            reward=reward,
            kill_delta=kill_delta,
            damage_delta=damage_delta,
            enemy_distance_delta=enemy_distance_delta,
            item_delta=item_delta,
            secret_delta=secret_delta,
            health_delta=health_delta,
            progress_delta=progress_delta,
            done=done,
        )

    def _progress_delta(self, previous: Any, current: Any) -> float:
        if self.goal.target_x_fp is None or self.goal.target_y_fp is None:
            return 0.0

        prior = previous.player.object.position
        now = current.player.object.position
        prior_dist = hypot(prior.x_fp - self.goal.target_x_fp, prior.y_fp - self.goal.target_y_fp)
        current_dist = hypot(now.x_fp - self.goal.target_x_fp, now.y_fp - self.goal.target_y_fp)
        return prior_dist - current_dist


def _enemy_damage_delta(previous: Any, current: Any) -> int:
    previous_health = _enemy_health_by_id(previous)
    current_health = _enemy_health_by_id(current)
    damage = 0
    for enemy_id, prior_health in previous_health.items():
        current_value = current_health.get(enemy_id)
        if current_value is not None and current_value < prior_health:
            damage += prior_health - current_value
    return damage


def _nearest_enemy_distance_delta(previous: Any, current: Any) -> float:
    previous_distance = _nearest_enemy_distance_units(previous)
    current_distance = _nearest_enemy_distance_units(current)
    if previous_distance is None or current_distance is None:
        return 0.0
    return previous_distance - current_distance


def _nearest_enemy_distance_units(state: Any) -> float | None:
    distances: list[float] = []
    for enemy in getattr(state, "enemies", []) or []:
        obj = getattr(enemy, "object", None)
        if obj is None:
            continue
        distance_fp = int(getattr(obj, "distance_fp", 0))
        if distance_fp > 0:
            distances.append(distance_fp / 65536.0)
    if not distances:
        return None
    return min(distances)


def _enemy_health_by_id(state: Any) -> dict[int, int]:
    health: dict[int, int] = {}
    for enemy in getattr(state, "enemies", []) or []:
        obj = getattr(enemy, "object", None)
        if obj is None:
            continue
        enemy_id = int(getattr(obj, "id", 0))
        if enemy_id:
            health[enemy_id] = int(getattr(obj, "health", 0))
    return health
