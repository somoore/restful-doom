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
    item_weight: float = 1.0
    secret_weight: float = 2.0
    health_weight: float = 0.05
    progress_weight: float = 0.001
    death_penalty: float = 25.0


@dataclass(frozen=True)
class TransitionReward:
    """Captures reward details for one transition."""

    reward: float
    kill_delta: int
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
            return TransitionReward(0.0, 0, 0, 0, 0, 0.0, False)

        kill_delta = current.player.kills - previous.player.kills
        item_delta = current.player.items - previous.player.items
        secret_delta = current.player.secrets - previous.player.secrets
        health_delta = current.player.health - previous.player.health
        progress_delta = self._progress_delta(previous, current)
        done = current.player.health <= 0

        reward = (
            kill_delta * self.goal.kill_weight
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
