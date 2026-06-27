"""Shared agent training schemas."""

from __future__ import annotations

from .skill_policy import FEATURE_NAMES

PPO_SKILL_ACTIONS = [
    "engage",
    "fire",
    "seek_enemy",
    "open_use_line",
    "route_progression",
    "retreat",
    "recover_stuck",
    "press_exit",
]

OBSERVATION_SCHEMA = {
    "schema": "restfuldoom.observation.v1",
    "feature_names": FEATURE_NAMES,
}

ACTION_SCHEMA = {
    "schema": "restfuldoom.skill_action.v1",
    "actions": PPO_SKILL_ACTIONS,
}
