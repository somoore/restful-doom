"""Compact PPO transition traces for debugging learning behavior."""

from __future__ import annotations

from typing import Any

from .schemas import ACTION_SCHEMA, OBSERVATION_SCHEMA

LEARNING_TRACE_SCHEMA = "restfuldoom.learning_trace.v1"

_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "player": (
        "health_norm",
        "ammo_norm",
        "kills_norm",
        "items_norm",
        "x_units_norm",
        "y_units_norm",
        "angle_sin",
        "angle_cos",
    ),
    "combat": (
        "visible_enemies_norm",
        "enemy_count_norm",
        "has_enemy",
        "enemy_distance_norm",
        "enemy_angle_sin",
        "enemy_angle_cos",
        "enemy_threat_norm",
        "enemy_health_norm",
        "combat_has_target",
        "combat_target_enemy",
        "combat_target_distance_norm",
    ),
    "navigation": (
        "nav_forward_open",
        "nav_back_open",
        "nav_left_open",
        "nav_right_open",
        "nav_use_line_ahead",
        "nav_front_distance_norm",
        "nav_front_special_manual",
        "nav_front_special_exit",
        "nav_open_probe_ratio",
        "nav_use_probe_ratio",
        "nav_best_open_angle_norm",
        "topology_frontier_count_norm",
    ),
    "use_line": (
        "has_use_line",
        "use_line_distance_norm",
        "use_line_angle_sin",
        "use_line_angle_cos",
        "use_line_manual",
        "use_line_exit",
        "use_line_side",
        "use_line_front_distance_norm",
    ),
    "route": (
        "route_has_waypoint",
        "route_waypoint_distance_norm",
        "route_waypoint_angle_sin",
        "route_waypoint_angle_cos",
        "route_waypoint_priority_norm",
        "route_waypoint_exit",
        "route_waypoint_walk_trigger",
        "prev_route_progression",
        "prev_route_progress_norm",
        "route_waypoint_reached_recently",
        "route_waypoint_failed_recently",
        "failed_route_attempt_count_norm",
        "recent_route_progress_norm",
        "recent_route_failure_ratio",
    ),
    "memory": (
        "known_enemies_norm",
        "remembered_enemies_norm",
        "blocked_targets_norm",
    ),
    "temporal": (
        "prev_skill_engage",
        "prev_skill_fire",
        "prev_skill_seek_enemy",
        "prev_skill_open_use_line",
        "prev_skill_route_progression",
        "prev_skill_retreat",
        "prev_skill_recover_stuck",
        "prev_skill_press_exit",
        "prev_had_shootable_target",
        "same_skill_streak_norm",
        "delta_x_norm",
        "delta_y_norm",
        "movement_distance_norm",
        "enemy_distance_delta_norm",
        "route_distance_delta_norm",
        "same_cell_observation_streak_norm",
        "cell_changed_recently",
        "visible_enemy_seen_recently",
        "shootable_target_seen_recently",
    ),
    "survival": (
        "stuck",
        "sector_damaging",
        "sector_damage_norm",
        "sector_exit_damage",
        "sector_floor_height_norm",
        "sector_ceiling_height_norm",
    ),
}

_OUTCOME_KEYS = (
    "skill",
    "had_visible_enemy",
    "had_shootable_target",
    "first_visible_contact",
    "first_shootable_contact",
    "action_reward",
    "route_action_reward",
    "contact_reward",
    "visible_contact_distance_delta",
    "visible_contact_progress_reward",
    "done_reason",
)

_TRANSITION_KEYS = (
    "reward",
    "kill_delta",
    "damage_delta",
    "enemy_distance_delta",
    "item_delta",
    "secret_delta",
    "health_delta",
    "progress_delta",
)


def build_learning_trace(
    *,
    obs: list[float],
    action_mask: list[bool],
    action: int,
    reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds a compact, named trace for one PPO transition."""
    row_info = dict(info or {})
    action_names = list(ACTION_SCHEMA["actions"])
    selected_skill = _safe_action_name(action_names, action)
    available_skills = [
        action_names[index]
        for index, allowed in enumerate(action_mask[: len(action_names)])
        if allowed
    ]
    feature_values = _named_feature_values(obs)
    return {
        "schema": LEARNING_TRACE_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA["schema"],
        "action_schema": ACTION_SCHEMA["schema"],
        "selected_action": {
            "index": int(action),
            "skill": selected_skill,
            "available": _action_available(action, action_mask),
        },
        "available_skills": available_skills,
        "available_skill_count": len(available_skills),
        "observation": {
            "feature_count": len(obs),
            "groups": _grouped_features(feature_values),
        },
        "controller": {
            "executed_skill": row_info.get("skill"),
            "decision": _compact_decision(row_info.get("decision")),
        },
        "outcome": _outcome(row_info, reward=reward, done=done),
    }


def _safe_action_name(action_names: list[str], action: int) -> str:
    if 0 <= int(action) < len(action_names):
        return action_names[int(action)]
    return f"unknown:{int(action)}"


def _action_available(action: int, action_mask: list[bool]) -> bool:
    action_index = int(action)
    return 0 <= action_index < len(action_mask) and bool(action_mask[action_index])


def _named_feature_values(obs: list[float]) -> dict[str, float]:
    names = OBSERVATION_SCHEMA["feature_names"]
    return {
        name: _round_float(obs[index])
        for index, name in enumerate(names)
        if index < len(obs)
    }


def _grouped_features(feature_values: dict[str, float]) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {}
    for group_name, names in _FEATURE_GROUPS.items():
        values = {
            name: feature_values[name]
            for name in names
            if name in feature_values
        }
        if values:
            groups[group_name] = values
    return groups


def _compact_decision(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("skill", "stuck", "use_line", "enemy", "line", "ray"):
        if key not in decision:
            continue
        value = decision[key]
        if isinstance(value, dict):
            compact[key] = _compact_nested(value)
        else:
            compact[key] = value
    return compact


def _compact_nested(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "id",
        "line_id",
        "special",
        "side",
        "distance",
        "distance_units",
        "front_distance",
        "angle_delta",
        "health",
        "line_of_sight",
        "threat",
    )
    return {
        key: _round_value(value[key])
        for key in allowed
        if key in value
    }


def _outcome(info: dict[str, Any], *, reward: float, done: bool) -> dict[str, Any]:
    outcome = {
        "reward": _round_float(reward),
        "done": bool(done),
    }
    for key in _OUTCOME_KEYS:
        if key in info:
            outcome[key] = _round_value(info[key])
    transition = info.get("transition")
    if isinstance(transition, dict):
        compact_transition = {
            key: _round_value(transition[key])
            for key in _TRANSITION_KEYS
            if key in transition
        }
        if compact_transition:
            outcome["transition"] = compact_transition
    route_outcome = info.get("route_outcome")
    if isinstance(route_outcome, dict):
        outcome["route_outcome"] = _compact_nested_route(route_outcome)
    return outcome


def _compact_nested_route(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "attempted",
        "reached",
        "failed",
        "progress_units",
        "distance_before",
        "distance_after",
    )
    return {
        key: _round_value(value[key])
        for key in allowed
        if key in value
    }


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, list):
        return [_round_value(item) for item in value]
    if isinstance(value, tuple):
        return [_round_value(item) for item in value]
    return value


def _round_float(value: float) -> float:
    return round(float(value), 6)
