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

EXPERT_TO_PPO_SKILL_ACTION = {
    "aim_at_enemy": "engage",
    "close_visible_enemy": "engage",
    "pursue_last_contact_corridor": "engage",
    "skirt_visible_enemy": "engage",
    "fire_on_enemy": "fire",
    "fire_on_shootable_target": "fire",
    "hold_attack": "fire",
    "defensive_fire": "fire",
    "critical_retreat": "retreat",
    "retreat": "retreat",
    "hunt_known_enemy": "seek_enemy",
    "seek_known_enemy": "seek_enemy",
    "approach_nearby_use_line": "open_use_line",
    "open_or_probe": "open_use_line",
    "retry_exit_assist_door": "open_use_line",
    "turn_to_exit_assist_door": "open_use_line",
    "turn_to_nearby_use_line": "open_use_line",
    "turn_to_retry_exit_assist_door": "open_use_line",
    "turn_to_use_line": "open_use_line",
    "use_ahead": "open_use_line",
    "use_blocking_line": "open_use_line",
    "use_directional_line": "open_use_line",
    "use_exit_assist_door": "open_use_line",
    "use_nearby_line": "open_use_line",
    "approach_progression_line": "route_progression",
    "cross_progression_line": "route_progression",
    "explore_frontier": "route_progression",
    "route_to_progression_line": "route_progression",
    "sidestep_left": "route_progression",
    "sidestep_right": "route_progression",
    "turn_to_progression_line": "route_progression",
    "wall_follow_left": "route_progression",
    "wall_follow_right": "route_progression",
    "break_cell_loop": "recover_stuck",
    "turn_from_block": "recover_stuck",
    "unstick_backtrack": "recover_stuck",
    "unstick_turn": "recover_stuck",
    "press_exit_switch": "press_exit",
    "push_exit_switch": "press_exit",
    "turn_to_exit_switch": "press_exit",
}

PPO_ACTION_INDEX_BY_SKILL = {
    skill: index for index, skill in enumerate(PPO_SKILL_ACTIONS)
}

OBSERVATION_SCHEMA = {
    "schema": "restfuldoom.observation.v1",
    "feature_names": FEATURE_NAMES,
}

ACTION_SCHEMA = {
    "schema": "restfuldoom.skill_action.v1",
    "actions": PPO_SKILL_ACTIONS,
}


def map_expert_skill_to_ppo_action(skill: str) -> int | None:
    """Maps a rich brain skill label to the stable PPO skill action index."""
    ppo_skill = EXPERT_TO_PPO_SKILL_ACTION.get(skill)
    if ppo_skill is None and skill in PPO_ACTION_INDEX_BY_SKILL:
        ppo_skill = skill
    if ppo_skill is None:
        return None
    return PPO_ACTION_INDEX_BY_SKILL[ppo_skill]
