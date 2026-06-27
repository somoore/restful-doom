"""Shared agent training schemas.

These dictionaries are intentionally machine-readable.  Checkpoints, rollout
buffers, and exported training jobs carry them so a cloud worker can understand
the observation/action contract without importing the controller code.
"""

from __future__ import annotations

from typing import Any

from .skill_policy import FEATURE_NAMES as TACTICAL_FEATURE_NAMES

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

ACTION_HISTORY_FEATURE_NAMES = [
    *(f"prev_skill_{skill}" for skill in PPO_SKILL_ACTIONS),
    "prev_had_shootable_target",
    "same_skill_streak_norm",
    "prev_route_progression",
    "prev_route_progress_norm",
    "route_waypoint_reached_recently",
    "route_waypoint_failed_recently",
    "failed_route_attempt_count_norm",
]

PPO_FEATURE_NAMES = [
    *TACTICAL_FEATURE_NAMES,
    *ACTION_HISTORY_FEATURE_NAMES,
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

ACTION_DEFINITIONS = [
    {
        "index": 0,
        "skill": "engage",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Turn toward, approach, or strafe around a visible enemy.",
        "controller_entrypoint": "SkillController._execute_skill('engage')",
        "primary_signal": "visible enemy distance and threat",
        "fallback": "explore when no enemy is visible",
    },
    {
        "index": 1,
        "skill": "fire",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Fire only when the combat probe reports a valid enemy target.",
        "controller_entrypoint": "SkillController._execute_skill('fire')",
        "primary_signal": "combat_has_target and combat_target_enemy",
        "fallback": "aim at visible enemy or send one shoot probe",
    },
    {
        "index": 2,
        "skill": "seek_enemy",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Route toward the best remembered or known enemy.",
        "controller_entrypoint": "SkillController._execute_skill('seek_enemy')",
        "primary_signal": "known and remembered enemy memory",
        "fallback": "explore when no enemy memory is usable",
    },
    {
        "index": 3,
        "skill": "open_use_line",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Turn toward and activate nearby doors, switches, or use lines.",
        "controller_entrypoint": "SkillController._execute_skill('open_use_line')",
        "primary_signal": "usable-line probes and manual line specials",
        "fallback": "press use ahead",
    },
    {
        "index": 4,
        "skill": "route_progression",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Move toward progression lines or open exploratory space.",
        "controller_entrypoint": "SkillController._execute_skill('route_progression')",
        "primary_signal": "progression line priorities and navigation probes",
        "fallback": "explore",
    },
    {
        "index": 5,
        "skill": "retreat",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Back up or strafe away from nearby threats.",
        "controller_entrypoint": "SkillController._execute_skill('retreat')",
        "primary_signal": "low health or close visible enemy",
        "fallback": "backward movement",
    },
    {
        "index": 6,
        "skill": "recover_stuck",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Run the deterministic unstuck routine.",
        "controller_entrypoint": "SkillController._execute_skill('recover_stuck')",
        "primary_signal": "stuck and blocked-target indicators",
        "fallback": "turn/backtrack sequence",
    },
    {
        "index": 7,
        "skill": "press_exit",
        "kind": "code_defined_option",
        "learned": False,
        "execution_owner": "BrainPolicy",
        "role": "Prioritize exit switch approach and activation.",
        "controller_entrypoint": "SkillController._execute_skill('press_exit')",
        "primary_signal": "exit-line affordances",
        "fallback": "use probe",
    },
]

def encode_action_history_features(
    *,
    previous_action_index: int | None,
    previous_had_shootable_target: bool,
    same_skill_streak: int,
    previous_route_progression: bool = False,
    previous_route_progress_units: float = 0.0,
    route_waypoint_reached_recently: bool = False,
    route_waypoint_failed_recently: bool = False,
    failed_route_attempt_count: int = 0,
) -> list[float]:
    """Encode stateful PPO action-history features."""
    one_hot = [0.0 for _ in PPO_SKILL_ACTIONS]
    if previous_action_index is not None and 0 <= previous_action_index < len(one_hot):
        one_hot[previous_action_index] = 1.0
    return [
        *one_hot,
        1.0 if previous_had_shootable_target else 0.0,
        max(0.0, min(1.0, float(same_skill_streak) / 8.0)),
        1.0 if previous_route_progression else 0.0,
        max(-1.0, min(1.0, float(previous_route_progress_units) / 256.0)),
        1.0 if route_waypoint_reached_recently else 0.0,
        1.0 if route_waypoint_failed_recently else 0.0,
        max(0.0, min(1.0, float(failed_route_attempt_count) / 8.0)),
    ]


def pad_observation_features(features: list[float]) -> list[float]:
    """Pad older tactical-only feature rows to the current PPO observation contract."""
    if len(features) == len(PPO_FEATURE_NAMES):
        return list(features)
    if len(features) == len(TACTICAL_FEATURE_NAMES):
        return [
            *features,
            *encode_action_history_features(
                previous_action_index=None,
                previous_had_shootable_target=False,
                same_skill_streak=0,
                previous_route_progression=False,
                previous_route_progress_units=0.0,
                route_waypoint_reached_recently=False,
                route_waypoint_failed_recently=False,
                failed_route_attempt_count=0,
            ),
        ]
    raise ValueError(
        "feature vector length does not match tactical or PPO observation schema: "
        f"got {len(features)}, expected {len(TACTICAL_FEATURE_NAMES)} or "
        f"{len(PPO_FEATURE_NAMES)}"
    )


def _feature_descriptors(names: list[str], *, source: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "source": source,
            "range": "-1..1 for normalized values, 0..1 for flags/one-hot values",
            "meaning": _feature_meaning(name),
        }
        for name in names
    ]


def _feature_meaning(name: str) -> str:
    if name.startswith("prev_skill_"):
        return f"One-hot flag: previous PPO macro action was {name.removeprefix('prev_skill_')}."
    meanings = {
        "health_norm": "Player health normalized by 100.",
        "ammo_norm": "Bullet ammo normalized by 80.",
        "kills_norm": "Current kill count normalized by 10.",
        "items_norm": "Current item count normalized by 20.",
        "x_units_norm": "Player X map position in Doom units normalized by 4096.",
        "y_units_norm": "Player Y map position in Doom units normalized by 4096.",
        "angle_sin": "Sine of player facing angle.",
        "angle_cos": "Cosine of player facing angle.",
        "visible_enemies_norm": "Line-of-sight enemy count normalized by 8.",
        "known_enemies_norm": "Current protobuf enemy count known to the engine normalized by 16.",
        "remembered_enemies_norm": "Recently remembered enemy count from AgentMemory normalized by 16.",
        "enemy_count_norm": "Total current enemy count normalized by 32.",
        "has_enemy": "Flag that the selected enemy feature slot is occupied.",
        "enemy_distance_norm": "Selected enemy distance normalized by 2400 Doom units.",
        "enemy_angle_sin": "Sine of signed angle to selected enemy.",
        "enemy_angle_cos": "Cosine of signed angle to selected enemy.",
        "enemy_threat_norm": "Hand-built enemy threat score normalized by 10.",
        "enemy_health_norm": "Selected enemy health normalized by 100.",
        "combat_has_target": "Engine combat probe has a shootable target.",
        "combat_target_enemy": "Engine combat probe target is an enemy.",
        "combat_target_distance_norm": "Combat probe target distance normalized by 2400 Doom units.",
        "nav_forward_open": "Navigation probe says forward movement is open.",
        "nav_back_open": "Navigation probe says backward movement is open.",
        "nav_left_open": "Navigation probe says left movement is open.",
        "nav_right_open": "Navigation probe says right movement is open.",
        "nav_use_line_ahead": "Navigation probe sees a usable line ahead.",
        "nav_front_distance_norm": "Front blocking distance normalized by 512 Doom units.",
        "nav_front_special_manual": "Front blocking line has a manual-use special.",
        "nav_front_special_exit": "Front blocking line is an exit special.",
        "nav_open_probe_ratio": "Fraction of direction probes that are open.",
        "nav_use_probe_ratio": "Fraction of direction probes with use lines.",
        "nav_best_open_angle_norm": "Closest open probe angle normalized by 90 degrees.",
        "has_use_line": "At least one nearby use line is available.",
        "use_line_distance_norm": "Nearest use-line distance normalized by 1600 Doom units.",
        "use_line_angle_sin": "Sine of angle to nearest use line.",
        "use_line_angle_cos": "Cosine of angle to nearest use line.",
        "use_line_manual": "Nearest use line has a manual-use special.",
        "use_line_exit": "Nearest use line is an exit special.",
        "use_line_side": "Doom line side for the nearest use line.",
        "use_line_front_distance_norm": "Distance to a point in front of the use line normalized by 1600.",
        "stuck": "Controller-level stuck detector is active.",
        "blocked_targets_norm": "Current blocked target count normalized by 16.",
        "sector_damaging": "Current sector applies periodic floor or exit damage.",
        "sector_damage_norm": "Current sector damage per 32 tics normalized by 20.",
        "sector_exit_damage": "Current sector is the special exit-damage sector.",
        "sector_floor_height_norm": "Current sector floor height normalized by 1024 Doom units.",
        "sector_ceiling_height_norm": "Current sector ceiling height normalized by 1024 Doom units.",
        "route_has_waypoint": "Navigation probe exposes a progression waypoint.",
        "route_waypoint_distance_norm": "Progression waypoint distance normalized by 2600 Doom units.",
        "route_waypoint_angle_sin": "Sine of angle to the progression waypoint.",
        "route_waypoint_angle_cos": "Cosine of angle to the progression waypoint.",
        "route_waypoint_priority_norm": "Progression waypoint priority normalized by 4.",
        "route_waypoint_exit": "Progression waypoint is an exit line.",
        "route_waypoint_walk_trigger": "Progression waypoint is a walk-trigger line.",
        "prev_had_shootable_target": "Previous macro-step had a shootable enemy target.",
        "same_skill_streak_norm": "Consecutive same PPO skill selections normalized by 8.",
        "prev_route_progression": "Previous macro-step selected route_progression.",
        "prev_route_progress_norm": "Previous route-progression distance gain toward its waypoint normalized by 256 Doom units.",
        "route_waypoint_reached_recently": "Previous route-progression macro-step reached or crossed its waypoint.",
        "route_waypoint_failed_recently": "Previous route-progression macro-step stalled or moved away from its waypoint.",
        "failed_route_attempt_count_norm": "Consecutive failed route-progression attempts normalized by 8.",
    }
    return meanings.get(name, "PPO observation feature.")


OBSERVATION_SCHEMA = {
    "schema": "restfuldoom.observation.v1",
    "source": "protobuf GameState plus AgentMemory plus previous macro-action",
    "feature_names": PPO_FEATURE_NAMES,
    "base_feature_names": TACTICAL_FEATURE_NAMES,
    "action_history_feature_names": ACTION_HISTORY_FEATURE_NAMES,
    "source_groups": [
        {
            "name": "protobuf_state",
            "producer": "Doom gRPC GameState",
            "features": [
                "health_norm",
                "ammo_norm",
                "kills_norm",
                "items_norm",
                "x_units_norm",
                "y_units_norm",
                "angle_sin",
                "angle_cos",
                "visible_enemies_norm",
                "known_enemies_norm",
                "enemy_count_norm",
                "combat_has_target",
                "combat_target_enemy",
                "nav_forward_open",
                "nav_back_open",
                "nav_left_open",
                "nav_right_open",
                "nav_use_line_ahead",
                "has_use_line",
                "sector_damaging",
                "sector_damage_norm",
                "sector_exit_damage",
                "sector_floor_height_norm",
                "sector_ceiling_height_norm",
                "route_has_waypoint",
                "route_waypoint_distance_norm",
                "route_waypoint_angle_sin",
                "route_waypoint_angle_cos",
                "route_waypoint_priority_norm",
                "route_waypoint_exit",
                "route_waypoint_walk_trigger",
            ],
        },
        {
            "name": "memory_queries",
            "producer": "AgentMemory.remembered_enemies and BrainPolicy blocked-target state",
            "features": [
                "remembered_enemies_norm",
                "blocked_targets_norm",
            ],
        },
        {
            "name": "controller_state",
            "producer": "SkillController and BrainPolicy transient episode state",
            "features": [
                "stuck",
                *ACTION_HISTORY_FEATURE_NAMES,
            ],
        },
    ],
    "learning_readiness": {
            "strengths": [
                "combat and navigation affordances are structured protobuf fields",
                "memory-derived enemy recall gives PPO a target when line of sight is lost",
                "macro-action history tells PPO whether it just ignored or used a shootable target",
                "current-sector hazard fields tell PPO when navigation is causing floor damage",
                "route-waypoint fields give spawn-to-contact training an explicit progression target",
                "route-outcome history tells PPO whether the last progression decision reached, stalled, or moved toward its waypoint",
            ],
            "known_gaps": [
                "no compact topological map graph",
                "no recurrent state beyond one previous macro action",
                "no projectile or incoming-damage predictor",
            "reset seeds are labels until the server reports seed_applied=true",
        ],
        "next_feature_candidates": [
                "recent_damage_window_norm",
                "enemy_projectile_threat_norm",
                "topology_frontier_count_norm",
                "route_waypoint_repeat_count_norm",
            ],
    },
    "feature_descriptors": [
        *_feature_descriptors(TACTICAL_FEATURE_NAMES, source="protobuf_or_memory"),
        *_feature_descriptors(ACTION_HISTORY_FEATURE_NAMES, source="macro_step_history"),
    ],
}

DECISION_CYCLE_SCHEMA = {
    "schema": "restfuldoom.decision_cycle.v1",
    "clock": "one learned decision per bounded macro-step, not one raw ticcmd per Doom tic",
    "layers": [
        {
            "name": "decision_layer",
            "owner": "PPOTrainer ActorCritic, SkillPolicyModel, or BrainPolicy",
            "input": "restfuldoom.observation.v1 feature vector",
            "output": "integer index into restfuldoom.skill_action.v1 actions",
        },
        {
            "name": "fast_controller",
            "owner": "SkillController plus BrainPolicy",
            "input": "selected skill index plus latest protobuf GameState",
            "output": "protobuf PlayerAction with optional raw ticcmd overlay",
        },
        {
            "name": "environment",
            "owner": "DoomAgentEnv plus DoomAgentClient",
            "input": "PlayerAction stream",
            "output": "next GameState, transition reward, terminal flag, trace metadata",
        },
    ],
    "handshake": [
        "DoomAgentEnv.action_mask() derives feasible skills from the current state.",
        "ActorCritic samples or selects one skill under that mask.",
        "SkillController.action_for() converts the skill into one concrete PlayerAction.",
        "DoomAgentEnv.step() sends the action and waits through bounded duration_tics.",
        "RewardEngine scores the resulting GameState transitions.",
        "SkillController.record_action_history() writes macro-action context for the next observation.",
    ],
    "trace_fields": {
        "rollout_record.action": "selected PPO skill index",
        "rollout_record.action_mask": "feasible-skill mask used for sampling and PPO logprobs",
        "rollout_record.info.decision": "controller decision details and selected primitive",
        "rollout_record.info.decision_cycle": "tick range and schema markers for this macro-step",
    },
}

MEMORY_CONTRACT = {
    "schema": "restfuldoom.agent_memory_contract.v1",
    "memory_schema": "restfuldoom.agent_memory.v1",
    "default_path": "agent_memory/e1m1.json",
    "role": "inspectable world ledger and training checkpoint index, not a neural hidden state",
    "sections": [
        {
            "name": "cells",
            "meaning": "coarse map-cell visits, enemy sightings, damage events, and last-seen ticks",
        },
        {
            "name": "enemies",
            "meaning": "enemy id to last position, health, distance, line of sight, and threat",
        },
        {
            "name": "episodes",
            "meaning": "recent rollout summaries used for audit and export",
        },
        {
            "name": "policy",
            "meaning": "best deterministic parameter set and promotion metadata",
        },
        {
            "name": "learned_policy",
            "meaning": "behavior-cloned skill selector metadata",
        },
        {
            "name": "ppo_policy",
            "meaning": "latest PPO checkpoint metadata, reward config, rollout summary, and eval history",
        },
        {
            "name": "ppo_checkpoints",
            "meaning": "PPO checkpoint lineage for resume/export",
        },
    ],
    "query_paths": [
        {
            "method": "AgentMemory.best_params()",
            "reader": "run_brain_training",
            "returns": "promoted BrainPolicyParams",
        },
        {
            "method": "AgentMemory.remembered_enemies(x_units, y_units, tick, max_age_tics)",
            "reader": "extract_features",
            "returns": "recent enemy sightings sorted by current distance while rejecting stale/future ticks",
        },
        {
            "method": "AgentMemory.summary()",
            "reader": "brain_agent --memory-summary and MCP brain_memory",
            "returns": "compact diagnostics for Codex/operator inspection",
        },
    ],
    "update_paths": [
        {
            "method": "AgentMemory.record_step(features, decision, reward, stats)",
            "writer": "run_brain_episode",
            "updates": "cells, enemies, per-episode stats, lessons",
        },
        {
            "method": "AgentMemory.finish_episode(stats, params, promoted)",
            "writer": "run_brain_episode",
            "updates": "episodes, policy best score/params, lessons",
        },
        {
            "method": "ppo_agent._record_ppo_checkpoint()",
            "writer": "ppo_agent train",
            "updates": "ppo_policy and ppo_checkpoints",
        },
        {
            "method": "ppo_agent._record_eval_history()",
            "writer": "ppo_agent eval",
            "updates": "ppo_policy.eval_history",
        },
        {
            "method": "train_skill_policy_from_memory()",
            "writer": "brain_agent --train-skill-model",
            "updates": "learned_policy",
        },
    ],
}

ACTION_SCHEMA = {
    "schema": "restfuldoom.skill_action.v1",
    "actions": PPO_SKILL_ACTIONS,
    "definitions": ACTION_DEFINITIONS,
    "representation": {
        "current": "code-defined options implemented by SkillController._execute_skill",
        "learned_now": "PPO learns when to choose each option",
        "learned_later": "movement primitives can become learned/subpolicy-backed after the option set stabilizes",
    },
    "mask_semantics": {
        "schema": "restfuldoom.skill_action_mask.v1",
        "source": "protobuf navigation/combat affordances plus AgentMemory",
        "meaning": "True entries are feasible skills for the current macro-step; PPO sampling and update logprobs use this mask.",
    },
}


def map_expert_skill_to_ppo_action(skill: str) -> int | None:
    """Maps a rich brain skill label to the stable PPO skill action index."""
    ppo_skill = EXPERT_TO_PPO_SKILL_ACTION.get(skill)
    if ppo_skill is None and skill in PPO_ACTION_INDEX_BY_SKILL:
        ppo_skill = skill
    if ppo_skill is None:
        return None
    return PPO_ACTION_INDEX_BY_SKILL[ppo_skill]
