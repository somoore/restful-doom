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

TEMPORAL_CONTEXT_FEATURE_NAMES = [
    "delta_x_norm",
    "delta_y_norm",
    "movement_distance_norm",
    "enemy_distance_delta_norm",
    "route_distance_delta_norm",
    "same_cell_observation_streak_norm",
    "cell_changed_recently",
    "visible_enemy_seen_recently",
    "shootable_target_seen_recently",
    "recent_route_progress_norm",
    "recent_route_failure_ratio",
]

PPO_FEATURE_NAMES = [
    *TACTICAL_FEATURE_NAMES,
    *ACTION_HISTORY_FEATURE_NAMES,
    *TEMPORAL_CONTEXT_FEATURE_NAMES,
]

EXPERT_TO_PPO_SKILL_ACTION = {
    "aim_at_enemy": "engage",
    "close_visible_enemy": "engage",
    "close_visible_contact": "engage",
    "pursue_last_contact_corridor": "engage",
    "skirt_visible_enemy": "engage",
    "ppo_close_visible_contact": "engage",
    "ppo_seek_visible_contact": "seek_enemy",
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


def encode_temporal_context_features(
    *,
    delta_x_units: float = 0.0,
    delta_y_units: float = 0.0,
    movement_distance_units: float = 0.0,
    enemy_distance_delta_units: float = 0.0,
    route_distance_delta_units: float = 0.0,
    same_cell_observation_streak: int = 0,
    cell_changed_recently: bool = False,
    visible_enemy_seen_recently: bool = False,
    shootable_target_seen_recently: bool = False,
    recent_route_progress_units: float = 0.0,
    recent_route_failure_ratio: float = 0.0,
) -> list[float]:
    """Encode bounded temporal context for PPO observations."""
    return [
        _clip(float(delta_x_units) / 512.0),
        _clip(float(delta_y_units) / 512.0),
        _clip(float(movement_distance_units) / 512.0, minimum=0.0),
        _clip(float(enemy_distance_delta_units) / 512.0),
        _clip(float(route_distance_delta_units) / 512.0),
        _clip(float(same_cell_observation_streak) / 8.0, minimum=0.0),
        1.0 if cell_changed_recently else 0.0,
        1.0 if visible_enemy_seen_recently else 0.0,
        1.0 if shootable_target_seen_recently else 0.0,
        _clip(float(recent_route_progress_units) / 512.0),
        _clip(float(recent_route_failure_ratio), minimum=0.0),
    ]


def _clip(value: float, *, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


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
            *encode_temporal_context_features(),
        ]
    legacy_action_history_len = len(TACTICAL_FEATURE_NAMES) + len(ACTION_HISTORY_FEATURE_NAMES)
    if len(features) == legacy_action_history_len:
        return [
            *features,
            *encode_temporal_context_features(),
        ]
    raise ValueError(
        "feature vector length does not match tactical or PPO observation schema: "
        f"got {len(features)}, expected {len(TACTICAL_FEATURE_NAMES)}, "
        f"{legacy_action_history_len}, or "
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
        "delta_x_norm": "Player X movement since the previous encoded PPO observation normalized by 512 units.",
        "delta_y_norm": "Player Y movement since the previous encoded PPO observation normalized by 512 units.",
        "movement_distance_norm": "Distance moved since the previous encoded PPO observation normalized by 512 units.",
        "enemy_distance_delta_norm": "Positive when nearest known enemy distance decreased since the previous observation, normalized by 512 units.",
        "route_distance_delta_norm": "Positive when route waypoint distance decreased since the previous observation, normalized by 512 units.",
        "same_cell_observation_streak_norm": "Consecutive encoded observations in the same coarse map cell normalized by 8.",
        "cell_changed_recently": "Flag that the current encoded observation moved into a different coarse map cell.",
        "visible_enemy_seen_recently": "Flag that any of the recent encoded observations saw a line-of-sight enemy.",
        "shootable_target_seen_recently": "Flag that any recent encoded observation or macro-step had a shootable enemy target.",
        "recent_route_progress_norm": "Rolling route-progression gain over recent macro-steps normalized by 512 Doom units.",
        "recent_route_failure_ratio": "Fraction of recent route-progression attempts that failed.",
    }
    return meanings.get(name, "PPO observation feature.")


OBSERVATION_SCHEMA = {
    "schema": "restfuldoom.observation.v1",
    "source": "protobuf GameState plus AgentMemory plus previous macro-action",
    "feature_names": PPO_FEATURE_NAMES,
    "base_feature_names": TACTICAL_FEATURE_NAMES,
    "action_history_feature_names": ACTION_HISTORY_FEATURE_NAMES,
    "temporal_context_feature_names": TEMPORAL_CONTEXT_FEATURE_NAMES,
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
        {
            "name": "temporal_context",
            "producer": "SkillController bounded observation and route-outcome history",
            "features": TEMPORAL_CONTEXT_FEATURE_NAMES,
        },
    ],
    "protobuf_to_observation_pipeline": [
        {
            "phase": "protobuf",
            "code": "Doom gRPC GameState",
            "payload": (
                "player, enemy, level, combat, navigation, sector, and "
                "route-waypoint affordances"
            ),
        },
        {
            "phase": "tactical_features",
            "code": "extract_features(state, memory, params)",
            "payload": (
                "normalized current state plus AgentMemory.remembered_enemies() "
                "query results"
            ),
        },
        {
            "phase": "base_vector",
            "code": "features_from_tactical(features)",
            "payload": "base protobuf/memory feature vector",
        },
        {
            "phase": "macro_history",
            "code": "encode_action_history_features(...)",
            "payload": "previous skill, shootable opportunity, and route outcome features",
        },
        {
            "phase": "temporal_context",
            "code": "encode_temporal_context_features(...)",
            "payload": "bounded movement, enemy-distance, route-distance, and contact trends",
        },
    ],
    "learning_readiness": {
        "rich_observation_definition": [
            "current live state must expose local combat and navigation affordances",
            "memory queries must preserve useful target context after line of sight drops",
            "macro-action history must show whether the last selected skill helped or failed",
            "temporal context must distinguish progress from stationary loops",
            "reset metadata must say whether a state is fresh, warmed up, or snapshot-restored",
        ],
        "strengths": [
            "combat and navigation affordances are structured protobuf fields",
            "memory-derived enemy recall gives PPO a target when line of sight is lost",
            "macro-action history tells PPO whether it just ignored or used a shootable target",
            "current-sector hazard fields tell PPO when navigation is causing floor damage",
            "route-waypoint fields give spawn-to-contact training an explicit progression target",
            "route-outcome history tells PPO whether the last progression decision reached, stalled, or moved toward its waypoint",
            "bounded temporal features expose recent movement, enemy-distance, route-distance, and shootable-contact trends",
        ],
        "known_gaps": [
            "no compact topological map graph",
            "no recurrent neural state beyond bounded hand-built temporal features",
            "no projectile or incoming-damage predictor",
            "reset seeds are labels until the server reports seed_applied=true",
        ],
        "next_feature_candidates": [
            "recent_damage_window_norm",
            "enemy_projectile_threat_norm",
            "topology_frontier_count_norm",
            "route_waypoint_repeat_count_norm",
        ],
        "upgrade_queue": [
            {
                "name": "topology_graph_features",
                "reason": "spawn-to-contact PPO needs map structure beyond a single local waypoint",
            },
            {
                "name": "snapshot_restore_context",
                "reason": "mid-trajectory curriculum requires progressed doors, enemies, and map mutations",
            },
            {
                "name": "combat_target_quality",
                "reason": "shootable yes/no is too coarse for learning aim margin and fire timing",
            },
            {
                "name": "recurrent_actor",
                "reason": "use only if bounded temporal features still cannot bridge contact state",
            },
        ],
        "gap_register": [
            {
                "name": "spawn_to_first_combat",
                "status": "open",
                "evidence": (
                    "spawn-only PPO buffers show route progress and positive route reward, "
                    "but zero shootable-target steps, damage, or kills"
                ),
                "needed_signal": (
                    "either true progressed-state snapshot restore or richer route/topology "
                    "observations that bridge from spawn to the first valid combat affordance"
                ),
            },
            {
                "name": "fresh_reset_trajectory_replay",
                "status": "invalid_for_progressed_map_context",
                "evidence": (
                    "trajectory-derived coordinates do not recreate opened doors, enemy "
                    "movement, or route mutations after a fresh ResetEpisode"
                ),
                "needed_signal": "save-state/Hellbox snapshot restore for progressed-map curriculum",
            },
        ],
    },
    "feature_descriptors": [
        *_feature_descriptors(TACTICAL_FEATURE_NAMES, source="protobuf_or_memory"),
        *_feature_descriptors(ACTION_HISTORY_FEATURE_NAMES, source="macro_step_history"),
        *_feature_descriptors(TEMPORAL_CONTEXT_FEATURE_NAMES, source="temporal_context"),
    ],
}

DECISION_CYCLE_SCHEMA = {
    "schema": "restfuldoom.decision_cycle.v1",
    "clock": "one learned decision per bounded macro-step, not one raw ticcmd per Doom tic",
    "runtime_trace": [
        {
            "phase": "observe",
            "code": "SkillController.observation(state)",
            "payload": "restfuldoom.observation.v1 feature vector",
            "persistent_writes": [],
        },
        {
            "phase": "mask",
            "code": "DoomAgentEnv.action_mask() -> SkillController.action_mask(state)",
            "payload": "restfuldoom.skill_action_mask.v1 boolean list",
            "persistent_writes": [],
        },
        {
            "phase": "decide",
            "code": "ActorCritic.sample/action() or heuristic_action_index()",
            "payload": "integer index into restfuldoom.skill_action.v1",
            "persistent_writes": [],
        },
        {
            "phase": "execute",
            "code": "SkillController.action_for(action_index, state)",
            "payload": "protobuf PlayerAction plus rollout_record.info.decision",
            "persistent_writes": [],
        },
        {
            "phase": "score",
            "code": "DoomAgentEnv.step(action_index)",
            "payload": "next observation, reward, done, transition metadata",
            "persistent_writes": ["rollout buffer row"],
        },
        {
            "phase": "macro_history",
            "code": "SkillController.record_action_history(...)",
            "payload": "previous-skill and route/contact outcome features for the next observation",
            "persistent_writes": [],
        },
    ],
    "controller_decision_interface": {
        "decision_layer_input": [
            "restfuldoom.observation.v1 feature vector",
            "restfuldoom.skill_action_mask.v1 feasible-action mask",
        ],
        "decision_layer_output": {
            "field": "rollout_record.action",
            "type": "integer index into restfuldoom.skill_action.v1",
        },
        "controller_input": [
            "selected skill index",
            "latest protobuf GameState",
            "AgentMemory query results already encoded into the observation",
            "SkillController episode-local action history",
        ],
        "controller_output": {
            "field": "rollout_record.info.decision",
            "meaning": "selected primitive plus one concrete protobuf PlayerAction sent to Doom",
        },
    },
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
        "rollout_record.info.route_outcome": "route waypoint attempt/reach/fail/progress metadata",
        "rollout_record.info.route_action_reward": "dense reward contribution from route-progress outcomes",
    },
    "interface_invariants": [
        "decision layer chooses exactly one integer skill index per macro-step",
        "fast controller converts that skill into one protobuf PlayerAction",
        "controller does not consume PPO gradients, logits, or optimizer state",
        "PPO does not emit raw ticcmd values directly",
        "the action mask used for sampling is stored and reused for PPO logprob recomputation",
        "persistent AgentMemory is not mutated during PPO collection macro-steps",
    ],
}

MEMORY_CONTRACT = {
    "schema": "restfuldoom.agent_memory_contract.v1",
    "memory_schema": "restfuldoom.agent_memory.v1",
    "default_path": "agent_memory/e1m1.json",
    "role": "inspectable world ledger and training checkpoint index, not a neural hidden state",
    "write_frequency": {
        "ppo_collection": "read-only persistent memory; only SkillController episode-local history mutates during macro-steps",
        "ppo_checkpoint": "writes checkpoint lineage, rollout summary, reward config, and eval history",
        "deterministic_episode": "writes step observations during the episode and appends a compact episode summary at finish",
        "export": "copies memory JSON plus referenced checkpoints and trajectories into a portable bundle",
    },
    "persisted_shape": {
        "cells": {
            "key": "coarse cell id such as '23:-36'",
            "fields": [
                "first_seen_tick",
                "visits",
                "enemy_sightings",
                "damage_events",
                "last_seen_tick",
                "last_seen_at",
            ],
        },
        "enemies": {
            "key": "Doom object id as a string",
            "fields": [
                "first_seen_tick",
                "last_seen_tick",
                "last_position",
                "last_distance",
                "last_health",
                "line_of_sight",
                "visible_count",
                "max_threat",
            ],
        },
        "training": [
            "episodes",
            "policy",
            "learned_policy",
            "ppo_policy",
            "ppo_best_checkpoint",
            "ppo_checkpoints",
            "lessons",
        ],
    },
    "access_rules": [
        {
            "rule": "ppo_inner_loop_read_only",
            "meaning": (
                "PPO rollout collection may read memory-derived features but must not "
                "write the persistent JSON ledger on every macro-step"
            ),
        },
        {
            "rule": "episode_local_controller_state",
            "meaning": (
                "SkillController may mutate short-lived action/contact/route history "
                "between reset() and terminal state"
            ),
        },
        {
            "rule": "checkpoint_boundary_writes",
            "meaning": (
                "PPO writes memory only after checkpoint, eval, or export boundaries "
                "so cloud resume remains auditable"
            ),
        },
    ],
    "query_update_lifecycle": [
        {
            "phase": "reset",
            "reads": ["ppo_policy", "ppo_checkpoints", "policy"],
            "writes": [],
            "purpose": "resume the selected controller or PPO checkpoint before collecting a new episode",
        },
        {
            "phase": "observe",
            "reads": ["remembered_enemies", "blocked_targets", "best_params"],
            "writes": [],
            "purpose": "derive observation features and action masks from live protobuf state plus persistent context",
        },
        {
            "phase": "act",
            "reads": ["remembered_enemies"],
            "writes": ["episode-local SkillController history"],
            "purpose": "turn the selected skill into one safe PlayerAction without mutating persistent memory",
        },
        {
            "phase": "learn",
            "reads": ["rollout buffer"],
            "writes": ["ppo_policy", "ppo_checkpoints", "ppo_eval_history"],
            "purpose": "persist checkpoint lineage, rollout summaries, and promotion-gate outcomes",
        },
        {
            "phase": "export",
            "reads": ["agent_memory/e1m1.json", "checkpoint files", "trajectory JSONL"],
            "writes": ["training job bundle"],
            "purpose": "move local learning state to Docker, Hellbox, or cloud without losing resume metadata",
        },
    ],
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
    "query_examples": [
        {
            "method": "AgentMemory.remembered_enemies",
            "input": {
                "x_units": 1024.0,
                "y_units": -2048.0,
                "tick": 12345,
                "max_age_tics": "BrainPolicyParams.enemy_memory_tics",
            },
            "output": [
                {
                    "id": 57,
                    "x": 1526.1,
                    "y": -2538.7,
                    "distance": 701.2,
                    "last_seen_tick": 12301,
                }
            ],
        }
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
            "updates": "ppo_policy, ppo_best_checkpoint, and ppo_checkpoints",
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
    "current_model": {
        "skill_definition": "code-defined option",
        "learned_object": "top-level selector over stable skill indexes",
        "not_learned_yet": "raw movement primitive or per-tic ticcmd policy",
        "runtime_dispatch": "SkillController._execute_skill(skill, features, stuck)",
    },
    "representation": {
        "current": "code-defined options implemented by SkillController._execute_skill",
        "learned_now": "PPO learns when to choose each option",
        "learned_later": "movement primitives can become learned/subpolicy-backed after the option set stabilizes",
    },
    "option_contract": {
        "skill_is": (
            "a stable option descriptor plus a SkillController dispatch branch "
            "that can emit one safe PlayerAction"
        ),
        "skill_is_not": [
            "an LLM-selected free-form function",
            "external configuration yet",
            "a learned per-tic movement primitive yet",
        ],
        "selector_learns": [
            "action probability",
            "value estimate",
            "advantage/reward signal for choosing the option",
        ],
        "controller_owns": [
            "movement amount and turn amount",
            "aiming tolerance and firing cadence",
            "door/switch use timing",
            "stuck recovery",
            "fallback primitive",
        ],
    },
    "skill_definition_contract": {
        "storage": "Python schema plus SkillController code branch; not external config yet",
        "stable_identifier": "integer action index and skill name in PPO_SKILL_ACTIONS",
        "learned_fields": ["selection probability", "value estimate", "advantage from reward"],
        "code_owned_fields": [
            "controller entrypoint",
            "low-level movement/aim/fire timing",
            "fallback primitive",
            "mask feasibility rule",
        ],
        "evolution_rule": (
            "a skill may become config-backed or subpolicy-backed only if it preserves "
            "the same index/name option contract for older checkpoints"
        ),
    },
    "mask_semantics": {
        "schema": "restfuldoom.skill_action_mask.v1",
        "source": "protobuf navigation/combat affordances plus AgentMemory",
        "meaning": "True entries are feasible skills for the current macro-step; PPO sampling and update logprobs use this mask.",
        "rules": [
            {
                "name": "shootable_followthrough",
                "meaning": (
                    "when the protobuf combat probe reports a shootable enemy and "
                    "the controller can fire, the normal combat mask exposes fire "
                    "and suppresses engage/use/route actions so the shot window is not lost"
                ),
            },
            {
                "name": "visible_contact",
                "meaning": (
                    "visible-but-not-shootable contact exposes engage, seek_enemy, "
                    "and contact use-line actions when those affordances exist"
                ),
            },
        ],
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
