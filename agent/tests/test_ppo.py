import asyncio
import json
from types import SimpleNamespace

import pytest

from restfuldoom_agent.curriculum import build_curriculum, stage_for_update
from restfuldoom_agent.env import EnvStep
from restfuldoom_agent.brain import AgentMemory, _memory_ppo_checkpoint_paths
from restfuldoom_agent.ppo_agent import (
    _annotate_buffer_curriculum,
    _checkpoint_selection_score,
    _checkpoint_resume_score,
    _checkpoint_resume_score_source,
    _policy_eval_selection_score,
    _record_ppo_checkpoint,
    _reset_start_from_trajectory,
    _resolve_resume_checkpoint,
    _should_replace_best_checkpoint,
    _summarize_buffer,
)
from restfuldoom_agent.learning_trace import LEARNING_TRACE_SCHEMA, build_learning_trace
from restfuldoom_agent.ppo import (
    PPOConfig,
    PPOTrainer,
    PromotionGate,
    EvaluationResult,
    RolloutBuffer,
    TORCH_AVAILABLE,
)
from restfuldoom_agent.ppo_eval import EpisodeEval, PolicyEval, decide_promotion
from restfuldoom_agent.schemas import (
    ACTION_SCHEMA,
    DECISION_CYCLE_SCHEMA,
    MEMORY_CONTRACT,
    OBSERVATION_SCHEMA,
    PRE_FRONTIER_TACTICAL_FEATURE_NAMES,
    PPO_SKILL_ACTIONS,
    pad_observation_features,
    map_expert_skill_to_ppo_action,
)
from restfuldoom_agent.skill_policy import FEATURE_NAMES


def test_rollout_buffer_saves_jsonl(tmp_path):
    buffer = RolloutBuffer()
    buffer.add(
        obs=[0.0, 1.0],
        action_mask=[False, True],
        action=1,
        reward=2.0,
        done=False,
        value=0.5,
        logprob=-0.7,
        info={"skill": "fire"},
    )

    path = buffer.save_jsonl(tmp_path / "buffer.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert rows[0]["schema"] == "restfuldoom.ppo_rollout.v1"
    assert rows[0]["count"] == 1
    assert rows[0]["decision_cycle_schema"]["schema"] == "restfuldoom.decision_cycle.v1"
    assert rows[0]["learning_trace_schema"] == "restfuldoom.learning_trace.v1"
    assert rows[0]["memory_contract"]["memory_schema"] == "restfuldoom.agent_memory.v1"
    assert rows[1]["action"] == 1
    assert rows[1]["action_mask"] == [False, True]
    assert rows[1]["info"]["skill"] == "fire"


def test_learning_trace_names_observation_mask_and_outcome():
    feature_names = OBSERVATION_SCHEMA["feature_names"]
    obs = [0.0 for _ in feature_names]
    obs[feature_names.index("health_norm")] = 0.75
    obs[feature_names.index("combat_has_target")] = 1.0
    obs[feature_names.index("route_waypoint_distance_norm")] = 0.25
    obs[feature_names.index("remembered_enemies_norm")] = 0.5
    obs[feature_names.index("visible_enemy_seen_recently")] = 1.0
    obs[feature_names.index("contact_use_line_active")] = 1.0
    obs[feature_names.index("contact_use_line_distance_norm")] = 0.4
    obs[feature_names.index("topology_frontier_active")] = 1.0
    obs[feature_names.index("topology_exhausted_open_ratio")] = 0.25
    action_mask = [False for _ in ACTION_SCHEMA["actions"]]
    action_mask[1] = True
    action_mask[3] = True

    trace = build_learning_trace(
        obs=obs,
        action_mask=action_mask,
        action=1,
        reward=3.25,
        done=False,
        info={
            "skill": "fire",
            "had_shootable_target": True,
            "action_reward": 0.5,
            "decision": {
                "skill": "ppo_fire",
                "enemy": {"id": 7, "distance": 128.125, "health": 20},
            },
            "transition": {"damage_delta": 10, "enemy_distance_delta": 4.25},
            "route_outcome": {
                "attempted": True,
                "reached": False,
                "progress_units": 32.0,
            },
        },
    )

    assert trace["schema"] == LEARNING_TRACE_SCHEMA
    assert trace["selected_action"] == {
        "index": 1,
        "skill": "fire",
        "available": True,
    }
    assert trace["available_skills"] == ["fire", "open_use_line"]
    assert trace["observation"]["groups"]["player"]["health_norm"] == 0.75
    assert trace["observation"]["groups"]["combat"]["combat_has_target"] == 1.0
    assert trace["observation"]["groups"]["route"]["route_waypoint_distance_norm"] == 0.25
    assert trace["observation"]["groups"]["memory"]["remembered_enemies_norm"] == 0.5
    assert trace["observation"]["groups"]["temporal"]["visible_enemy_seen_recently"] == 1.0
    assert trace["observation"]["groups"]["contact"]["contact_use_line_active"] == 1.0
    assert trace["observation"]["groups"]["contact"]["contact_use_line_distance_norm"] == 0.4
    assert trace["observation"]["groups"]["topology"]["topology_frontier_active"] == 1.0
    assert trace["observation"]["groups"]["topology"]["topology_exhausted_open_ratio"] == 0.25
    assert trace["controller"]["executed_skill"] == "fire"
    assert trace["controller"]["decision"]["enemy"]["distance"] == 128.125
    assert trace["outcome"]["reward"] == 3.25
    assert trace["outcome"]["action_reward"] == 0.5
    assert trace["outcome"]["transition"]["damage_delta"] == 10
    assert trace["outcome"]["route_outcome"]["progress_units"] == 32.0


def test_promotion_gate_blocks_regressions():
    baseline = EvaluationResult(
        policy_id="brain",
        level_completion_rate=1.0,
        mean_kills=6.0,
        survival_rate=1.0,
        mean_steps_to_exit=5000,
        mean_stuck_events=20,
        episode_count=3,
    )
    candidate = EvaluationResult(
        policy_id="ppo",
        level_completion_rate=0.5,
        mean_kills=5.0,
        survival_rate=1.0,
        mean_steps_to_exit=6000,
        mean_stuck_events=30,
        episode_count=3,
    )

    decision = PromotionGate().decide(candidate=candidate, baseline=baseline)

    assert not decision.promote
    assert "completion rate did not beat baseline" in decision.reasons
    assert "mean kills did not beat baseline" in decision.reasons


def test_ppo_eval_contract_serializes_and_blocks_weak_candidate():
    candidate = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:test",
            level_completion_rate=0.0,
            mean_kills=0.0,
            survival_rate=1.0,
            mean_steps_to_exit=256,
            mean_stuck_events=5,
            episode_count=1,
            mean_reward=1.25,
        ),
        episodes=[
            EpisodeEval(
                seed=7,
                total_reward=1.25,
                level_completed=False,
                death=False,
                max_kills=0,
                min_health=100,
                steps=256,
                steps_to_exit=256,
                stuck_events=5,
                done_reason="max_steps",
            )
        ],
    )
    baseline = PolicyEval(
        result=EvaluationResult(
            policy_id="heuristic",
            level_completion_rate=1.0,
            mean_kills=6.0,
            survival_rate=1.0,
            mean_steps_to_exit=200,
            mean_stuck_events=2,
            episode_count=1,
            mean_reward=2.0,
        ),
        episodes=[],
    )

    payload = candidate.to_dict()
    decision = decide_promotion(candidate=candidate, baseline=baseline)

    assert payload["result"]["policy_id"] == "ppo:test"
    assert payload["episodes"][0]["total_reward"] == 1.25
    assert not decision.promote
    assert "completion rate did not beat baseline" in decision.reasons
    assert "completion rate below promotion minimum" in decision.reasons
    assert "mean kills did not beat baseline" in decision.reasons
    assert "mean kills below promotion minimum" in decision.reasons
    assert "mean reward did not beat baseline" in decision.reasons


def test_expert_skill_labels_map_to_ppo_actions():
    assert map_expert_skill_to_ppo_action("fire_on_shootable_target") == 1
    assert map_expert_skill_to_ppo_action("seek_known_enemy") == 2
    assert map_expert_skill_to_ppo_action("push_exit_switch") == 7
    assert map_expert_skill_to_ppo_action("not_a_skill") is None


def test_training_schemas_describe_features_and_actions():
    assert OBSERVATION_SCHEMA["base_feature_names"] == FEATURE_NAMES
    assert len(OBSERVATION_SCHEMA["feature_names"]) > len(FEATURE_NAMES)
    assert len(OBSERVATION_SCHEMA["feature_descriptors"]) == len(
        OBSERVATION_SCHEMA["feature_names"]
    )
    assert ACTION_SCHEMA["actions"] == PPO_SKILL_ACTIONS
    assert [definition["skill"] for definition in ACTION_SCHEMA["definitions"]] == (
        PPO_SKILL_ACTIONS
    )
    assert all(
        definition["kind"] == "code_defined_option"
        and not definition["learned"]
        and definition["execution_owner"] == "BrainPolicy"
        for definition in ACTION_SCHEMA["definitions"]
    )
    assert ACTION_SCHEMA["mask_semantics"]["schema"] == "restfuldoom.skill_action_mask.v1"
    assert any(
        rule["name"] == "shootable_followthrough"
        for rule in ACTION_SCHEMA["mask_semantics"]["rules"]
    )
    assert any(
        rule["name"] == "recent_contact_route_backoff"
        for rule in ACTION_SCHEMA["mask_semantics"]["rules"]
    )
    assert ACTION_SCHEMA["representation"]["learned_now"] == (
        "PPO learns when to choose each option"
    )
    assert ACTION_SCHEMA["option_contract"]["skill_is"].startswith(
        "a stable option descriptor"
    )
    assert "action probability" in ACTION_SCHEMA["option_contract"]["selector_learns"]
    assert "aiming tolerance and firing cadence" in ACTION_SCHEMA["option_contract"][
        "controller_owns"
    ]
    assert ACTION_SCHEMA["skill_definition_contract"]["storage"].startswith("Python schema")
    assert ACTION_SCHEMA["current_model"]["learned_object"] == (
        "top-level selector over stable skill indexes"
    )
    assert DECISION_CYCLE_SCHEMA["schema"] == "restfuldoom.decision_cycle.v1"
    assert [phase["phase"] for phase in DECISION_CYCLE_SCHEMA["runtime_trace"]] == [
        "observe",
        "mask",
        "decide",
        "execute",
        "score",
        "macro_history",
    ]
    assert "controller_input" in DECISION_CYCLE_SCHEMA["controller_decision_interface"]
    assert "rollout_record.action_mask" in DECISION_CYCLE_SCHEMA["trace_fields"]
    assert "rollout_record.info.route_outcome" in DECISION_CYCLE_SCHEMA["trace_fields"]
    assert "PPO does not emit raw ticcmd values directly" in DECISION_CYCLE_SCHEMA[
        "interface_invariants"
    ]
    assert MEMORY_CONTRACT["memory_schema"] == "restfuldoom.agent_memory.v1"
    assert MEMORY_CONTRACT["write_frequency"]["ppo_collection"].startswith("read-only")
    assert "cells" in MEMORY_CONTRACT["persisted_shape"]
    assert "enemies" in MEMORY_CONTRACT["persisted_shape"]
    assert "ppo_best_checkpoint" in MEMORY_CONTRACT["persisted_shape"]["training"]
    assert any(
        rule["rule"] == "ppo_inner_loop_read_only"
        for rule in MEMORY_CONTRACT["access_rules"]
    )
    assert any(
        phase["phase"] == "learn" and "ppo_checkpoints" in phase["writes"]
        for phase in MEMORY_CONTRACT["query_update_lifecycle"]
    )
    assert any(path["method"].startswith("AgentMemory.remembered_enemies") for path in MEMORY_CONTRACT["query_paths"])
    assert MEMORY_CONTRACT["query_examples"][0]["method"] == "AgentMemory.remembered_enemies"
    assert any(group["name"] == "memory_queries" for group in OBSERVATION_SCHEMA["source_groups"])
    assert [phase["phase"] for phase in OBSERVATION_SCHEMA["protobuf_to_observation_pipeline"]] == [
        "protobuf",
        "tactical_features",
        "base_vector",
        "macro_history",
        "temporal_context",
        "contact_context",
        "topology_context",
    ]
    assert "sector_damaging" in OBSERVATION_SCHEMA["feature_names"]
    assert "route_waypoint_distance_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "prev_route_progress_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "failed_route_attempt_count_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "enemy_distance_delta_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "recent_route_failure_ratio" in OBSERVATION_SCHEMA["feature_names"]
    assert "topology_frontier_count_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "contact_use_line_active" in OBSERVATION_SCHEMA["feature_names"]
    assert "contact_use_line_followthrough_active" in OBSERVATION_SCHEMA["feature_names"]
    assert "topology_current_cell_visits_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "topology_frontier_angle_cos" in OBSERVATION_SCHEMA["feature_names"]
    assert "temporal_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "contact_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "topology_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "no compact topological map graph" in OBSERVATION_SCHEMA["learning_readiness"]["known_gaps"]
    assert "reset metadata must say whether a state is fresh, warmed up, or snapshot-restored" in (
        OBSERVATION_SCHEMA["learning_readiness"]["rich_observation_definition"]
    )
    assert any(
        item["name"] == "snapshot_restore_context"
        for item in OBSERVATION_SCHEMA["learning_readiness"]["upgrade_queue"]
    )
    assert any(
        gap["name"] == "spawn_to_first_combat"
        for gap in OBSERVATION_SCHEMA["learning_readiness"]["gap_register"]
    )


def test_pad_observation_features_adds_neutral_action_history():
    tactical = [0.0 for _ in FEATURE_NAMES]

    padded = pad_observation_features(tactical)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[: len(FEATURE_NAMES)] == tactical
    assert all(value == 0.0 for value in padded[len(FEATURE_NAMES) :])


def test_pad_observation_features_adds_neutral_frontier_to_old_tactical_rows():
    old_tactical = [0.125 for _ in PRE_FRONTIER_TACTICAL_FEATURE_NAMES]
    frontier_index = OBSERVATION_SCHEMA["feature_names"].index(
        "topology_frontier_count_norm"
    )

    padded = pad_observation_features(old_tactical)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[frontier_index] == 0.0
    assert padded[:frontier_index] == old_tactical[:frontier_index]
    assert padded[frontier_index + 1 : len(FEATURE_NAMES)] == old_tactical[
        frontier_index:
    ]


def test_pad_observation_features_adds_neutral_frontier_to_old_ppo_rows():
    old_row = [0.25 for _ in PRE_FRONTIER_TACTICAL_FEATURE_NAMES]
    old_row.extend([0.5 for _ in OBSERVATION_SCHEMA["action_history_feature_names"]])
    old_row.extend([0.75 for _ in OBSERVATION_SCHEMA["temporal_context_feature_names"]])
    frontier_index = OBSERVATION_SCHEMA["feature_names"].index(
        "topology_frontier_count_norm"
    )
    contact_start = len(OBSERVATION_SCHEMA["feature_names"]) - len(
        OBSERVATION_SCHEMA["contact_context_feature_names"]
    ) - len(
        OBSERVATION_SCHEMA["topology_context_feature_names"]
    )

    padded = pad_observation_features(old_row)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[frontier_index] == 0.0
    assert padded[:frontier_index] == old_row[:frontier_index]
    assert padded[frontier_index + 1 : contact_start] == old_row[frontier_index:]
    assert all(value == 0.0 for value in padded[contact_start:])


def test_pad_observation_features_adds_neutral_contact_context_to_legacy_ppo_rows():
    old_row = [0.25 for _ in OBSERVATION_SCHEMA["base_feature_names"]]
    old_row.extend([0.5 for _ in OBSERVATION_SCHEMA["action_history_feature_names"]])
    old_row.extend([0.75 for _ in OBSERVATION_SCHEMA["temporal_context_feature_names"]])

    padded = pad_observation_features(old_row)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[: len(old_row)] == old_row
    assert all(value == 0.0 for value in padded[len(old_row) :])


def test_pad_observation_features_adds_neutral_topology_context_to_contact_rows():
    old_row = [0.25 for _ in OBSERVATION_SCHEMA["base_feature_names"]]
    old_row.extend([0.5 for _ in OBSERVATION_SCHEMA["action_history_feature_names"]])
    old_row.extend([0.75 for _ in OBSERVATION_SCHEMA["temporal_context_feature_names"]])
    old_row.extend([1.0 for _ in OBSERVATION_SCHEMA["contact_context_feature_names"]])

    padded = pad_observation_features(old_row)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[: len(old_row)] == old_row
    assert all(value == 0.0 for value in padded[len(old_row) :])


def test_pad_observation_features_adds_neutral_temporal_context_to_legacy_rows():
    legacy = [0.0 for _ in OBSERVATION_SCHEMA["base_feature_names"]]
    legacy.extend([0.25 for _ in OBSERVATION_SCHEMA["action_history_feature_names"]])

    padded = pad_observation_features(legacy)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[: len(legacy)] == legacy
    assert all(value == 0.0 for value in padded[len(legacy) :])


def test_rollout_summary_deduplicates_reset_warmup_metadata():
    buffer = SimpleNamespace(
        records=[
            SimpleNamespace(
                reward=0.0,
                done=False,
                info={
                    "skill": "seek_enemy",
                    "reset_warmup": {
                        "enabled": True,
                        "episode_index": 1,
                        "steps": 12,
                        "tics": 30,
                        "stop_reason": "visible",
                    },
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
            SimpleNamespace(
                reward=0.0,
                done=False,
                info={
                    "skill": "engage",
                    "reset_warmup": {
                        "enabled": True,
                        "episode_index": 1,
                        "steps": 12,
                        "tics": 30,
                        "stop_reason": "visible",
                    },
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
        ]
    )

    summary = _summarize_buffer(buffer)

    assert summary["reset_warmup_steps"] == 12
    assert summary["reset_warmup_tics"] == 30
    assert summary["reset_warmup_stop_reasons"] == {"visible": 1}


def test_reset_start_from_trajectory_row(tmp_path):
    trajectory = tmp_path / "combat.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "state": {
                    "position_fp": [123, -456, 0],
                    "health": 95,
                    "armor": 7,
                    "ammo_bullets": 37,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    start = _reset_start_from_trajectory(trajectory, index=0)

    assert start == {
        "x_fp": 123,
        "y_fp": -456,
        "health": 95,
        "armor": 7,
        "ammo_bullets": 37,
    }


def test_named_curriculum_selects_e1m1_stages():
    curriculum = build_curriculum(
        name="e1m1-spawn-to-combat",
        manual_reset_start={},
        mode="round_robin",
        start_index=0,
        seed=7,
    )

    first = stage_for_update(curriculum, 0)
    second = stage_for_update(curriculum, 1)
    spawn = stage_for_update(curriculum, 3)

    assert curriculum["schema"] == "restfuldoom.ppo_curriculum.v1"
    assert first["name"] == "combat_start"
    assert first["reset_start"]["face_nearest_enemy"] is True
    assert first["validated"] is True
    assert first["evidence"]["shootable_target_on_reset"] is True
    assert second["name"] == "combat_wide_left"
    assert second["requires_progressed_state"] is False
    assert spawn["name"] == "fresh_spawn"
    assert spawn["reset_start"] == {}
    assert spawn["evidence"]["shootable_target_on_reset"] is False


def test_named_curriculum_selects_contact_to_combat_stages():
    curriculum = build_curriculum(
        name="e1m1-contact-to-combat",
        manual_reset_start={},
        mode="round_robin",
        start_index=0,
        seed=7,
    )

    visible = stage_for_update(curriculum, 0)
    combat = stage_for_update(curriculum, 3)

    assert curriculum["schema"] == "restfuldoom.ppo_curriculum.v1"
    assert visible["name"] == "visible_contact_fast"
    assert visible["evidence"]["visible_enemy_on_reset"] is True
    assert visible["evidence"]["shootable_target_on_reset"] is False
    assert visible["reset_start"]["face_nearest_enemy"] is True
    assert combat["name"] == "combat_start"
    assert combat["evidence"]["shootable_target_on_reset"] is True


def test_curriculum_fixed_mode_repeats_start_index_stage():
    curriculum = build_curriculum(
        name="e1m1-contact-to-combat",
        manual_reset_start={},
        mode="fixed",
        start_index=1,
        seed=7,
    )

    first = stage_for_update(curriculum, 0)
    later = stage_for_update(curriculum, 8)

    assert first["name"] == "visible_contact_route"
    assert later["name"] == "visible_contact_route"
    assert first["selected_index"] == 1
    assert later["selected_index"] == 1


def test_curriculum_rejects_manual_start_mix():
    with pytest.raises(ValueError, match="cannot be combined"):
        build_curriculum(
            name="e1m1-spawn-to-combat",
            manual_reset_start={"x_fp": 1},
            mode="round_robin",
            start_index=0,
            seed=7,
        )


def test_rollout_summary_counts_curriculum_stages():
    buffer = RolloutBuffer()
    buffer.add(
        obs=[0.0, 1.0],
        action_mask=[True, False],
        action=0,
        reward=1.0,
        done=False,
        value=0.0,
        logprob=0.0,
        info={
            "skill": "engage",
            "transition": {},
            "state": {"health": 100, "kills": 0},
        },
    )
    curriculum = build_curriculum(
        name="e1m1-spawn-to-combat",
        manual_reset_start={},
        mode="round_robin",
        start_index=0,
        seed=7,
    )
    stage = stage_for_update(curriculum, 0)

    _annotate_buffer_curriculum(buffer, curriculum, stage)
    summary = _summarize_buffer(buffer)

    assert buffer.records[0].info["curriculum"]["name"] == "e1m1-spawn-to-combat"
    assert buffer.records[0].info["curriculum_stage"]["name"] == "combat_start"
    assert summary["curriculum_stage_counts"] == {"combat_start": 1}


def test_rollout_summary_preserves_mixed_curriculum_stages_in_jsonl(tmp_path):
    buffer = RolloutBuffer()
    for stage_name in ["combat_start", "fresh_spawn", "combat_start"]:
        buffer.add(
            obs=[0.0, 1.0],
            action_mask=[True, False],
            action=0,
            reward=1.0,
            done=False,
            value=0.0,
            logprob=0.0,
            info={
                "skill": "engage",
                "transition": {},
                "state": {"health": 100, "kills": 0},
                "curriculum": {
                    "schema": "restfuldoom.ppo_curriculum.v1",
                    "name": "e1m1-contact-to-combat",
                },
                "curriculum_stage": {"name": stage_name},
            },
        )

    summary = _summarize_buffer(buffer)
    path = buffer.save_jsonl(tmp_path / "mixed.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert summary["curriculum_stage_counts"] == {
        "combat_start": 2,
        "fresh_spawn": 1,
    }
    assert [
        row["info"]["curriculum_stage"]["name"]
        for row in rows[1:]
    ] == ["combat_start", "fresh_spawn", "combat_start"]


def test_rollout_summary_counts_route_outcomes():
    buffer = RolloutBuffer()
    for index, route_outcome in enumerate(
        [
            {"attempted": True, "reached": False, "failed": False, "progress_units": 64.0},
            {"attempted": True, "reached": False, "failed": True, "progress_units": -12.5},
            {"attempted": True, "reached": True, "failed": False, "progress_units": 32.0},
        ]
    ):
        buffer.add(
            obs=[0.0, 1.0],
            action_mask=[True, False],
            action=0,
            reward=1.0,
            done=False,
            value=0.0,
            logprob=0.0,
            info={
                "skill": "route_progression",
                "route_outcome": route_outcome,
                "route_action_reward": 0.25 * (index + 1),
                "transition": {},
                "state": {"health": 100, "kills": 0},
            },
        )

    summary = _summarize_buffer(buffer)

    assert summary["route_attempt_steps"] == 3
    assert summary["route_reached_steps"] == 1
    assert summary["route_failed_steps"] == 1
    assert summary["route_progress_units"] == 83.5
    assert summary["route_action_reward"] == 1.5


def test_rollout_summary_counts_contact_context_from_learning_trace():
    buffer = RolloutBuffer()
    for contact in [
        {
            "recent_contact_active": 1.0,
            "contact_use_line_active": 1.0,
            "contact_use_line_distance_norm": 0.5,
            "contact_use_line_close": 0.0,
            "contact_use_line_followthrough_active": 1.0,
            "contact_use_line_age_norm": 0.25,
        },
        {
            "recent_contact_active": 1.0,
            "contact_use_line_active": 1.0,
            "contact_use_line_distance_norm": 0.25,
            "contact_use_line_close": 1.0,
            "contact_use_line_followthrough_active": 0.0,
            "contact_use_line_age_norm": 0.5,
        },
        {
            "recent_contact_active": 0.0,
            "contact_use_line_active": 0.0,
            "contact_use_line_distance_norm": 0.0,
            "contact_use_line_close": 0.0,
            "contact_use_line_followthrough_active": 0.0,
            "contact_use_line_age_norm": 0.0,
        },
    ]:
        buffer.add(
            obs=[0.0, 1.0],
            action_mask=[True, False],
            action=0,
            reward=0.0,
            done=False,
            value=0.0,
            logprob=0.0,
            info={
                "skill": "engage",
                "learning_trace": {
                    "observation": {
                        "groups": {
                            "contact": contact,
                        },
                    },
                },
                "transition": {},
                "state": {"health": 100, "kills": 0},
            },
        )

    summary = _summarize_buffer(buffer)

    assert summary["contact_context_active_steps"] == 2
    assert summary["contact_use_line_active_steps"] == 2
    assert summary["contact_use_line_close_steps"] == 1
    assert summary["contact_use_line_followthrough_steps"] == 1
    assert summary["mean_contact_use_line_distance_norm"] == 0.375
    assert summary["mean_contact_use_line_age_norm"] == 0.375


def test_rollout_summary_counts_topology_context_from_learning_trace():
    buffer = RolloutBuffer()
    for topology in [
        {
            "topology_current_cell_visits_norm": 0.25,
            "topology_open_cell_min_visit_norm": 0.0,
            "topology_open_cell_mean_visit_norm": 0.125,
            "topology_frontier_active": 1.0,
            "topology_exhausted_open_ratio": 0.25,
        },
        {
            "topology_current_cell_visits_norm": 0.5,
            "topology_open_cell_min_visit_norm": 0.375,
            "topology_open_cell_mean_visit_norm": 0.625,
            "topology_frontier_active": 0.0,
            "topology_exhausted_open_ratio": 1.0,
        },
    ]:
        buffer.add(
            obs=[0.0, 1.0],
            action_mask=[True, False],
            action=0,
            reward=0.0,
            done=False,
            value=0.0,
            logprob=0.0,
            info={
                "skill": "route_progression",
                "learning_trace": {
                    "observation": {
                        "groups": {
                            "topology": topology,
                        },
                    },
                },
                "transition": {},
                "state": {"health": 100, "kills": 0},
            },
        )

    summary = _summarize_buffer(buffer)

    assert summary["topology_frontier_active_steps"] == 1
    assert summary["mean_topology_current_cell_visits_norm"] == 0.375
    assert summary["mean_topology_open_cell_min_visit_norm"] == 0.1875
    assert summary["mean_topology_exhausted_open_ratio"] == 0.625


def test_rollout_summary_counts_first_contact_events():
    buffer = RolloutBuffer()
    buffer.add(
        obs=[0.0, 1.0],
        action_mask=[True, False],
        action=0,
        reward=3.0,
        done=True,
        value=0.0,
        logprob=0.0,
        info={
            "skill": "route_progression",
            "had_visible_enemy": True,
            "had_shootable_target": False,
            "first_visible_contact": True,
            "first_shootable_contact": False,
            "contact_reward": 3.0,
            "visible_contact_distance_delta": 16.0,
            "visible_contact_progress_reward": 0.16,
            "transition": {},
            "state": {"health": 100, "kills": 0},
        },
    )
    buffer.add(
        obs=[0.0, 1.0],
        action_mask=[True, False],
        action=0,
        reward=5.0,
        done=True,
        value=0.0,
        logprob=0.0,
        info={
            "skill": "engage",
            "had_visible_enemy": True,
            "had_shootable_target": True,
            "first_visible_contact": False,
            "first_shootable_contact": True,
            "contact_reward": 5.0,
            "visible_contact_distance_delta": 8.0,
            "visible_contact_progress_reward": 0.08,
            "transition": {},
            "state": {"health": 100, "kills": 0},
        },
    )
    buffer.add(
        obs=[0.0, 1.0],
        action_mask=[False, True],
        action=1,
        reward=1.0,
        done=False,
        value=0.0,
        logprob=0.0,
        info={
            "skill": "fire",
            "had_visible_enemy": True,
            "had_shootable_target": True,
            "first_visible_contact": False,
            "first_shootable_contact": False,
            "contact_reward": 0.0,
            "visible_contact_distance_delta": 0.0,
            "visible_contact_progress_reward": 0.0,
            "transition": {},
            "state": {"health": 100, "kills": 0},
        },
    )

    summary = _summarize_buffer(buffer)

    assert summary["visible_enemy_steps"] == 3
    assert summary["shootable_target_steps"] == 2
    assert summary["fire_on_shootable_steps"] == 1
    assert summary["missed_shootable_fire_steps"] == 1
    assert summary["checkpoint_selection_score"] > summary["total_reward"]
    assert summary["first_visible_contacts"] == 1
    assert summary["first_shootable_contacts"] == 1
    assert summary["contact_reward"] == 8.0
    assert summary["visible_contact_distance_delta"] == 24.0
    assert summary["visible_contact_progress_reward"] == 0.24


def test_checkpoint_selection_score_prefers_damage_contact_and_fire():
    weak = {
        "total_reward": 3.0,
        "max_kills": 0,
        "damage_delta": 0,
        "first_shootable_contacts": 0,
        "fire_on_shootable_steps": 0,
        "missed_shootable_fire_steps": 0,
        "route_failed_steps": 1,
    }
    useful = {
        "total_reward": 2.0,
        "max_kills": 0,
        "damage_delta": 10,
        "first_shootable_contacts": 1,
        "fire_on_shootable_steps": 4,
        "missed_shootable_fire_steps": 0,
        "route_failed_steps": 2,
    }

    assert _checkpoint_selection_score(useful) > _checkpoint_selection_score(weak)


def test_policy_eval_selection_score_rewards_cross_stage_competence():
    weak = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:weak",
            level_completion_rate=0.0,
            mean_kills=0.0,
            survival_rate=1.0,
            mean_steps_to_exit=256,
            mean_stuck_events=4.0,
            episode_count=1,
            mean_reward=20.0,
        ),
        episodes=[],
    )
    useful = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:useful",
            level_completion_rate=0.0,
            mean_kills=1.0,
            survival_rate=1.0,
            mean_steps_to_exit=256,
            mean_stuck_events=1.0,
            episode_count=1,
            mean_reward=15.0,
        ),
        episodes=[],
    )

    assert _policy_eval_selection_score(useful) > _policy_eval_selection_score(weak)


def test_checkpoint_resume_score_prefers_curriculum_eval_when_present():
    rollout_summary = {"checkpoint_selection_score": 500.0}
    checkpoint_eval = {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "selection_score": 42.5,
    }

    assert _checkpoint_resume_score(rollout_summary, checkpoint_eval) == 42.5
    assert _checkpoint_resume_score_source(checkpoint_eval) == "checkpoint_curriculum_eval"
    assert _checkpoint_resume_score(rollout_summary, None) == 500.0
    assert _checkpoint_resume_score_source(None) == "rollout_summary"


def test_curriculum_eval_best_replaces_legacy_rollout_best():
    legacy_best = {
        "checkpoint_selection_score": 890.0,
        "checkpoint_path": "old-rollout.pt",
    }
    eval_best = {
        "checkpoint_selection_score": 80.0,
        "checkpoint_selection_source": "checkpoint_curriculum_eval",
        "checkpoint_path": "old-eval.pt",
    }

    assert _should_replace_best_checkpoint(
        legacy_best,
        score=24.0,
        score_source="checkpoint_curriculum_eval",
    )
    assert not _should_replace_best_checkpoint(
        eval_best,
        score=500.0,
        score_source="rollout_summary",
    )
    assert _should_replace_best_checkpoint(
        eval_best,
        score=90.0,
        score_source="checkpoint_curriculum_eval",
    )
    assert not _should_replace_best_checkpoint(
        eval_best,
        score=70.0,
        score_source="checkpoint_curriculum_eval",
    )


def test_record_ppo_checkpoint_preserves_best_resume_candidate(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    first_summary = {
        "checkpoint_selection_score": 50.0,
        "total_reward": 5.0,
        "damage_delta": 10,
    }
    second_summary = {
        "checkpoint_selection_score": -1.0,
        "total_reward": -1.0,
        "damage_delta": 0,
    }

    _record_ppo_checkpoint(
        memory,
        tmp_path / "good.pt",
        goal_preset="combat",
        reward_config={"goal_preset": "combat"},
        metrics={"policy_loss": 0.1},
        rollout_summary=first_summary,
        update_index=0,
        buffer_path=tmp_path / "good.jsonl",
        curriculum_stage={"name": "visible_contact_fast"},
    )
    _record_ppo_checkpoint(
        memory,
        tmp_path / "bad.pt",
        goal_preset="combat",
        reward_config={"goal_preset": "combat"},
        metrics={"policy_loss": 0.2},
        rollout_summary=second_summary,
        update_index=1,
        buffer_path=tmp_path / "bad.jsonl",
        curriculum_stage={"name": "visible_contact_fast"},
    )

    assert memory.data["ppo_policy"]["checkpoint_path"].endswith("bad.pt")
    assert memory.data["ppo_best_checkpoint"]["checkpoint_path"].endswith("good.pt")
    assert memory.data["ppo_best_checkpoint"]["checkpoint_selection_score"] == 50.0


def test_record_ppo_checkpoint_uses_curriculum_eval_for_best_candidate(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    high_rollout_low_eval = {
        "checkpoint_selection_score": 500.0,
        "total_reward": 200.0,
        "damage_delta": 30,
    }
    lower_rollout_higher_eval = {
        "checkpoint_selection_score": 100.0,
        "total_reward": 80.0,
        "damage_delta": 10,
    }

    _record_ppo_checkpoint(
        memory,
        tmp_path / "overfit.pt",
        goal_preset="combat",
        reward_config={"goal_preset": "combat"},
        metrics={"policy_loss": 0.1},
        rollout_summary=high_rollout_low_eval,
        update_index=0,
        buffer_path=tmp_path / "overfit.jsonl",
        curriculum_stage={"name": "visible_contact_seek"},
        checkpoint_eval={
            "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
            "selection_score": 10.0,
            "stages": [],
        },
    )
    _record_ppo_checkpoint(
        memory,
        tmp_path / "general.pt",
        goal_preset="combat",
        reward_config={"goal_preset": "combat"},
        metrics={"policy_loss": 0.2},
        rollout_summary=lower_rollout_higher_eval,
        update_index=1,
        buffer_path=tmp_path / "general.jsonl",
        curriculum_stage={"name": "visible_contact_route"},
        checkpoint_eval={
            "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
            "selection_score": 80.0,
            "stages": [],
        },
    )

    best = memory.data["ppo_best_checkpoint"]
    assert memory.data["ppo_policy"]["checkpoint_path"].endswith("general.pt")
    assert best["checkpoint_path"].endswith("general.pt")
    assert best["checkpoint_selection_score"] == 80.0
    assert best["checkpoint_selection_source"] == "checkpoint_curriculum_eval"
    assert best["checkpoint_eval"]["schema"] == "restfuldoom.ppo_checkpoint_curriculum_eval.v1"


def test_ppo_export_paths_include_best_checkpoint(tmp_path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    lineage = tmp_path / "lineage.pt"
    latest.write_text("latest", encoding="utf-8")
    best.write_text("best", encoding="utf-8")
    lineage.write_text("lineage", encoding="utf-8")
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["ppo_policy"] = {"checkpoint_path": str(latest)}
    memory.data["ppo_best_checkpoint"] = {"checkpoint_path": str(best)}
    memory.data["ppo_checkpoints"] = [{"checkpoint_path": str(lineage)}]

    paths = _memory_ppo_checkpoint_paths(memory)

    assert paths == [latest, best, lineage]


def test_resolve_resume_best_checkpoint_from_memory(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["ppo_best_checkpoint"] = {"checkpoint_path": str(checkpoint)}
    args = SimpleNamespace(
        resume_checkpoint=None,
        resume_best_checkpoint=True,
    )

    assert _resolve_resume_checkpoint(args, memory) == checkpoint


def test_resolve_resume_checkpoint_rejects_conflicts(tmp_path):
    checkpoint = tmp_path / "explicit.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    args = SimpleNamespace(
        resume_checkpoint=checkpoint,
        resume_best_checkpoint=True,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_resume_checkpoint(args, AgentMemory.load(tmp_path / "memory.json"))


def test_resolve_resume_best_checkpoint_requires_existing_file(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    memory.data["ppo_best_checkpoint"] = {
        "checkpoint_path": str(tmp_path / "missing.pt")
    }
    args = SimpleNamespace(
        resume_checkpoint=None,
        resume_best_checkpoint=True,
    )

    with pytest.raises(ValueError, match="does not exist"):
        _resolve_resume_checkpoint(args, memory)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_actor_respects_action_mask():
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=3,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8, seed=17),
    )

    actions = {
        trainer.model.act(
            [0.0, 1.0],
            action_mask=[False, True, False],
        )[0]
        for _ in range(12)
    }
    deterministic, _logprob, _value = trainer.model.act(
        [0.0, 1.0],
        deterministic=True,
        action_mask=[False, False, True],
    )

    assert actions == {1}
    assert deterministic == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_collect_rollout_calls_before_reset_for_episode_stages():
    class FakeEnv:
        def __init__(self) -> None:
            self.stage = 0
            self.steps_in_episode = 0
            self.reset_seeds: list[int | None] = []

        async def reset(self, *, seed=None):
            self.reset_seeds.append(seed)
            self.steps_in_episode = 0
            return [float(self.stage), 0.0]

        def action_mask(self):
            return [True, False]

        async def step(self, _action):
            self.steps_in_episode += 1
            done = self.steps_in_episode >= 2
            return EnvStep(
                observation=[float(self.stage), float(self.steps_in_episode)],
                reward=1.0,
                done=done,
                info={
                    "skill": "engage",
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                    "curriculum_stage": {"name": f"stage_{self.stage}"},
                },
            )

    async def collect():
        env = FakeEnv()
        reset_calls: list[int] = []
        trainer = PPOTrainer(
            obs_dim=2,
            action_dim=2,
            config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=5, seed=17),
        )

        def before_reset(reset_index: int) -> None:
            reset_calls.append(reset_index)
            env.stage = reset_index

        buffer = await trainer.collect_rollout(
            env,  # type: ignore[arg-type]
            steps=5,
            seed=99,
            before_reset=before_reset,
        )
        return env, reset_calls, buffer

    env, reset_calls, buffer = asyncio.run(collect())

    assert reset_calls == [0, 1, 2]
    assert env.reset_seeds == [99, 101, 103]
    assert [record.info["curriculum_stage"]["name"] for record in buffer.records] == [
        "stage_0",
        "stage_0",
        "stage_1",
        "stage_1",
        "stage_2",
    ]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_update_and_checkpoint_roundtrip(tmp_path):
    buffer = RolloutBuffer()
    for index in range(8):
        buffer.add(
            obs=[float(index % 2), 1.0],
            action_mask=[index % 2 == 0, index % 2 == 1],
            action=index % 2,
            reward=1.0 if index % 2 else 0.0,
            done=index == 7,
            value=0.1,
            logprob=-0.69,
        )
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8),
    )

    metrics = trainer.update(buffer)
    checkpoint = trainer.save_checkpoint(tmp_path / "ppo.pt")
    loaded = PPOTrainer.load_checkpoint(checkpoint)

    assert checkpoint.exists()
    assert metrics["value_loss"] >= 0
    assert loaded.update_index == trainer.update_index


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_checkpoint_expands_appended_observation_features(tmp_path):
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8),
    )
    checkpoint = trainer.save_checkpoint(tmp_path / "old.pt")

    loaded = PPOTrainer.load_checkpoint(
        checkpoint,
        target_obs_dim=4,
        target_action_dim=2,
    )

    action, _logprob, _value = loaded.model.act([0.0, 1.0, 0.5, -0.5])
    assert action in {0, 1}
    assert loaded.obs_dim == 4
    assert loaded.resume_migration is not None
    assert loaded.resume_migration["from_obs_dim"] == 2
    assert not loaded.resume_migration["optimizer_state_loaded"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_actor_behavior_clone_warm_start():
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8, seed=11),
    )
    samples = [([1.0, 0.0], 0) for _ in range(8)] + [([0.0, 1.0], 1) for _ in range(8)]

    metrics = trainer.pretrain_actor(samples, epochs=8, minibatch_size=4, learning_rate=0.05)

    assert metrics["bc_samples"] == 16
    assert metrics["bc_accuracy"] >= 0.75
