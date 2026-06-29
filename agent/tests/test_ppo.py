import asyncio
import json
from types import SimpleNamespace

import pytest

from restfuldoom_agent.curriculum import build_curriculum, stage_for_update
from restfuldoom_agent.env import EnvStep
from restfuldoom_agent.brain import AgentMemory, _memory_ppo_checkpoint_paths
import restfuldoom_agent.ppo_agent as ppo_agent_module
from restfuldoom_agent.ppo_agent import (
    _annotate_buffer_curriculum,
    _append_true_spawn_stage,
    _load_behavior_clone_samples,
    _checkpoint_selection_score,
    _checkpoint_resume_score,
    _checkpoint_resume_score_source,
    _checkpoint_eval_training_guard,
    _checkpoint_eval_stage_order,
    _checkpoint_eval_trace_path,
    _evaluate_checkpoint_curriculum,
    _env_config_for_stage,
    _redacted_restore_argv,
    _render_snapshot_restore_command,
    _policy_eval_selection_components,
    _policy_eval_selection_score,
    _record_ppo_checkpoint,
    _reset_start_from_trajectory,
    _restore_snapshot_for_stage,
    _resolve_resume_checkpoint,
    _should_replace_best_checkpoint,
    _summarize_buffer,
    evaluate,
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
from restfuldoom_agent.ppo_eval import (
    EpisodeEval,
    PolicyEval,
    _aggregate,
    evaluate_checkpoint,
    _open_trace,
    _reset_context_snapshot_verification_failed,
    _write_trace_step,
    decide_promotion,
)
from restfuldoom_agent.snapshot_curriculum import load_snapshot_curriculum
from restfuldoom_agent.schemas import (
    ACTION_SCHEMA,
    DECISION_CYCLE_SCHEMA,
    MEMORY_CONTRACT,
    OBSERVATION_SCHEMA,
    LEGACY_ACTION_HISTORY_FEATURE_NAMES_V1,
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


def test_ppo_eval_trace_writer_serializes_steps(tmp_path):
    path = tmp_path / "eval-trace.jsonl"
    handle = _open_trace(path, "ppo:checkpoint.pt")
    try:
        _write_trace_step(
            handle,
            policy_id="ppo:checkpoint.pt",
            episode_index=0,
            seed=7,
            step_index=1,
            action_index=4,
            action_mask=[False, False, False, False, True],
            reward=1.25,
            done=False,
            observation=[0.1, 0.2],
            info={
                "skill": "route_progression",
                "decision": {"skill": "route_to_progression_line"},
                "action": {
                    "raw": {"forward_move": 28, "side_move": 0, "angle_turn": -256},
                    "duration_tics": 4,
                },
                "state": {"kills": 2, "items": 1},
                "transition": {"kill_delta": 1},
                "route_outcome": {"attempted": True},
                "had_visible_enemy": False,
                "had_shootable_target": False,
            },
        )
    finally:
        handle.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["schema"] == "restfuldoom.ppo_eval_trace.v1"
    assert rows[1]["policy_id"] == "ppo:checkpoint.pt"
    assert rows[1]["skill"] == "route_progression"
    assert rows[1]["decision"]["skill"] == "route_to_progression_line"
    assert rows[1]["action"] == {
        "raw": {"forward_move": 28, "side_move": 0, "angle_turn": -256},
        "duration_tics": 4,
    }
    assert rows[1]["action_mask"] == [False, False, False, False, True]


def test_checkpoint_eval_trace_path_uses_stage_suffix_for_multi_stage(tmp_path):
    base = tmp_path / "eval-trace.jsonl"

    assert (
        _checkpoint_eval_trace_path(
            base,
            stage_name="first visible/stage",
            stage_index=0,
            stage_count=3,
            update_index=2,
        )
        == tmp_path / "eval-trace-update0002-stage00-first-visible-stage.jsonl"
    )
    assert (
        _checkpoint_eval_trace_path(
            base,
            stage_name="only-stage",
            stage_index=0,
            stage_count=1,
            update_index=2,
        )
        == tmp_path / "eval-trace-update0002-stage00-only-stage.jsonl"
    )
    assert (
        _checkpoint_eval_trace_path(
            base,
            stage_name="only-stage",
            stage_index=0,
            stage_count=1,
            update_index=0,
        )
        == base
    )
    assert (
        _checkpoint_eval_trace_path(
            base,
            stage_name="a/b",
            stage_index=1,
            stage_count=2,
            update_index=2,
        )
        != _checkpoint_eval_trace_path(
            base,
            stage_name="a:b",
            stage_index=2,
            stage_count=2,
            update_index=2,
        )
    )


def test_eval_candidate_only_skips_baseline_and_history(monkeypatch, tmp_path):
    async def fake_evaluate_checkpoint(checkpoint_path, env_config, **kwargs):
        assert checkpoint_path == str(tmp_path / "candidate.pt")
        assert kwargs["episodes"] == 5
        assert kwargs["max_steps"] == 6000
        assert kwargs["seed"] == 7
        return _aggregate(
            "ppo:candidate",
            [
                EpisodeEval(
                    seed=7,
                    total_reward=100.0,
                    level_completed=True,
                    death=False,
                    max_kills=5,
                    min_health=40,
                    steps=1800,
                    steps_to_exit=1800,
                    stuck_events=0,
                    done_reason="level_complete",
                    start_kills=0,
                    kill_delta=5,
                    max_kill_gain=5,
                    start_episode=1,
                    start_map=1,
                    end_episode=1,
                    end_map=2,
                    level_transition_delta=1,
                    reset_source="episode",
                )
            ],
        )

    async def fail_baseline(*args, **kwargs):
        raise AssertionError("candidate-only eval must not run a baseline")

    def fail_history(*args, **kwargs):
        raise AssertionError("candidate-only eval must not record promotion history")

    monkeypatch.setattr(ppo_agent_module, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(ppo_agent_module, "evaluate_random_policy", fail_baseline)
    monkeypatch.setattr(ppo_agent_module, "evaluate_heuristic_policy", fail_baseline)
    monkeypatch.setattr(ppo_agent_module, "_record_eval_history", fail_history)

    args = SimpleNamespace(
        endpoint="127.0.0.1:50051",
        token=None,
        agent_port=50051,
        tls=False,
        authority=None,
        skill=2,
        episode=1,
        map=1,
        seed=7,
        run_id="candidate-only",
        goal_preset="exit_seeking",
        target_x_fp=None,
        target_y_fp=None,
        eval_max_steps=6000,
        level_complete_bonus=100.0,
        kill_goal_bonus=10.0,
        required_kills=5,
        memory_path=None,
        reset_timeout_seconds=5.0,
        reset_attempts=2,
        reset_start_x_fp=None,
        reset_start_y_fp=None,
        reset_start_angle_degrees=None,
        reset_start_face_nearest_enemy=False,
        reset_start_health=None,
        reset_start_armor=None,
        reset_start_ammo_bullets=None,
        reset_start_trajectory=None,
        reset_start_index=0,
        reset_warmup_steps=0,
        reset_warmup_max_tics=0,
        reset_warmup_until_visible=False,
        reset_warmup_until_shootable=False,
        first_visible_bonus=0.0,
        first_shootable_bonus=0.0,
        visible_contact_progress_reward=0.0,
        visible_contact_loss_penalty=0.0,
        pre_shootable_route_penalty=0.0,
        pre_required_kill_route_penalty=0.0,
        exit_route_progress_reward=0.01,
        exit_route_reached_reward=0.5,
        exit_route_failure_penalty=0.05,
        exit_ready_press_reward=0.0,
        exit_ready_route_penalty=0.0,
        terminate_on_first_visible=False,
        terminate_on_first_shootable=False,
        terminate_on_required_kills=False,
        allowed_skill=[],
        strict_allowed_skills=False,
        snapshot_verify_restored_state=True,
        snapshot_verify_tick_tolerance=35,
        snapshot_verify_stream_tick=False,
        snapshot_verify_position_tolerance_fp=160 * 65536,
        eval_checkpoint=tmp_path / "candidate.pt",
        eval_episodes=5,
        eval_sample=False,
        eval_trace_jsonl=tmp_path / "candidate-trace.jsonl",
        eval_candidate_only=True,
        eval_baseline="heuristic",
        device="cpu",
        promotion_min_completion_delta=0.0,
        promotion_min_kill_delta=0.0,
        promotion_min_reward_delta=0.0,
        promotion_min_completion_rate=1.0,
        promotion_min_mean_kills=1.0,
    )

    payload = asyncio.run(evaluate(args))

    assert payload["schema"] == "restfuldoom.ppo_eval.v1"
    assert payload["checkpoint_path"] == str(tmp_path / "candidate.pt")
    assert payload["candidate"]["episodes"][0]["level_transition_delta"] == 1
    assert payload["baseline"] is None
    assert payload["promotion"] is None


def test_checkpoint_eval_stage_order_runs_true_spawn_before_snapshots():
    stages = [
        {
            "index": 0,
            "name": "1995-post-combat-exit-route_snapshot",
            "reset_mode": "snapshot",
        },
        {
            "index": 1,
            "name": "fresh_spawn_true_spawn_gate",
            "reset_mode": "episode",
            "evidence": {"true_spawn_promotion_stage": True},
        },
        {
            "index": 2,
            "name": "visible_contact_snapshot",
            "reset_mode": "snapshot",
        },
    ]

    ordered = _checkpoint_eval_stage_order(stages)

    assert [index for index, _stage in ordered] == [1, 0, 2]
    assert [stage["name"] for _index, stage in ordered] == [
        "fresh_spawn_true_spawn_gate",
        "1995-post-combat-exit-route_snapshot",
        "visible_contact_snapshot",
    ]


def test_learning_trace_names_observation_mask_and_outcome():
    feature_names = OBSERVATION_SCHEMA["feature_names"]
    obs = [0.0 for _ in feature_names]
    obs[feature_names.index("health_norm")] = 0.75
    obs[feature_names.index("combat_has_target")] = 1.0
    obs[feature_names.index("route_waypoint_distance_norm")] = 0.25
    obs[feature_names.index("remembered_enemies_norm")] = 0.5
    obs[feature_names.index("visible_enemy_seen_recently")] = 1.0
    obs[feature_names.index("prev_skill_close_visible_contact")] = 1.0
    obs[feature_names.index("contact_use_line_active")] = 1.0
    obs[feature_names.index("contact_use_line_distance_norm")] = 0.4
    obs[feature_names.index("topology_frontier_active")] = 1.0
    obs[feature_names.index("topology_exhausted_open_ratio")] = 0.25
    obs[feature_names.index("visible_contact_active")] = 1.0
    obs[feature_names.index("visible_contact_needs_closure")] = 1.0
    obs[feature_names.index("visible_contact_distance_norm")] = 0.5
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
    assert (
        trace["observation"]["groups"]["temporal"]["prev_skill_close_visible_contact"]
        == 1.0
    )
    assert trace["observation"]["groups"]["contact"]["contact_use_line_active"] == 1.0
    assert trace["observation"]["groups"]["contact"]["contact_use_line_distance_norm"] == 0.4
    assert trace["observation"]["groups"]["topology"]["topology_frontier_active"] == 1.0
    assert trace["observation"]["groups"]["topology"]["topology_exhausted_open_ratio"] == 0.25
    assert trace["observation"]["groups"]["visible_contact"]["visible_contact_active"] == 1.0
    assert trace["observation"]["groups"]["visible_contact"]["visible_contact_needs_closure"] == 1.0
    assert trace["observation"]["groups"]["visible_contact"]["visible_contact_distance_norm"] == 0.5
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


def test_promotion_gate_blocks_snapshot_verification_failures():
    baseline = EvaluationResult(
        policy_id="brain",
        level_completion_rate=1.0,
        mean_kills=5.0,
        survival_rate=1.0,
        mean_steps_to_exit=5000,
        mean_stuck_events=20,
        episode_count=3,
        mean_reward=100.0,
    )
    candidate = EvaluationResult(
        policy_id="ppo",
        level_completion_rate=1.0,
        mean_kills=6.0,
        survival_rate=1.0,
        mean_steps_to_exit=4000,
        mean_stuck_events=10,
        episode_count=3,
        mean_reward=120.0,
        snapshot_verification_failures=1,
    )

    decision = PromotionGate().decide(candidate=candidate, baseline=baseline)

    assert not decision.promote
    assert "snapshot verification failures present" in decision.reasons


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
    assert payload["episodes"][0]["skill_counts"] == {}
    assert payload["episodes"][0]["damage_delta"] == 0
    assert payload["episodes"][0]["first_shootable_contacts"] == 0
    assert not decision.promote
    assert "completion rate did not beat baseline" in decision.reasons
    assert "completion rate below promotion minimum" in decision.reasons
    assert "mean kills did not beat baseline" in decision.reasons
    assert "mean kills below promotion minimum" in decision.reasons
    assert "mean reward did not beat baseline" in decision.reasons


def test_ppo_eval_episode_serializes_contact_diagnostics():
    episode = EpisodeEval(
        seed=9,
        total_reward=12.5,
        level_completed=False,
        death=False,
        max_kills=2,
        min_health=80,
        steps=128,
        steps_to_exit=640,
        steps_to_required_kills=128,
        stuck_events=0,
        done_reason="required_kills",
        kill_delta=2,
        max_kill_gain=2,
        skill_counts={"close_visible_contact": 40, "fire": 12},
        visible_enemy_steps=54,
        first_shootable_contacts=1,
        shootable_target_steps=12,
        fire_on_shootable_steps=12,
        damage_delta=20,
        route_action_reward=1.5,
        route_attempt_steps=6,
        route_reached_steps=2,
        route_failed_steps=1,
        route_progress_units=96.5,
        exit_route_attempt_steps=3,
        exit_route_reached_steps=1,
        exit_route_failed_steps=1,
        exit_route_progress_units=32.25,
    )
    payload = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:diagnostic",
            level_completion_rate=0.0,
            mean_kills=2.0,
            survival_rate=1.0,
            mean_steps_to_exit=640,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=12.5,
        ),
        episodes=[episode],
    ).to_dict()

    row = payload["episodes"][0]
    assert row["skill_counts"] == {"close_visible_contact": 40, "fire": 12}
    assert row["first_shootable_contacts"] == 1
    assert row["shootable_target_steps"] == 12
    assert row["fire_on_shootable_steps"] == 12
    assert row["damage_delta"] == 20
    assert row["steps_to_required_kills"] == 128
    assert row["route_action_reward"] == 1.5
    assert row["route_attempt_steps"] == 6
    assert row["route_progress_units"] == 96.5
    assert row["exit_route_attempt_steps"] == 3
    assert row["exit_route_progress_units"] == 32.25


def test_expert_skill_labels_map_to_ppo_actions():
    assert map_expert_skill_to_ppo_action("fire_on_shootable_target") == 1
    assert map_expert_skill_to_ppo_action("seek_known_enemy") == 2
    assert map_expert_skill_to_ppo_action("push_exit_switch") == 7
    assert map_expert_skill_to_ppo_action("close_visible_contact") == 8
    assert map_expert_skill_to_ppo_action("ppo_seek_visible_contact") == 8
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
        "visible_contact_context",
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
    assert "visible_contact_needs_closure" in OBSERVATION_SCHEMA["feature_names"]
    assert "visible_contact_distance_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "temporal_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "contact_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "topology_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "visible_contact_context" in {
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
    old_row.extend([0.5 for _ in LEGACY_ACTION_HISTORY_FEATURE_NAMES_V1])
    old_row.extend([0.75 for _ in OBSERVATION_SCHEMA["temporal_context_feature_names"]])
    frontier_index = OBSERVATION_SCHEMA["feature_names"].index(
        "topology_frontier_count_norm"
    )
    close_contact_history_index = OBSERVATION_SCHEMA["feature_names"].index(
        "prev_skill_close_visible_contact"
    )
    contact_start = len(OBSERVATION_SCHEMA["feature_names"]) - len(
        OBSERVATION_SCHEMA["contact_context_feature_names"]
    ) - len(
        OBSERVATION_SCHEMA["topology_context_feature_names"]
    ) - len(
        OBSERVATION_SCHEMA["visible_contact_context_feature_names"]
    )

    padded = pad_observation_features(old_row)

    assert len(padded) == len(OBSERVATION_SCHEMA["feature_names"])
    assert padded[frontier_index] == 0.0
    assert padded[close_contact_history_index] == 0.0
    expected_prefix = list(old_row)
    expected_prefix.insert(frontier_index, 0.0)
    expected_prefix.insert(close_contact_history_index, 0.0)
    assert padded[:contact_start] == expected_prefix[:contact_start]
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


def test_pad_observation_features_adds_neutral_visible_contact_to_topology_rows():
    old_row = [0.25 for _ in OBSERVATION_SCHEMA["base_feature_names"]]
    old_row.extend([0.5 for _ in OBSERVATION_SCHEMA["action_history_feature_names"]])
    old_row.extend([0.75 for _ in OBSERVATION_SCHEMA["temporal_context_feature_names"]])
    old_row.extend([1.0 for _ in OBSERVATION_SCHEMA["contact_context_feature_names"]])
    old_row.extend([0.125 for _ in OBSERVATION_SCHEMA["topology_context_feature_names"]])

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


def test_rollout_summary_counts_allowed_skill_filter_fallbacks():
    buffer = SimpleNamespace(
        records=[
            SimpleNamespace(
                reward=0.0,
                done=False,
                action=0,
                action_mask=[True, False],
                info={
                    "skill": "close_visible_contact",
                    "action_mask_enforced": True,
                    "action_mask_requested_allowed": False,
                    "action_mask_fallback_applied": True,
                    "action_mask_filter": {
                        "strict": True,
                        "fallback_applied": True,
                        "fallback_skill": "close_visible_contact",
                    },
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
            SimpleNamespace(
                reward=0.0,
                done=False,
                action=1,
                action_mask=[False, True],
                info={
                    "skill": "fire",
                    "action_mask_filter": {
                        "strict": False,
                        "fallback_applied": True,
                        "fallback_skill": "unfiltered_mask",
                    },
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
        ]
    )

    summary = _summarize_buffer(buffer)

    assert summary["selected_disallowed_steps"] == 1
    assert summary["action_mask_fallback_steps"] == 1
    assert summary["allowed_skill_filter_steps"] == 2
    assert summary["allowed_skill_filter_fallback_steps"] == 2
    assert summary["strict_allowed_skill_filter_steps"] == 1
    assert summary["strict_allowed_skill_fallback_steps"] == 1
    assert summary["allowed_skill_filter_fallback_skills"] == {
        "close_visible_contact": 1,
        "unfiltered_mask": 1,
    }


def test_rollout_summary_splits_exit_route_outcomes():
    buffer = SimpleNamespace(
        records=[
            SimpleNamespace(
                reward=0.25,
                done=False,
                info={
                    "skill": "route_progression",
                    "route_outcome": {
                        "attempted": True,
                        "reached": False,
                        "failed": False,
                        "exit": False,
                        "progress_units": 8.0,
                    },
                    "route_action_reward": 0.08,
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
            SimpleNamespace(
                reward=0.5,
                done=False,
                info={
                    "skill": "route_progression",
                    "route_outcome": {
                        "attempted": True,
                        "reached": True,
                        "failed": False,
                        "exit": True,
                        "progress_units": 12.0,
                    },
                    "route_action_reward": 0.37,
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
            SimpleNamespace(
                reward=-0.25,
                done=False,
                info={
                    "skill": "route_progression",
                    "route_outcome": {
                        "attempted": True,
                        "reached": False,
                        "failed": True,
                        "exit": True,
                        "progress_units": -3.0,
                    },
                    "route_action_reward": -0.06,
                    "transition": {},
                    "state": {"health": 100, "kills": 0},
                },
            ),
        ]
    )

    summary = _summarize_buffer(buffer)

    assert summary["route_attempt_steps"] == 3
    assert summary["route_progress_units"] == 17.0
    assert summary["exit_route_attempt_steps"] == 2
    assert summary["exit_route_reached_steps"] == 1
    assert summary["exit_route_failed_steps"] == 1
    assert summary["exit_route_progress_units"] == 9.0


def test_rollout_summary_counts_exit_ready_handoff_choices():
    buffer = SimpleNamespace(
        records=[
            SimpleNamespace(
                reward=1.25,
                done=False,
                action=7,
                action_mask=[False, False, False, False, True, False, False, True],
                info={
                    "skill": "press_exit",
                    "exit_ready_press_available": True,
                    "exit_ready_switch_attempt": True,
                    "exit_ready_action_reward": 1.25,
                    "transition": {},
                    "state": {"health": 100, "kills": 6},
                },
            ),
            SimpleNamespace(
                reward=-0.6,
                done=False,
                action=4,
                action_mask=[False, False, False, False, True, False, False, True],
                info={
                    "skill": "route_progression",
                    "exit_ready_press_available": True,
                    "exit_ready_switch_attempt": True,
                    "exit_ready_action_reward": -0.6,
                    "transition": {},
                    "state": {"health": 100, "kills": 6},
                },
            ),
            SimpleNamespace(
                reward=0.0,
                done=False,
                action=4,
                action_mask=[False, False, False, False, True, False, False, False],
                info={
                    "skill": "route_progression",
                    "exit_ready_press_available": False,
                    "exit_ready_switch_attempt": False,
                    "exit_ready_action_reward": 0.0,
                    "transition": {},
                    "state": {"health": 100, "kills": 6},
                },
            ),
        ]
    )

    summary = _summarize_buffer(buffer)

    assert summary["exit_ready_press_available_steps"] == 2
    assert summary["exit_ready_press_selected_steps"] == 1
    assert summary["exit_ready_route_selected_steps"] == 1
    assert summary["exit_ready_switch_attempt_steps"] == 2
    assert summary["exit_ready_action_reward"] == pytest.approx(0.65)


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


def test_named_curriculum_selects_true_spawn_contact_bridge_stages():
    curriculum = build_curriculum(
        name="e1m1-true-spawn-contact-bridge",
        manual_reset_start={},
        mode="round_robin",
        start_index=0,
        seed=7,
    )

    spawn = stage_for_update(curriculum, 0)
    visible = stage_for_update(curriculum, 1)
    combat = stage_for_update(curriculum, 4)

    assert curriculum["schema"] == "restfuldoom.ppo_curriculum.v1"
    assert spawn["name"] == "fresh_spawn"
    assert spawn["reset_start"] == {}
    assert spawn["evidence"]["true_spawn_gate_bottleneck"] == "first_contact"
    assert spawn["evidence"]["latest_true_spawn_gate"]["first_shootable_contacts"] == 0
    assert visible["name"] == "visible_contact_fast"
    assert visible["evidence"]["visible_enemy_on_reset"] is True
    assert visible["evidence"]["shootable_target_on_reset"] is False
    assert combat["name"] == "combat_start"
    assert combat["evidence"]["shootable_target_on_reset"] is True


def test_behavior_clone_loader_accepts_forced_option_records(tmp_path):
    trajectory = tmp_path / "forced.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.forced_option_eval_record.v1",
                "forced_skill": "close_visible_contact",
                "record": {
                    "obs": [0.25 for _ in OBSERVATION_SCHEMA["feature_names"]],
                    "action": 8,
                    "action_mask": [False, False, False, False, False, False, False, False, True],
                    "reward": 1.0,
                    "done": False,
                    "info": {
                        "selected_action_allowed": True,
                        "selected_forced_skill": "close_visible_contact",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples, summary = _load_behavior_clone_samples(
        SimpleNamespace(bc_trajectory=[trajectory], bc_max_samples=16)
    )
    aux_samples, aux_summary = _load_behavior_clone_samples(
        SimpleNamespace(aux_bc_trajectory=[trajectory], aux_bc_max_samples=16),
        trajectory_attr="aux_bc_trajectory",
        max_samples_attr="aux_bc_max_samples",
    )

    assert len(samples) == 1
    assert samples[0][1] == 8
    assert samples[0][2] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert len(samples[0][0]) == len(OBSERVATION_SCHEMA["feature_names"])
    assert summary["samples"] == 1
    assert summary["action_masked_samples"] == 1
    assert summary["expert_skill_counts"] == {"close_visible_contact": 1}
    assert summary["ppo_skill_counts"] == {"close_visible_contact": 1}
    assert aux_samples == samples
    assert aux_summary["trajectory_paths"] == summary["trajectory_paths"]


def test_behavior_clone_loader_accepts_ppo_eval_trace_records(tmp_path):
    trajectory = tmp_path / "eval-trace.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.ppo_eval_trace.v1",
                "policy_id": "ppo:test",
            }
        )
        + "\n"
        + json.dumps(
            {
                "observation": [0.5 for _ in OBSERVATION_SCHEMA["feature_names"]],
                "action_index": 4,
                "action_mask": [
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                "skill": "advance_route",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples, summary = _load_behavior_clone_samples(
        SimpleNamespace(bc_trajectory=[trajectory], bc_max_samples=16)
    )

    assert len(samples) == 1
    assert samples[0][1] == 4
    assert samples[0][2] == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert summary["skipped"] == 1
    assert summary["action_masked_samples"] == 1
    assert summary["expert_skill_counts"] == {"advance_route": 1}
    assert summary["ppo_skill_counts"] == {"route_progression": 1}


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


def test_snapshot_curriculum_manifest_loads_progressed_stages(tmp_path):
    manifest = tmp_path / "snapshots.json"
    snapshot = tmp_path / "first-contact.snap"
    snapshot.write_bytes(b"snapshot")
    manifest.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.snapshot_curriculum.v1",
                "name": "e1m1-progressed",
                "source": {"run_id": "brain-success"},
                "stages": [
                    {
                        "name": "first_contact_snapshot",
                        "snapshot": {
                            "id": "snap-1",
                            "path": str(snapshot),
                            "digest": "sha256:test",
                        },
                        "expected_state": {"episode": 1, "map": 1, "tick": 1200},
                        "evidence": {"source_record_index": 47},
                        "training": {
                            "required_kills": 1,
                            "terminate_on_required_kills": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    curriculum = load_snapshot_curriculum(
        manifest,
        mode="fixed",
        start_index=0,
        seed=11,
    )
    stage = stage_for_update(curriculum, 0)

    assert curriculum["schema"] == "restfuldoom.ppo_curriculum.v1"
    assert curriculum["source_schema"] == "restfuldoom.snapshot_curriculum.v1"
    assert curriculum["snapshot_curriculum"]["manifest_path"] == str(manifest)
    assert stage["name"] == "first_contact_snapshot"
    assert stage["reset_mode"] == "snapshot"
    assert stage["requires_progressed_state"] is True
    assert stage["snapshot"]["id"] == "snap-1"
    assert stage["evidence"]["snapshot_backed"] is True
    assert stage["training"] == {
        "required_kills": 1,
        "terminate_on_required_kills": True,
    }


def test_true_spawn_stage_can_be_appended_to_snapshot_curriculum(tmp_path):
    manifest = tmp_path / "snapshots.json"
    snapshot = tmp_path / "post-combat.snap"
    snapshot.write_bytes(b"snapshot")
    manifest.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.snapshot_curriculum.v1",
                "name": "post-combat",
                "stages": [
                    {
                        "name": "post_combat_snapshot",
                        "snapshot": {
                            "id": "post-combat",
                            "path": str(snapshot),
                        },
                        "expected_state": {"kills": 6},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    curriculum = load_snapshot_curriculum(manifest)

    mixed = _append_true_spawn_stage(
        SimpleNamespace(
            include_true_spawn_stage=True,
            true_spawn_stage_name="fresh_spawn_gate",
        ),
        curriculum,
    )

    assert mixed["includes_true_spawn_stage"] is True
    assert [stage["name"] for stage in mixed["stages"]] == [
        "post_combat_snapshot",
        "fresh_spawn_gate",
    ]
    snapshot_stage, true_spawn_stage = mixed["stages"]
    assert snapshot_stage["reset_mode"] == "snapshot"
    assert snapshot_stage["requires_progressed_state"] is True
    assert true_spawn_stage["reset_mode"] == "episode"
    assert true_spawn_stage["reset_start"] == {}
    assert true_spawn_stage["requires_progressed_state"] is False
    assert "snapshot" not in true_spawn_stage
    assert true_spawn_stage["evidence"]["true_spawn_promotion_stage"] is True


def test_snapshot_stage_training_overrides_required_kill_goal():
    args = _stage_config_args()
    curriculum = {
        "schema": "restfuldoom.ppo_curriculum.v1",
        "name": "gate-b-anchors",
        "stages": [],
    }
    stage = {
        "index": 0,
        "name": "0434-pre-required-kill_snapshot",
        "reset_mode": "snapshot",
        "reset_start": {"x_fp": 1, "y_fp": 2},
        "snapshot": {"id": "slot-5", "slot": 5, "ref": "save_slot:5"},
        "training": {
            "required_kills": 1,
            "terminate_on_required_kills": True,
        },
    }

    config = _env_config_for_stage(
        args,
        curriculum,
        stage,
        run_id="gate-b-pre-required",
    )

    assert config.required_kills == 1
    assert config.terminate_on_required_kills is True
    assert config.reset_mode == "snapshot"
    assert config.snapshot == {"id": "slot-5", "slot": 5, "ref": "save_slot:5"}
    assert config.curriculum_stage["training"]["required_kills"] == 1


def test_snapshot_stage_training_override_does_not_leak_to_normal_stage():
    args = _stage_config_args(required_kills=5, terminate_on_required_kills=False)
    curriculum = {
        "schema": "restfuldoom.ppo_curriculum.v1",
        "name": "gate-b-anchors",
        "stages": [],
    }
    stage = {
        "index": 1,
        "name": "fresh_spawn_true_spawn_gate",
        "reset_mode": "episode",
        "reset_start": {},
    }

    config = _env_config_for_stage(
        args,
        curriculum,
        stage,
        run_id="gate-b-true-spawn",
    )

    assert config.required_kills == 5
    assert config.terminate_on_required_kills is False
    assert config.reset_mode == "episode"


def test_snapshot_curriculum_manifest_rejects_bad_schema(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"schema": "old", "stages": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected restfuldoom.snapshot_curriculum.v1"):
        load_snapshot_curriculum(manifest)


def _stage_config_args(
    *,
    required_kills: int = 5,
    terminate_on_required_kills: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        endpoint="127.0.0.1:50051",
        token=None,
        agent_port=50051,
        tls=False,
        authority=None,
        skill=2,
        episode=1,
        map=1,
        seed=7,
        run_id="stage-config-test",
        goal_preset="exit_seeking",
        target_x_fp=None,
        target_y_fp=None,
        max_steps=700,
        level_complete_bonus=100.0,
        kill_goal_bonus=10.0,
        required_kills=required_kills,
        memory_path=None,
        reset_timeout_seconds=5.0,
        reset_attempts=2,
        reset_start_angle_degrees=None,
        reset_start_face_nearest_enemy=False,
        reset_start_health=None,
        reset_start_armor=None,
        reset_start_ammo_bullets=None,
        reset_warmup_steps=0,
        reset_warmup_max_tics=0,
        reset_warmup_until_visible=False,
        reset_warmup_until_shootable=False,
        first_visible_bonus=0.0,
        first_shootable_bonus=0.0,
        visible_contact_progress_reward=0.0,
        visible_contact_loss_penalty=0.0,
        pre_shootable_route_penalty=0.0,
        pre_required_kill_route_penalty=0.0,
        exit_route_progress_reward=0.01,
        exit_route_reached_reward=0.5,
        exit_route_failure_penalty=0.05,
        exit_ready_press_reward=0.0,
        exit_ready_route_penalty=0.0,
        terminate_on_first_visible=False,
        terminate_on_first_shootable=False,
        terminate_on_required_kills=terminate_on_required_kills,
        allowed_skill=[],
        strict_allowed_skills=False,
    )


def test_snapshot_restore_command_rendering_redacts_secrets():
    stage = {
        "name": "first_contact",
        "selected_index": 2,
        "snapshot": {
            "id": "snap one",
            "path": "/tmp/snap one.bin",
            "microvm_id": "vm-1",
        },
        "expected_state": {"tick": 123},
    }

    command = _render_snapshot_restore_command(
        "shrink restore --name {microvm_id} --snapshot {snapshot_path_sh} --token abc",
        stage,
        update_index=4,
        reset_index=5,
    )
    argv = command.split()

    assert "/tmp/snap one.bin" in command
    assert _redacted_restore_argv(argv)[-1] == "<redacted>"


def test_snapshot_restore_uses_grpc_slot_without_external_command():
    stage = {
        "name": "first_contact",
        "index": 2,
        "snapshot": {
            "id": "slot-4",
            "slot": 4,
            "ref": "save_slot:4",
        },
    }
    args = SimpleNamespace(snapshot_restore_command=None)

    restore = _restore_snapshot_for_stage(stage, args, update_index=3, reset_index=5)

    assert restore["schema"] == "restfuldoom.snapshot_restore.v1"
    assert restore["api_method"] == "grpc_load_snapshot"
    assert restore["restore_command_configured"] is False
    assert restore["slot"] == 4
    assert restore["returncode"] == 0
    assert restore["update_index"] == 3
    assert restore["reset_index"] == 5


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


def test_rollout_summary_counts_snapshot_restore_contexts():
    buffer = RolloutBuffer()
    for episode_index, stage_name in [(1, "first_contact_snapshot"), (1, "first_contact_snapshot"), (2, "exit_room_snapshot")]:
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
                "curriculum_stage": {
                    "name": stage_name,
                    "reset_mode": "snapshot",
                },
                "reset_context": {
                    "schema": "restfuldoom.reset_context.v1",
                    "source": "snapshot_restore",
                    "episode_index": episode_index,
                    "restore": {
                        "returncode": 0,
                        "elapsed_seconds": 0.25 * episode_index,
                    },
                },
            },
        )

    summary = _summarize_buffer(buffer)

    assert summary["reset_context_sources"] == {"snapshot_restore": 2}
    assert summary["snapshot_restore_count"] == 2
    assert summary["snapshot_restore_failures"] == 0
    assert summary["snapshot_stage_counts"] == {
        "exit_room_snapshot": 1,
        "first_contact_snapshot": 2,
    }
    assert summary["mean_snapshot_restore_seconds"] == pytest.approx(0.375)
    assert summary["max_snapshot_restore_seconds"] == pytest.approx(0.5)


def test_rollout_summary_separates_inherited_snapshot_kills_from_earned_kills():
    buffer = RolloutBuffer()
    reset_context = {
        "schema": "restfuldoom.reset_context.v1",
        "source": "snapshot_restore",
        "episode_index": 1,
        "actual_first_state": {"kills": 1, "health": 96},
        "expected_state": {"kills": 1},
        "restore": {"returncode": 0, "elapsed_seconds": 0.1},
        "restored_state_verification": {"valid": True},
    }
    for state_kills, kill_delta in [(1, 0), (1, 0), (2, 1)]:
        buffer.add(
            obs=[0.0, 1.0],
            action_mask=[True, False],
            action=0,
            reward=float(kill_delta),
            done=False,
            value=0.0,
            logprob=0.0,
            info={
                "skill": "fire",
                "transition": {"kill_delta": kill_delta, "damage_delta": 0},
                "state": {"health": 96, "kills": state_kills},
                "curriculum_stage": {
                    "name": "first_kill_snapshot",
                    "reset_mode": "snapshot",
                    "expected_state": {"kills": 1},
                },
                "reset_context": reset_context,
            },
        )

    summary = _summarize_buffer(buffer)

    assert summary["max_kills"] == 2
    assert summary["kill_delta"] == 1
    assert summary["max_kill_gain"] == 1
    assert summary["snapshot_kill_delta"] == 1
    assert summary["snapshot_max_kill_gain"] == 1


def test_eval_detects_current_snapshot_restore_verification_failure_key():
    reset_context = {
        "schema": "restfuldoom.reset_context.v1",
        "source": "snapshot_restore",
        "restored_state_verification": {
            "schema": "restfuldoom.snapshot_restored_state_verification.v1",
            "enabled": True,
            "valid": False,
            "mismatches": [{"field": "level_time", "expected": 500, "actual": 264}],
        },
    }

    assert _reset_context_snapshot_verification_failed(reset_context) is True


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


def test_rollout_summary_counts_visible_contact_context_from_learning_trace():
    buffer = RolloutBuffer()
    for visible_contact in [
        {
            "visible_contact_active": 1.0,
            "visible_contact_shootable": 0.0,
            "visible_contact_needs_closure": 1.0,
            "visible_contact_distance_norm": 0.5,
            "visible_contact_aligned": 1.0,
            "visible_contact_close": 0.0,
        },
        {
            "visible_contact_active": 1.0,
            "visible_contact_shootable": 1.0,
            "visible_contact_needs_closure": 0.0,
            "visible_contact_distance_norm": 0.25,
            "visible_contact_aligned": 1.0,
            "visible_contact_close": 1.0,
        },
        {
            "visible_contact_active": 0.0,
            "visible_contact_shootable": 0.0,
            "visible_contact_needs_closure": 0.0,
            "visible_contact_distance_norm": 0.0,
            "visible_contact_aligned": 0.0,
            "visible_contact_close": 0.0,
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
                            "visible_contact": visible_contact,
                        },
                    },
                },
                "transition": {},
                "state": {"health": 100, "kills": 0},
            },
        )

    summary = _summarize_buffer(buffer)

    assert summary["visible_contact_active_steps"] == 2
    assert summary["visible_contact_needs_closure_steps"] == 1
    assert summary["visible_contact_shootable_steps"] == 1
    assert summary["visible_contact_aligned_steps"] == 2
    assert summary["visible_contact_close_steps"] == 1
    assert summary["mean_visible_contact_distance_norm"] == 0.375


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
            "visible_contact_loss_penalty": -0.5,
            "pre_shootable_route_penalty": -0.25,
            "pre_required_kill_route_penalty": -0.3,
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
            "visible_contact_loss_penalty": 0.0,
            "pre_shootable_route_penalty": 0.0,
            "pre_required_kill_route_penalty": 0.0,
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
            "visible_contact_loss_penalty": -0.25,
            "pre_shootable_route_penalty": 0.0,
            "pre_required_kill_route_penalty": -0.2,
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
    assert summary["visible_contact_loss_penalty"] == -0.75
    assert summary["pre_shootable_route_penalty"] == -0.25
    assert summary["pre_required_kill_route_penalty"] == -0.5


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


def test_checkpoint_selection_score_ignores_inherited_snapshot_kills():
    inherited = {
        "total_reward": 0.0,
        "max_kills": 5,
        "max_kill_gain": 0,
        "kill_delta": 0,
        "damage_delta": 0,
        "first_shootable_contacts": 0,
        "fire_on_shootable_steps": 0,
        "missed_shootable_fire_steps": 0,
        "route_failed_steps": 0,
    }
    earned = {
        **inherited,
        "max_kills": 1,
        "max_kill_gain": 1,
    }

    assert _checkpoint_selection_score(inherited) == 0.0
    assert _checkpoint_selection_score(earned) == 75.0


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


def test_policy_eval_selection_score_prefers_faster_required_kill():
    slow = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:slow",
            level_completion_rate=0.0,
            mean_kills=1.0,
            survival_rate=1.0,
            mean_steps_to_exit=640,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=20.0,
        ),
        episodes=[
            EpisodeEval(
                seed=7,
                total_reward=20.0,
                level_completed=False,
                death=False,
                max_kills=1,
                min_health=100,
                steps=400,
                steps_to_exit=640,
                steps_to_required_kills=400,
                stuck_events=0,
                done_reason="required_kills",
                kill_delta=1,
                max_kill_gain=1,
            )
        ],
    )
    fast = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:fast",
            level_completion_rate=0.0,
            mean_kills=1.0,
            survival_rate=1.0,
            mean_steps_to_exit=640,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=20.0,
        ),
        episodes=[
            EpisodeEval(
                seed=8,
                total_reward=20.0,
                level_completed=False,
                death=False,
                max_kills=1,
                min_health=100,
                steps=100,
                steps_to_exit=640,
                steps_to_required_kills=100,
                stuck_events=0,
                done_reason="required_kills",
                kill_delta=1,
                max_kill_gain=1,
            )
        ],
    )

    assert _policy_eval_selection_score(fast) > _policy_eval_selection_score(slow)


def test_required_kill_selection_score_caps_reward_tiebreak_for_speed():
    slow_high_reward = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:slow-high-reward",
            level_completion_rate=0.0,
            mean_kills=1.0,
            survival_rate=1.0,
            mean_steps_to_exit=640,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=55.0,
        ),
        episodes=[
            EpisodeEval(
                seed=7,
                total_reward=55.0,
                level_completed=False,
                death=False,
                max_kills=1,
                min_health=100,
                steps=240,
                steps_to_exit=640,
                steps_to_required_kills=240,
                stuck_events=0,
                done_reason="required_kills",
                kill_delta=1,
                max_kill_gain=1,
            )
        ],
    )
    fast_lower_reward = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:fast-lower-reward",
            level_completion_rate=0.0,
            mean_kills=1.0,
            survival_rate=1.0,
            mean_steps_to_exit=640,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=10.0,
        ),
        episodes=[
            EpisodeEval(
                seed=8,
                total_reward=10.0,
                level_completed=False,
                death=False,
                max_kills=1,
                min_health=100,
                steps=100,
                steps_to_exit=640,
                steps_to_required_kills=100,
                stuck_events=0,
                done_reason="required_kills",
                kill_delta=1,
                max_kill_gain=1,
            )
        ],
    )

    slow_components = _policy_eval_selection_components(slow_high_reward)
    fast_components = _policy_eval_selection_components(fast_lower_reward)

    assert slow_components["mode"] == "required_kill_speed"
    assert slow_components["reward_tiebreak"] == 20.0
    assert fast_components["required_kill_speed_bonus"] > slow_components[
        "required_kill_speed_bonus"
    ]
    assert fast_components["selection_score"] > slow_components["selection_score"]


def test_exit_routing_selection_score_prefers_fast_exit_over_slow_reward():
    slow_high_reward = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:slow-exit",
            level_completion_rate=1.0,
            mean_kills=0.0,
            survival_rate=1.0,
            mean_steps_to_exit=600,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=120.0,
        ),
        episodes=[
            EpisodeEval(
                seed=7,
                total_reward=120.0,
                level_completed=True,
                death=False,
                max_kills=5,
                min_health=80,
                steps=300,
                steps_to_exit=600,
                stuck_events=0,
                done_reason="level_complete",
                start_kills=5,
            )
        ],
    )
    fast_lower_reward = PolicyEval(
        result=EvaluationResult(
            policy_id="ppo:fast-exit",
            level_completion_rate=1.0,
            mean_kills=0.0,
            survival_rate=1.0,
            mean_steps_to_exit=160,
            mean_stuck_events=0.0,
            episode_count=1,
            mean_reward=10.0,
        ),
        episodes=[
            EpisodeEval(
                seed=8,
                total_reward=10.0,
                level_completed=True,
                death=False,
                max_kills=5,
                min_health=80,
                steps=80,
                steps_to_exit=160,
                stuck_events=0,
                done_reason="level_complete",
                start_kills=5,
            )
        ],
    )

    slow_components = _policy_eval_selection_components(slow_high_reward)
    fast_components = _policy_eval_selection_components(fast_lower_reward)

    assert slow_components["mode"] == "exit_routing_speed"
    assert slow_components["reward_tiebreak"] == 20.0
    assert fast_components["exit_speed_bonus"] > slow_components["exit_speed_bonus"]
    assert fast_components["selection_score"] > slow_components["selection_score"]


def test_post_combat_snapshot_selection_score_caps_no_exit_credit():
    post_combat_stage = {
        "name": "1420-post-combat_snapshot",
        "evidence": {
            "selector": "post-combat",
            "selectors": ["post-combat"],
        },
    }
    eval_result = _aggregate(
        "ppo:post-combat-no-exit",
        [
            EpisodeEval(
                seed=7,
                total_reward=210.0035,
                level_completed=False,
                death=False,
                max_kills=6,
                min_health=84,
                steps=1428,
                steps_to_exit=4000,
                stuck_events=0,
                done_reason="max_steps",
                start_kills=2,
                kill_delta=4,
                max_kill_gain=4,
                reset_source="snapshot_restore",
            )
        ],
    )

    components = _policy_eval_selection_components(
        eval_result,
        stage=post_combat_stage,
    )

    assert components["schema"] == "restfuldoom.ppo_checkpoint_eval_score.v5"
    assert components["mode"] == "post_combat_exit_routing"
    assert components["exit_success_rate"] == 0.0
    assert components["exit_speed_bonus"] == 0.0
    assert components["reward_tiebreak"] == 20.0
    assert components["earned_kill_bonus"] == 0.0
    assert components["selection_score"] == 40.0


def test_required_kill_selection_score_takes_priority_over_post_combat_hint():
    post_combat_stage = {
        "name": "1420-post-combat_snapshot",
        "evidence": {
            "selector": "post-combat-exit-route",
            "selectors": ["post-combat", "post-combat-exit-route"],
        },
    }
    eval_result = _aggregate(
        "ppo:post-combat-required-kill",
        [
            EpisodeEval(
                seed=7,
                total_reward=60.0,
                level_completed=False,
                death=False,
                max_kills=3,
                min_health=100,
                steps=120,
                steps_to_exit=4000,
                stuck_events=0,
                done_reason="required_kills",
                steps_to_required_kills=120,
                start_kills=2,
                kill_delta=1,
                max_kill_gain=1,
                reset_source="snapshot_restore",
            )
        ],
    )

    components = _policy_eval_selection_components(
        eval_result,
        stage=post_combat_stage,
    )

    assert components["mode"] == "required_kill_speed"
    assert components["required_kill_success_rate"] == 1.0
    assert components["earned_kill_bonus"] == 120.0


def test_true_spawn_promotion_selection_score_requires_full_chain():
    true_spawn_stage = {
        "name": "fresh_spawn_true_spawn_gate",
        "evidence": {"true_spawn_promotion_stage": True},
    }
    kills_only = _aggregate(
        "ppo:true-spawn-kills-only",
        [
            EpisodeEval(
                seed=7,
                total_reward=220.0,
                level_completed=False,
                death=False,
                max_kills=5,
                min_health=32,
                steps=1000,
                steps_to_exit=6000,
                steps_to_required_kills=1000,
                stuck_events=4,
                done_reason="required_kills",
                start_kills=0,
                kill_delta=5,
                max_kill_gain=5,
                reset_source="episode",
                first_visible_contacts=1,
                first_shootable_contacts=1,
                route_attempt_steps=300,
                exit_route_attempt_steps=20,
            )
        ],
    )
    completed = _aggregate(
        "ppo:true-spawn-complete",
        [
            EpisodeEval(
                seed=7,
                total_reward=90.0,
                level_completed=True,
                death=False,
                max_kills=5,
                min_health=40,
                steps=1300,
                steps_to_exit=1300,
                stuck_events=1,
                done_reason="level_complete",
                start_kills=0,
                kill_delta=5,
                max_kill_gain=5,
                reset_source="episode",
                start_episode=1,
                start_map=1,
                end_episode=1,
                end_map=2,
                level_transition_delta=1,
                first_visible_contacts=1,
                first_shootable_contacts=1,
                allowed_skill_filter_steps=1300,
                strict_allowed_skill_filter_steps=1300,
                skill_counts={"route_progression": 5, "fire": 2, "press_exit": 1},
                route_attempt_steps=420,
                exit_route_attempt_steps=70,
                exit_route_reached_steps=8,
            )
        ],
    )

    kills_only_components = _policy_eval_selection_components(
        kills_only,
        stage=true_spawn_stage,
    )
    completed_components = _policy_eval_selection_components(
        completed,
        stage=true_spawn_stage,
    )

    assert kills_only_components["mode"] == "true_spawn_promotion"
    assert completed_components["mode"] == "true_spawn_promotion"
    assert kills_only_components["completion_rate"] == 0.0
    assert completed_components["completion_rate"] == 1.0
    assert kills_only_components["gate_ok"] is False
    assert completed_components["gate_ok"] is True
    assert completed_components["true_spawn_gate"]["ok"] is True
    assert completed_components["true_spawn_gate"]["summary"]["passed_episodes"] == 1
    assert completed_components["selection_score"] > kills_only_components[
        "selection_score"
    ]


def test_true_spawn_promotion_selection_score_penalizes_non_episode_reset():
    true_spawn_stage = {
        "name": "fresh_spawn_true_spawn_gate",
        "evidence": {"true_spawn_promotion_stage": True},
    }
    snapshot_eval = _aggregate(
        "ppo:true-spawn-snapshot",
        [
            EpisodeEval(
                seed=7,
                total_reward=90.0,
                level_completed=True,
                death=False,
                max_kills=5,
                min_health=40,
                steps=1300,
                steps_to_exit=1300,
                stuck_events=1,
                done_reason="level_complete",
                start_kills=0,
                kill_delta=5,
                max_kill_gain=5,
                reset_source="snapshot_restore",
                level_transition_delta=1,
                first_visible_contacts=1,
                first_shootable_contacts=1,
                route_attempt_steps=420,
                exit_route_attempt_steps=70,
            )
        ],
    )

    components = _policy_eval_selection_components(snapshot_eval, stage=true_spawn_stage)

    assert components["mode"] == "true_spawn_promotion"
    assert components["valid_true_spawn_rate"] == 0.0
    assert components["completion_rate"] == 0.0
    assert components["true_spawn_gate"]["ok"] is False


def test_true_spawn_promotion_selection_score_gates_route_credit_after_kills():
    true_spawn_stage = {
        "name": "fresh_spawn_true_spawn_gate",
        "evidence": {"true_spawn_promotion_stage": True},
    }
    dead_before_required_kills = _aggregate(
        "ppo:true-spawn-dead-route",
        [
            EpisodeEval(
                seed=7,
                total_reward=-450.0,
                level_completed=False,
                death=True,
                max_kills=4,
                min_health=-6,
                steps=1300,
                steps_to_exit=6000,
                stuck_events=19,
                done_reason="death",
                start_kills=0,
                kill_delta=4,
                max_kill_gain=4,
                reset_source="episode",
                first_visible_contacts=1,
                first_shootable_contacts=1,
                route_attempt_steps=450,
                exit_route_attempt_steps=12,
            )
        ],
    )
    alive_at_required_kills = _aggregate(
        "ppo:true-spawn-alive-route",
        [
            EpisodeEval(
                seed=8,
                total_reward=-60.0,
                level_completed=False,
                death=False,
                max_kills=5,
                min_health=10,
                steps=1900,
                steps_to_exit=6000,
                stuck_events=120,
                done_reason="max_steps",
                start_kills=0,
                kill_delta=5,
                max_kill_gain=5,
                reset_source="episode",
                first_visible_contacts=1,
                first_shootable_contacts=1,
                route_attempt_steps=800,
                exit_route_attempt_steps=120,
            )
        ],
    )

    dead_components = _policy_eval_selection_components(
        dead_before_required_kills,
        stage=true_spawn_stage,
    )
    alive_components = _policy_eval_selection_components(
        alive_at_required_kills,
        stage=true_spawn_stage,
    )

    assert dead_components["exit_route_attempt_rate"] == 0.0
    assert dead_components["death_penalty"] == -300.0
    assert alive_components["exit_route_attempt_rate"] == 1.0
    assert alive_components["selection_score"] > dead_components["selection_score"]


def test_policy_eval_aggregates_earned_kills_not_restored_snapshot_kills():
    inherited = EpisodeEval(
        seed=7,
        total_reward=0.0,
        level_completed=False,
        death=False,
        max_kills=3,
        min_health=100,
        steps=64,
        steps_to_exit=64,
        stuck_events=0,
        done_reason="max_steps",
        start_kills=3,
        kill_delta=0,
        max_kill_gain=0,
    )
    earned = EpisodeEval(
        seed=8,
        total_reward=10.0,
        level_completed=False,
        death=False,
        max_kills=4,
        min_health=96,
        steps=64,
        steps_to_exit=64,
        stuck_events=0,
        done_reason="max_steps",
        start_kills=3,
        kill_delta=1,
        max_kill_gain=1,
        max_items=3,
        start_items=2,
        item_delta=1,
        max_item_gain=1,
        max_secrets=1,
        start_secrets=1,
        secret_delta=0,
        max_secret_gain=0,
        start_episode=1,
        start_map=1,
        end_episode=1,
        end_map=1,
        level_transition_delta=0,
        reset_source="snapshot_restore",
    )

    inherited_eval = _aggregate("ppo:inherited", [inherited])
    earned_eval = _aggregate("ppo:earned", [earned])

    assert inherited_eval.result.mean_kills == 0.0
    assert inherited_eval.result.mean_items == 0.0
    assert inherited_eval.result.mean_item_gain == 0.0
    assert inherited_eval.result.mean_secrets == 0.0
    assert inherited_eval.result.mean_secret_gain == 0.0
    assert inherited_eval.episodes[0].start_kills == 3
    assert inherited_eval.episodes[0].max_kill_gain == 0
    assert earned_eval.result.mean_kills == 1.0
    assert earned_eval.result.mean_items == 1.0
    assert earned_eval.result.mean_item_gain == 1.0
    assert earned_eval.result.reset_source_breakdown["snapshot_restore"]["mean_items"] == 1.0
    assert (
        earned_eval.result.reset_source_breakdown["snapshot_restore"]["mean_item_gain"]
        == 1.0
    )
    assert earned_eval.to_dict()["episodes"][0]["start_items"] == 2
    assert earned_eval.to_dict()["episodes"][0]["max_item_gain"] == 1
    assert earned_eval.to_dict()["episodes"][0]["reset_source"] == "snapshot_restore"
    assert _policy_eval_selection_score(earned_eval) > _policy_eval_selection_score(
        inherited_eval
    )


def test_policy_eval_aggregates_snapshot_verification_failures():
    ok = EpisodeEval(
        seed=7,
        total_reward=1.0,
        level_completed=False,
        death=False,
        max_kills=0,
        min_health=100,
        steps=64,
        steps_to_exit=64,
        stuck_events=0,
        done_reason="max_steps",
        snapshot_verification_failures=0,
        route_action_reward=0.75,
        route_attempt_steps=2,
        route_reached_steps=1,
        route_progress_units=16.0,
        allowed_skill_filter_steps=2,
        strict_allowed_skill_filter_steps=2,
        exit_route_attempt_steps=1,
        exit_route_reached_steps=1,
        exit_route_progress_units=8.0,
    )
    failed = EpisodeEval(
        seed=8,
        total_reward=1.0,
        level_completed=False,
        death=False,
        max_kills=0,
        min_health=100,
        steps=64,
        steps_to_exit=64,
        stuck_events=0,
        done_reason="max_steps",
        snapshot_verification_failures=1,
        invalid_action_steps=1,
        selected_disallowed_steps=1,
        action_mask_fallback_steps=1,
        allowed_skill_filter_steps=1,
        allowed_skill_filter_fallback_steps=1,
        strict_allowed_skill_filter_steps=1,
        strict_allowed_skill_fallback_steps=1,
        route_action_reward=-0.25,
        route_attempt_steps=1,
        route_failed_steps=1,
        route_progress_units=-4.0,
        exit_route_attempt_steps=1,
        exit_route_failed_steps=1,
        exit_route_progress_units=-4.0,
    )

    result = _aggregate("ppo:snapshot-failure", [ok, failed])

    assert result.result.snapshot_verification_failures == 1
    assert result.result.invalid_action_steps == 1
    assert result.result.selected_disallowed_steps == 1
    assert result.result.action_mask_fallback_steps == 1
    assert result.result.allowed_skill_filter_steps == 3
    assert result.result.allowed_skill_filter_fallback_steps == 1
    assert result.result.strict_allowed_skill_filter_steps == 3
    assert result.result.strict_allowed_skill_fallback_steps == 1
    payload = result.to_dict()["result"]
    assert payload["snapshot_verification_failures"] == 1
    assert payload["selected_disallowed_steps"] == 1
    assert payload["strict_allowed_skill_filter_steps"] == 3
    assert payload["strict_allowed_skill_fallback_steps"] == 1
    assert payload["route_action_reward"] == 0.5
    assert payload["route_attempt_steps"] == 3
    assert payload["route_reached_steps"] == 1
    assert payload["route_failed_steps"] == 1
    assert payload["route_progress_units"] == 12.0
    assert payload["exit_route_attempt_steps"] == 2
    assert payload["exit_route_reached_steps"] == 1
    assert payload["exit_route_failed_steps"] == 1
    assert payload["exit_route_progress_units"] == 4.0


def test_policy_eval_reports_reset_source_breakdown():
    snapshot_episode = EpisodeEval(
        seed=7,
        total_reward=20.0,
        level_completed=True,
        death=False,
        max_kills=5,
        min_health=80,
        steps=96,
        steps_to_exit=192,
        stuck_events=1,
        done_reason="level_complete",
        start_kills=5,
        reset_source="snapshot_restore",
        route_action_reward=2.5,
        route_attempt_steps=4,
        route_reached_steps=1,
        route_progress_units=128.0,
        exit_route_attempt_steps=4,
        exit_route_reached_steps=1,
        exit_route_progress_units=128.0,
    )
    fresh_episode = EpisodeEval(
        seed=8,
        total_reward=-5.0,
        level_completed=False,
        death=False,
        max_kills=1,
        min_health=70,
        steps=256,
        steps_to_exit=256,
        stuck_events=4,
        done_reason="max_steps",
        start_kills=0,
        kill_delta=1,
        max_kill_gain=1,
        reset_source="reset_episode",
        route_action_reward=-0.5,
        route_attempt_steps=3,
        route_failed_steps=2,
        route_progress_units=-12.0,
    )

    result = _aggregate("ppo:mixed-reset", [snapshot_episode, fresh_episode])
    breakdown = result.to_dict()["result"]["reset_source_breakdown"]

    assert breakdown["snapshot_restore"]["episode_count"] == 1
    assert breakdown["snapshot_restore"]["level_completion_rate"] == 1.0
    assert breakdown["snapshot_restore"]["mean_steps_to_exit"] == 192
    assert breakdown["snapshot_restore"]["mean_kills"] == 0.0
    assert breakdown["snapshot_restore"]["exit_route_attempt_steps"] == 4
    assert breakdown["snapshot_restore"]["exit_route_reached_steps"] == 1
    assert breakdown["snapshot_restore"]["exit_route_progress_units"] == 128.0
    assert breakdown["reset_episode"]["level_completion_rate"] == 0.0
    assert breakdown["reset_episode"]["mean_kills"] == 1.0
    assert breakdown["reset_episode"]["route_failed_steps"] == 2


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


def test_checkpoint_curriculum_eval_restores_snapshot_stages(monkeypatch, tmp_path):
    seen_configs = []
    seen_trace_paths = []
    seen_eval_seeds = []

    async def fake_evaluate_checkpoint(checkpoint_path, env_config, **kwargs):
        env = SimpleNamespace(config=env_config)
        kwargs["before_reset"](env, 0)
        seen_configs.append(env.config)
        seen_trace_paths.append(kwargs.get("trace_path"))
        seen_eval_seeds.append(kwargs.get("seed"))
        return PolicyEval(
            result=EvaluationResult(
                policy_id=f"ppo:{checkpoint_path}",
                level_completion_rate=0.0,
                mean_kills=0.0,
                survival_rate=1.0,
                mean_steps_to_exit=16,
                mean_stuck_events=0.0,
                episode_count=1,
                mean_reward=0.0,
            ),
            episodes=[
                EpisodeEval(
                    seed=7,
                    total_reward=0.0,
                    level_completed=False,
                    death=False,
                    max_kills=1,
                    min_health=100,
                    steps=16,
                    steps_to_exit=16,
                    stuck_events=0,
                    done_reason="max_steps",
                    start_kills=1,
                    kill_delta=0,
                    max_kill_gain=0,
                )
            ],
        )

    monkeypatch.setattr(
        "restfuldoom_agent.ppo_agent.evaluate_checkpoint",
        fake_evaluate_checkpoint,
    )
    args = SimpleNamespace(
        endpoint="127.0.0.1:50051",
        token=None,
        agent_port=50051,
        tls=False,
        authority=None,
        skill=2,
        episode=1,
        map=1,
        seed=7,
        run_id="eval-snapshot-test",
        goal_preset="combat",
        target_x_fp=None,
        target_y_fp=None,
        max_steps=700,
        checkpoint_eval_max_steps=16,
        checkpoint_eval_episodes=1,
        checkpoint_eval_sample=False,
        eval_trace_jsonl=tmp_path / "eval-trace.jsonl",
        device="cpu",
        level_complete_bonus=100.0,
        kill_goal_bonus=10.0,
        required_kills=1,
        memory_path=tmp_path / "memory.json",
        reset_timeout_seconds=5.0,
        reset_attempts=2,
        reset_start_angle_degrees=None,
        reset_start_health=None,
        reset_start_armor=None,
        reset_start_ammo_bullets=None,
        reset_start_face_nearest_enemy=False,
        reset_warmup_steps=0,
        reset_warmup_max_tics=0,
        reset_warmup_until_visible=False,
        reset_warmup_until_shootable=False,
        first_visible_bonus=0.0,
        first_shootable_bonus=0.0,
        visible_contact_progress_reward=0.0,
        terminate_on_first_visible=False,
        terminate_on_first_shootable=False,
        snapshot_restore_command=None,
        snapshot_restore_cwd=None,
        snapshot_restore_timeout_seconds=60.0,
        snapshot_verify_restored_state=True,
        snapshot_verify_tick_tolerance=35,
        snapshot_verify_stream_tick=False,
        snapshot_verify_position_tolerance_fp=160 * 65536,
    )
    curriculum = {
        "schema": "restfuldoom.ppo_curriculum.v1",
        "name": "snapshot-eval-test",
        "mode": "fixed",
        "start_index": 0,
        "stages": [
            {
                "index": 0,
                "name": "first_kill_snapshot",
                "reset_mode": "snapshot",
                "expected_state": {"kills": 1},
                "snapshot": {"id": "slot-3", "slot": 3, "ref": "save_slot:3"},
            }
        ],
    }

    payload = asyncio.run(
        _evaluate_checkpoint_curriculum(
            tmp_path / "candidate.pt",
            args,
            curriculum=curriculum,
            update_index=2,
        )
    )

    assert payload["schema"] == "restfuldoom.ppo_checkpoint_curriculum_eval.v1"
    assert payload["score_schema"] == "restfuldoom.ppo_checkpoint_eval_score.v5"
    assert seen_configs
    assert seen_trace_paths[0] == (
        tmp_path / "eval-trace-update0002-stage00-first_kill_snapshot.jsonl"
    )
    config = seen_configs[0]
    assert config.reset_mode == "snapshot"
    assert config.snapshot == {"id": "slot-3", "slot": 3, "ref": "save_slot:3"}
    assert config.curriculum_stage["snapshot_restore"]["api_method"] == "grpc_load_snapshot"
    assert config.curriculum_stage["snapshot_restore"]["slot"] == 3
    assert payload["stages"][0]["selection_score_components"]["mode"] == "standard"
    assert payload["stages"][0]["result"]["result"]["mean_kills"] == 0.0

    post_combat_curriculum = {
        **curriculum,
        "stages": [
            {
                "index": 0,
                "name": "1420-post-combat_snapshot",
                "reset_mode": "snapshot",
                "expected_state": {"kills": 2},
                "snapshot": {"id": "slot-5", "slot": 5, "ref": "save_slot:5"},
                "evidence": {
                    "selector": "post-combat",
                    "selectors": ["post-combat"],
                },
            }
        ],
    }
    post_combat_payload = asyncio.run(
        _evaluate_checkpoint_curriculum(
            tmp_path / "candidate.pt",
            args,
            curriculum=post_combat_curriculum,
            update_index=3,
        )
    )

    assert (
        post_combat_payload["stages"][0]["selection_score_components"]["mode"]
        == "post_combat_exit_routing"
    )
    assert (
        post_combat_payload["stages"][0]["selection_score_components"][
            "earned_kill_bonus"
        ]
        == 0.0
    )
    assert seen_trace_paths[1] == (
        tmp_path / "eval-trace-update0003-stage00-1420-post-combat_snapshot.jsonl"
    )

    true_spawn_curriculum = {
        **curriculum,
        "stages": [
            {
                "index": 0,
                "name": "fresh_spawn_true_spawn_gate",
                "reset_mode": "episode",
                "reset_start": {},
                "evidence": {
                    "true_spawn_promotion_stage": True,
                    "snapshot_allowed": False,
                    "forced_skill_allowed": False,
                    "reset_source_required": "episode",
                },
            }
        ],
    }
    asyncio.run(
        _evaluate_checkpoint_curriculum(
            tmp_path / "candidate.pt",
            args,
            curriculum=true_spawn_curriculum,
            update_index=4,
        )
    )

    assert seen_eval_seeds == [2007, 3007, 7]


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


def test_curriculum_eval_best_prefers_stronger_eval_protocol():
    weak_eval_best = {
        "checkpoint_selection_score": 500.0,
        "checkpoint_selection_source": "checkpoint_curriculum_eval",
        "checkpoint_eval": {
            "selection_score": 500.0,
            "stage_count": 3,
            "episodes_per_stage": 1,
            "max_steps": 640,
            "sample": False,
        },
    }
    stronger_eval = {
        "selection_score": 450.0,
        "stage_count": 3,
        "episodes_per_stage": 3,
        "max_steps": 640,
        "sample": False,
    }
    stronger_but_regressed_eval = {
        "selection_score": 300.0,
        "stage_count": 3,
        "episodes_per_stage": 3,
        "max_steps": 640,
        "sample": False,
    }
    weaker_eval = {
        "selection_score": 900.0,
        "stage_count": 3,
        "episodes_per_stage": 1,
        "max_steps": 320,
        "sample": False,
    }

    assert _should_replace_best_checkpoint(
        weak_eval_best,
        score=450.0,
        score_source="checkpoint_curriculum_eval",
        checkpoint_eval=stronger_eval,
    )
    assert not _should_replace_best_checkpoint(
        weak_eval_best,
        score=300.0,
        score_source="checkpoint_curriculum_eval",
        checkpoint_eval=stronger_but_regressed_eval,
    )
    assert not _should_replace_best_checkpoint(
        weak_eval_best,
        score=900.0,
        score_source="checkpoint_curriculum_eval",
        checkpoint_eval=weaker_eval,
    )


def test_curriculum_eval_best_prefers_current_score_schema():
    stale_eval_best = {
        "checkpoint_selection_score": 630.0,
        "checkpoint_selection_source": "checkpoint_curriculum_eval",
        "checkpoint_eval": {
            "selection_score": 630.0,
            "score_schema": "restfuldoom.ppo_checkpoint_eval_score.v3",
            "stage_count": 1,
            "episodes_per_stage": 1,
            "max_steps": 4000,
            "sample": False,
        },
    }
    current_schema_eval = {
        "selection_score": 536.0,
        "score_schema": "restfuldoom.ppo_checkpoint_eval_score.v5",
        "stage_count": 1,
        "episodes_per_stage": 1,
        "max_steps": 4000,
        "sample": False,
    }
    current_schema_regression = {
        **current_schema_eval,
        "selection_score": 400.0,
    }

    assert _should_replace_best_checkpoint(
        stale_eval_best,
        score=536.0,
        score_source="checkpoint_curriculum_eval",
        checkpoint_eval=current_schema_eval,
    )
    assert not _should_replace_best_checkpoint(
        stale_eval_best,
        score=400.0,
        score_source="checkpoint_curriculum_eval",
        checkpoint_eval=current_schema_regression,
    )


def test_current_score_schema_does_not_outrank_stronger_eval_protocol():
    stronger_stale_best = {
        "checkpoint_selection_score": 630.0,
        "checkpoint_selection_source": "checkpoint_curriculum_eval",
        "checkpoint_eval": {
            "selection_score": 630.0,
            "score_schema": "restfuldoom.ppo_checkpoint_eval_score.v3",
            "stage_count": 3,
            "episodes_per_stage": 3,
            "max_steps": 4000,
            "sample": False,
        },
    }
    weaker_current_schema_eval = {
        "selection_score": 620.0,
        "score_schema": "restfuldoom.ppo_checkpoint_eval_score.v5",
        "stage_count": 1,
        "episodes_per_stage": 1,
        "max_steps": 4000,
        "sample": False,
    }

    assert not _should_replace_best_checkpoint(
        stronger_stale_best,
        score=620.0,
        score_source="checkpoint_curriculum_eval",
        checkpoint_eval=weaker_current_schema_eval,
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


def test_failed_true_spawn_gate_is_not_best_checkpoint(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    failed_true_spawn_eval = {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "selection_score": 624.0,
        "stages": [
            {
                "selection_score_components": {
                    "mode": "true_spawn_promotion",
                    "gate_ok": False,
                    "gate_passed_episodes": 0,
                    "true_spawn_gate": {"ok": False},
                },
            }
        ],
    }

    _record_ppo_checkpoint(
        memory,
        tmp_path / "failed-true-spawn.pt",
        goal_preset="exit_seeking",
        reward_config={"goal_preset": "exit_seeking"},
        metrics={"policy_loss": 0.1},
        rollout_summary={"checkpoint_selection_score": 10.0},
        update_index=0,
        buffer_path=tmp_path / "failed.jsonl",
        curriculum_stage={"name": "fresh_spawn_true_spawn_gate"},
        checkpoint_eval=failed_true_spawn_eval,
    )

    assert memory.data["ppo_policy"]["checkpoint_path"].endswith(
        "failed-true-spawn.pt"
    )
    assert "ppo_best_checkpoint" not in memory.data


def test_checkpoint_eval_training_guard_stops_failed_true_spawn_gate():
    checkpoint_eval = {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "selection_score": 416.0,
        "stages": [
            {
                "stage": {"name": "fresh_spawn_true_spawn_gate"},
                "selection_score_components": {
                    "mode": "true_spawn_promotion",
                    "gate_ok": False,
                    "gate_passed_episodes": 0,
                    "gate_episode_count": 1,
                    "true_spawn_gate": {
                        "ok": False,
                        "summary": {
                            "bottleneck_counts": {"combat": 1},
                            "done_reasons": {"death": 1},
                        },
                    },
                },
            }
        ],
    }

    guard = _checkpoint_eval_training_guard(
        SimpleNamespace(stop_on_true_spawn_regression=True),
        checkpoint_eval,
    )

    assert guard == {
        "schema": "restfuldoom.ppo_training_guard.v1",
        "reason": "true_spawn_gate_regression",
        "stop_training": True,
        "stage_name": "fresh_spawn_true_spawn_gate",
        "gate_ok": False,
        "gate_passed_episodes": 0,
        "gate_episode_count": 1,
        "gate_bottleneck_counts": {"combat": 1},
        "done_reasons": {"death": 1},
    }


def test_checkpoint_eval_training_guard_allows_passing_true_spawn_gate():
    checkpoint_eval = {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "stages": [
            {
                "stage": {"name": "fresh_spawn_true_spawn_gate"},
                "selection_score_components": {
                    "mode": "true_spawn_promotion",
                    "gate_ok": True,
                    "gate_passed_episodes": 1,
                    "gate_episode_count": 1,
                    "true_spawn_gate": {"ok": True},
                },
            }
        ],
    }

    assert (
        _checkpoint_eval_training_guard(
            SimpleNamespace(stop_on_true_spawn_regression=True),
            checkpoint_eval,
        )
        is None
    )


def test_checkpoint_eval_training_guard_is_opt_in():
    checkpoint_eval = {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "stages": [
            {
                "selection_score_components": {
                    "mode": "true_spawn_promotion",
                    "true_spawn_gate": {"ok": False},
                },
            }
        ],
    }

    assert (
        _checkpoint_eval_training_guard(
            SimpleNamespace(stop_on_true_spawn_regression=False),
            checkpoint_eval,
        )
        is None
    )


def test_passing_true_spawn_gate_can_be_best_checkpoint(tmp_path):
    memory = AgentMemory.load(tmp_path / "memory.json")
    passing_true_spawn_eval = {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
        "selection_score": 1000.0,
        "stages": [
            {
                "selection_score_components": {
                    "mode": "true_spawn_promotion",
                    "gate_ok": True,
                    "gate_passed_episodes": 1,
                    "true_spawn_gate": {"ok": True},
                },
            }
        ],
    }

    _record_ppo_checkpoint(
        memory,
        tmp_path / "passing-true-spawn.pt",
        goal_preset="exit_seeking",
        reward_config={"goal_preset": "exit_seeking"},
        metrics={"policy_loss": 0.1},
        rollout_summary={"checkpoint_selection_score": 10.0},
        update_index=0,
        buffer_path=tmp_path / "passing.jsonl",
        curriculum_stage={"name": "fresh_spawn_true_spawn_gate"},
        checkpoint_eval=passing_true_spawn_eval,
    )

    assert memory.data["ppo_best_checkpoint"]["checkpoint_path"].endswith(
        "passing-true-spawn.pt"
    )


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
        config=PPOConfig(
            update_epochs=1,
            minibatch_size=4,
            rollout_steps=8,
            reference_kl_coef=0.25,
            aux_bc_coef=0.5,
            aux_bc_batch_size=3,
        ),
    )

    metrics = trainer.update(buffer)
    checkpoint = trainer.save_checkpoint(tmp_path / "ppo.pt")
    loaded = PPOTrainer.load_checkpoint(checkpoint)

    assert checkpoint.exists()
    assert metrics["value_loss"] >= 0
    assert loaded.update_index == trainer.update_index
    assert loaded.config.reference_kl_coef == 0.25
    assert loaded.config.aux_bc_coef == 0.5
    assert loaded.config.aux_bc_batch_size == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_update_reports_reference_kl_and_preserves_reference_model():
    buffer = RolloutBuffer()
    for index in range(8):
        buffer.add(
            obs=[float(index % 2), 1.0],
            action_mask=[True, True],
            action=index % 2,
            reward=1.0 if index % 2 else 0.0,
            done=index == 7,
            value=0.1,
            logprob=-0.69,
        )
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(
            update_epochs=2,
            minibatch_size=4,
            rollout_steps=8,
            reference_kl_coef=0.5,
        ),
    )
    reference_model = trainer.clone_reference_model()
    before = {
        name: parameter.detach().clone()
        for name, parameter in reference_model.state_dict().items()
    }

    metrics = trainer.update(buffer, reference_model=reference_model)

    assert "reference_kl" in metrics
    assert metrics["reference_kl"] >= 0.0
    assert trainer.update_index == 1
    for name, parameter in reference_model.state_dict().items():
        assert parameter.detach().equal(before[name])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_update_reports_aux_behavior_clone_metrics():
    buffer = RolloutBuffer()
    for index in range(8):
        buffer.add(
            obs=[float(index % 2), 1.0],
            action_mask=[True, True],
            action=index % 2,
            reward=1.0,
            done=index == 7,
            value=0.1,
            logprob=-0.69,
        )
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(
            update_epochs=1,
            minibatch_size=4,
            rollout_steps=8,
            aux_bc_coef=0.75,
            aux_bc_batch_size=4,
        ),
    )
    samples = [([1.0, 0.0], 0) for _ in range(4)] + [
        ([0.0, 1.0], 1) for _ in range(4)
    ]

    metrics = trainer.update(buffer, behavior_clone_samples=samples)

    assert metrics["aux_bc_loss"] > 0.0
    assert 0.0 <= metrics["aux_bc_accuracy"] <= 1.0
    assert trainer.update_index == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_aux_behavior_clone_applies_action_masks():
    buffer = RolloutBuffer()
    for index in range(8):
        buffer.add(
            obs=[float(index % 2), 1.0],
            action_mask=[True, True],
            action=index % 2,
            reward=1.0,
            done=index == 7,
            value=0.1,
            logprob=-0.69,
        )
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(
            update_epochs=1,
            minibatch_size=4,
            rollout_steps=8,
            aux_bc_coef=1.0,
            aux_bc_batch_size=4,
        ),
    )
    samples = [([0.0, 1.0], 1, [False, True]) for _ in range(8)]

    metrics = trainer.update(buffer, behavior_clone_samples=samples)

    assert metrics["aux_bc_loss"] == pytest.approx(0.0)
    assert metrics["aux_bc_accuracy"] == pytest.approx(1.0)


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
def test_ppo_checkpoint_expands_appended_action_head(tmp_path):
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8),
    )
    checkpoint = trainer.save_checkpoint(tmp_path / "old-actions.pt")

    loaded = PPOTrainer.load_checkpoint(
        checkpoint,
        target_obs_dim=2,
        target_action_dim=3,
    )

    action, _logprob, _value = loaded.model.act([0.0, 1.0], action_mask=[False, False, True])
    assert action == 2
    assert loaded.action_dim == 3
    assert loaded.resume_migration is not None
    assert loaded.resume_migration["from_action_dim"] == 2
    assert loaded.resume_migration["to_action_dim"] == 3
    assert loaded.resume_migration["action_expanded"] is True
    assert not loaded.resume_migration["optimizer_state_loaded"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_checkpoint_rejects_reordered_action_prefix(tmp_path):
    torch = pytest.importorskip("torch")
    trainer = PPOTrainer(
        obs_dim=2,
        action_dim=2,
        config=PPOConfig(update_epochs=1, minibatch_size=4, rollout_steps=8),
    )
    checkpoint = trainer.save_checkpoint(tmp_path / "reordered-actions.pt")
    payload = torch.load(checkpoint, map_location="cpu")
    actions = list(payload["action_schema"]["actions"])
    actions[0], actions[1] = actions[1], actions[0]
    payload["action_schema"] = dict(payload["action_schema"])
    payload["action_schema"]["actions"] = actions
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="not a prefix"):
        PPOTrainer.load_checkpoint(checkpoint)


def test_evaluate_checkpoint_loads_checkpoint_against_current_schema(monkeypatch):
    captured = {}

    class StopEval(Exception):
        pass

    def fake_load_checkpoint(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        raise StopEval

    monkeypatch.setattr(PPOTrainer, "load_checkpoint", fake_load_checkpoint)

    with pytest.raises(StopEval):
        asyncio.run(evaluate_checkpoint("old.pt", object()))

    assert captured["path"] == "old.pt"
    assert captured["target_obs_dim"] == len(OBSERVATION_SCHEMA["feature_names"])
    assert captured["target_action_dim"] == len(ACTION_SCHEMA["actions"])


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
