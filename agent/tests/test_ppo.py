import json
from types import SimpleNamespace

import pytest

from restfuldoom_agent.curriculum import build_curriculum, stage_for_update
from restfuldoom_agent.ppo_agent import (
    _annotate_buffer_curriculum,
    _reset_start_from_trajectory,
    _summarize_buffer,
)
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
    assert rows[0]["memory_contract"]["memory_schema"] == "restfuldoom.agent_memory.v1"
    assert rows[1]["action"] == 1
    assert rows[1]["action_mask"] == [False, True]
    assert rows[1]["info"]["skill"] == "fire"


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
    assert ACTION_SCHEMA["representation"]["learned_now"] == (
        "PPO learns when to choose each option"
    )
    assert ACTION_SCHEMA["skill_definition_contract"]["storage"].startswith("Python schema")
    assert DECISION_CYCLE_SCHEMA["schema"] == "restfuldoom.decision_cycle.v1"
    assert "controller_input" in DECISION_CYCLE_SCHEMA["controller_decision_interface"]
    assert "rollout_record.action_mask" in DECISION_CYCLE_SCHEMA["trace_fields"]
    assert "rollout_record.info.route_outcome" in DECISION_CYCLE_SCHEMA["trace_fields"]
    assert MEMORY_CONTRACT["memory_schema"] == "restfuldoom.agent_memory.v1"
    assert any(
        phase["phase"] == "learn" and "ppo_checkpoints" in phase["writes"]
        for phase in MEMORY_CONTRACT["query_update_lifecycle"]
    )
    assert any(path["method"].startswith("AgentMemory.remembered_enemies") for path in MEMORY_CONTRACT["query_paths"])
    assert any(group["name"] == "memory_queries" for group in OBSERVATION_SCHEMA["source_groups"])
    assert "sector_damaging" in OBSERVATION_SCHEMA["feature_names"]
    assert "route_waypoint_distance_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "prev_route_progress_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "failed_route_attempt_count_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "enemy_distance_delta_norm" in OBSERVATION_SCHEMA["feature_names"]
    assert "recent_route_failure_ratio" in OBSERVATION_SCHEMA["feature_names"]
    assert "temporal_context" in {
        group["name"] for group in OBSERVATION_SCHEMA["source_groups"]
    }
    assert "no compact topological map graph" in OBSERVATION_SCHEMA["learning_readiness"]["known_gaps"]
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
