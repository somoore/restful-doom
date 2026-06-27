import json
from types import SimpleNamespace

import pytest

from restfuldoom_agent.ppo_agent import _summarize_buffer
from restfuldoom_agent.ppo import (
    PPOConfig,
    PPOTrainer,
    PromotionGate,
    EvaluationResult,
    RolloutBuffer,
    TORCH_AVAILABLE,
)
from restfuldoom_agent.ppo_eval import EpisodeEval, PolicyEval, decide_promotion
from restfuldoom_agent.schemas import map_expert_skill_to_ppo_action


def test_rollout_buffer_saves_jsonl(tmp_path):
    buffer = RolloutBuffer()
    buffer.add(
        obs=[0.0, 1.0],
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
    assert rows[1]["action"] == 1
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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_ppo_update_and_checkpoint_roundtrip(tmp_path):
    buffer = RolloutBuffer()
    for index in range(8):
        buffer.add(
            obs=[float(index % 2), 1.0],
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
