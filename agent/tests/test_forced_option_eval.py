import json
from pathlib import Path

import pytest

from restfuldoom_agent.curriculum import build_curriculum
from restfuldoom_agent.env import SKILL_ACTIONS
from restfuldoom_agent.forced_option_eval import (
    ForcedOptionEvalConfig,
    _comparison,
    _config_from_args,
    _forced_option_stop_reason,
    _forced_summary,
    _selected_stages,
    _skill_action_index,
    _write_jsonl_record,
)
from restfuldoom_agent.ppo_agent import _summarize_buffer
from restfuldoom_agent.ppo import RolloutBuffer


def test_forced_option_eval_selects_requested_stages():
    curriculum = build_curriculum(
        name="e1m1-contact-to-combat",
        manual_reset_start={},
        mode="fixed",
        start_index=0,
        seed=0,
    )

    default_stage = _selected_stages(curriculum, ())
    selected = _selected_stages(curriculum, (1, 3))

    assert [stage["name"] for stage in default_stage] == ["visible_contact_fast"]
    assert [stage["name"] for stage in selected] == [
        "visible_contact_route",
        "combat_start",
    ]
    assert selected[0]["selected_index"] == 1


def test_forced_option_eval_rejects_unknown_stage_index():
    curriculum = build_curriculum(
        name="e1m1-contact-to-combat",
        manual_reset_start={},
        mode="fixed",
        start_index=0,
        seed=0,
    )

    with pytest.raises(ValueError, match="outside range"):
        _selected_stages(curriculum, (99,))


def test_forced_option_eval_summarizes_forced_rollout():
    buffer = RolloutBuffer()
    for index, (visible, shootable, allowed, handoff, skill) in enumerate(
        [
            (True, False, True, False, "close_visible_contact"),
            (False, False, True, False, "close_visible_contact"),
            (True, True, False, True, "fire"),
        ]
    ):
        buffer.add(
            obs=[0.0],
            action_mask=[False, True, False, False, False, False, False, False, allowed],
            action=1 if handoff else 8,
            reward=0.0,
            done=index == 2,
            value=0.0,
            logprob=0.0,
            info={
                "had_visible_enemy": visible,
                "had_shootable_target": shootable,
                "forced_action_allowed": allowed,
                "selected_action_allowed": True,
                "shootable_handoff_applied": handoff,
                "skill": skill,
                "decision": {
                    "skill": "contact_use_line" if index == 1 else skill,
                    "stuck": index == 2,
                },
                "learning_trace": {
                    "observation": {
                        "groups": {
                            "contact": {
                                "contact_use_line_close": index == 1,
                            },
                        },
                    },
                },
            },
        )

    summary = _forced_summary(buffer)

    assert summary["records"] == 3
    assert summary["forced_allowed_steps"] == 2
    assert summary["forced_disallowed_steps"] == 1
    assert summary["forced_action_allowed_steps"] == 2
    assert summary["forced_action_disallowed_steps"] == 1
    assert summary["selected_disallowed_steps"] == 0
    assert summary["selected_action_disallowed_steps"] == 0
    assert summary["shootable_handoff_steps"] == 1
    assert summary["forced_handoff_disallowed_steps"] == 1
    assert summary["unhandled_forced_disallowed_steps"] == 0
    assert summary["lost_visible_contact_steps"] == 1
    assert summary["first_shootable_step"] == 2
    assert summary["actual_skill_counts"] == {
        "close_visible_contact": 2,
        "fire": 1,
    }
    assert summary["decision_skill_counts"] == {
        "close_visible_contact": 1,
        "contact_use_line": 1,
        "fire": 1,
    }
    assert summary["first_contact_use_line_close_step"] == 1
    assert summary["stuck_steps"] == 1
    assert _summarize_buffer(buffer)["invalid_action_steps"] == 0


def test_forced_option_eval_jsonl_records_include_observations(tmp_path):
    buffer = RolloutBuffer()
    buffer.add(
        obs=[0.1, 0.2],
        action_mask=[False, True],
        action=1,
        reward=2.0,
        done=False,
        value=0.0,
        logprob=0.0,
        info={"selected_action_allowed": True},
    )
    config = ForcedOptionEvalConfig(jsonl_path=tmp_path / "forced.jsonl")

    _write_jsonl_record(
        config,
        {"name": "visible_contact_fast", "index": 1},
        "close_visible_contact",
        buffer.records[0],
    )

    row = json.loads(config.jsonl_path.read_text(encoding="utf-8"))
    assert row["schema"] == "restfuldoom.forced_option_eval_record.v1"
    assert row["record"]["obs"] == [0.1, 0.2]
    assert row["record"]["action"] == 1
    assert row["stage"]["name"] == "visible_contact_fast"


def test_forced_option_eval_classifies_close_contact_as_complete_when_fire_allowed():
    action_mask = [False for _ in range(len(SKILL_ACTIONS))]
    action_mask[_skill_action_index("fire")] = True

    assert (
        _forced_option_stop_reason("close_visible_contact", action_mask)
        == "forced_option_completed_shootable"
    )
    assert _forced_option_stop_reason("seek_enemy", action_mask) == "forced_option_disallowed"


def test_forced_option_eval_comparison_keeps_failure_visible():
    rows = _comparison(
        [
            {
                "ok": False,
                "forced_skill": "close_visible_contact",
                "stage": {"name": "first-visible"},
                "reset_error": "level_time drift",
            },
            {
                "ok": True,
                "forced_skill": "seek_enemy",
                "stage": {"name": "first-visible"},
                "termination_reason": "forced_option_disallowed",
                "termination_step": 7,
                "summary": {
                    "first_shootable_contacts": 1,
                    "shootable_target_steps": 4,
                    "damage_delta": 10,
                    "kill_delta": 0,
                    "enemy_distance_delta": 32.0,
                    "visible_contact_distance_delta": 128.0,
                    "contact_use_line_close_steps": 3,
                    "invalid_action_steps": 0,
                },
                "forced_summary": {
                    "lost_visible_contact_steps": 2,
                    "forced_disallowed_steps": 0,
                    "selected_disallowed_steps": 0,
                    "shootable_handoff_steps": 3,
                    "unhandled_forced_disallowed_steps": 0,
                    "stuck_steps": 1,
                    "recovery_steps": 0,
                },
            },
        ]
    )

    assert rows[0]["ok"] is False
    assert rows[0]["reset_error"] == "level_time drift"
    assert rows[1]["first_shootable_contacts"] == 1
    assert rows[1]["visible_contact_distance_delta"] == 128.0
    assert rows[1]["enemy_distance_delta"] == 32.0
    assert rows[1]["contact_use_line_close_steps"] == 3
    assert rows[1]["selected_disallowed_steps"] == 0
    assert rows[1]["shootable_handoff_steps"] == 3
    assert rows[1]["unhandled_forced_disallowed_steps"] == 0
    assert rows[1]["termination_reason"] == "forced_option_disallowed"
    assert rows[1]["termination_step"] == 7
    assert rows[1]["stuck_steps"] == 1


def test_forced_option_eval_arg_defaults():
    config = _config_from_args(
        type(
            "Args",
            (),
            {
                "endpoint": "127.0.0.1:50051",
                "token": None,
                "agent_port": 50051,
                "tls": False,
                "authority": None,
                "skill": 2,
                "episode": 1,
                "map": 1,
                "seed": 0,
                "run_id": "test",
                "goal_preset": "combat",
                "max_steps": 64,
                "macro_steps": 8,
                "memory_path": Path("memory.json"),
                "reset_timeout_seconds": 5.0,
                "reset_attempts": 2,
                "curriculum": "e1m1-contact-to-combat",
                "snapshot_curriculum": None,
                "curriculum_mode": "fixed",
                "curriculum_start_index": 0,
                "stage_index": [0, 2],
                "force_skill": ["close_visible_contact"],
                "first_shootable_bonus": 0.0,
                "visible_contact_progress_reward": 0.001,
                "terminate_on_first_visible": False,
                "terminate_on_first_shootable": False,
                "terminate_on_required_kills": True,
                "shootable_handoff_skill": "fire",
                "no_snapshot_verify_restored_state": False,
                "snapshot_verify_tick_tolerance": 35,
                "snapshot_verify_stream_tick": False,
                "snapshot_verify_position_tolerance_fp": 160 * 65536,
                "output": None,
                "jsonl": None,
            },
        )()
    )

    assert config.stage_indexes == (0, 2)
    assert config.forced_skills == ("close_visible_contact",)
    assert config.terminate_on_required_kills
    assert config.shootable_handoff_skill == "fire"
    assert _skill_action_index("close_visible_contact") == 8
