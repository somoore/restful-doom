import json

from restfuldoom_agent.post_combat_exit_gate import (
    DEFAULT_ALLOWED_SKILLS,
    main as gate_main,
    validate_post_combat_exit_gate,
)


def test_post_combat_exit_gate_accepts_strict_restored_transition():
    report = validate_post_combat_exit_gate(_payload([_episode()]))

    assert report["ok"] is True
    assert report["summary"]["episode_count"] == 1
    assert report["summary"]["passed_episodes"] == 1
    assert report["summary"]["level_transition_episode_count"] == 1
    assert report["summary"]["invalid_action_steps"] == 0
    assert report["summary"]["selected_disallowed_steps"] == 0
    assert report["summary"]["strict_allowed_skill_fallback_steps"] == 0
    assert report["summary"]["snapshot_verification_failures"] == 0
    assert report["summary"]["exit_route_attempt_steps"] == 5
    assert report["summary"]["configured_allowed_skills"] == sorted(DEFAULT_ALLOWED_SKILLS)


def test_post_combat_exit_gate_rejects_non_transition_or_illegal_semantics():
    bad = _episode(
        start_kills=4,
        level_completed=False,
        done_reason="max_steps",
        level_transition_delta=0,
        invalid_action_steps=1,
        selected_disallowed_steps=1,
        action_mask_fallback_steps=1,
        strict_allowed_skill_fallback_steps=1,
        snapshot_verification_failures=1,
        skill_counts={"route_progression": 2, "fire": 1},
        exit_route_attempt_steps=0,
    )

    report = validate_post_combat_exit_gate(_payload([bad]))
    reasons = {failure["reason"] for failure in report["failures"]}

    assert report["ok"] is False
    assert "post_combat_start_kills_below_threshold" in reasons
    assert "missing_level_transition" in reasons
    assert "missing_level_complete_done" in reasons
    assert "invalid_action_steps_present" in reasons
    assert "selected_disallowed_steps_present" in reasons
    assert "action_mask_fallback_steps_present" in reasons
    assert "strict_allowed_skill_fallback_steps_present" in reasons
    assert "snapshot_verification_failures_present" in reasons
    assert "exit_route_attempt_steps_below_threshold" in reasons
    assert "disallowed_skill_executed" in reasons
    assert report["summary"]["passed_episodes"] == 0
    assert report["summary"]["disallowed_skill_counts"] == {"fire": 1}


def test_post_combat_exit_gate_requires_strict_filter_evidence():
    missing_strict = _payload(
        [
            _episode(
                allowed_skill_filter_steps=0,
                strict_allowed_skill_filter_steps=0,
            )
        ],
        strict_allowed_skills=False,
    )

    strict_report = validate_post_combat_exit_gate(missing_strict)
    relaxed_report = validate_post_combat_exit_gate(
        missing_strict,
        require_strict_skill_filter=False,
    )

    assert strict_report["ok"] is False
    assert {
        failure["reason"] for failure in strict_report["failures"]
    } == {"allowed_skill_filter_missing", "strict_allowed_skill_filter_missing"}
    assert relaxed_report["ok"] is True


def test_post_combat_exit_gate_cli_exits_nonzero_on_failure(tmp_path, capsys):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_payload([_episode(level_transition_delta=0)])))

    exit_code = gate_main([str(path)])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 1
    assert report["ok"] is False
    assert any(
        failure["reason"] == "missing_level_transition"
        for failure in report["failures"]
    )


def _payload(
    episodes,
    *,
    allowed_skills=DEFAULT_ALLOWED_SKILLS,
    strict_allowed_skills=True,
):
    return {
        "schema": "restfuldoom.ppo_checkpoint_curriculum_eval_run.v1",
        "checkpoint_path": "agent_models/ppo/post-combat.pt",
        "checkpoint_eval": {
            "schema": "restfuldoom.ppo_checkpoint_curriculum_eval.v1",
            "allowed_skills": list(allowed_skills),
            "strict_allowed_skills": strict_allowed_skills,
            "stages": [
                {
                    "stage": {
                        "name": "0003-post-combat-exit-route_snapshot",
                        "evidence": {
                            "selector": "post-combat-exit-route",
                            "selectors": ["post-combat", "post-combat-exit-route"],
                        },
                    },
                    "result": {
                        "result": {
                            "policy_id": "ppo:post-combat.pt",
                            "level_completion_rate": 1.0,
                            "snapshot_verification_failures": 0,
                            "invalid_action_steps": 0,
                            "selected_disallowed_steps": 0,
                            "strict_allowed_skill_fallback_steps": 0,
                        },
                        "episodes": episodes,
                    },
                }
            ],
        },
    }


def _episode(**overrides):
    episode = {
        "seed": 7,
        "total_reward": 30.0,
        "level_completed": True,
        "death": False,
        "max_kills": 5,
        "min_health": 72,
        "steps": 64,
        "steps_to_exit": 128,
        "stuck_events": 0,
        "done_reason": "level_complete",
        "start_kills": 5,
        "kill_delta": 0,
        "max_kill_gain": 0,
        "start_episode": 1,
        "start_map": 1,
        "end_episode": 1,
        "end_map": 2,
        "level_transition_delta": 1,
        "reset_source": "snapshot_restore",
        "skill_counts": {"route_progression": 4, "press_exit": 1},
        "invalid_action_steps": 0,
        "selected_disallowed_steps": 0,
        "action_mask_fallback_steps": 0,
        "allowed_skill_filter_steps": 5,
        "allowed_skill_filter_fallback_steps": 0,
        "strict_allowed_skill_filter_steps": 5,
        "strict_allowed_skill_fallback_steps": 0,
        "snapshot_verification_failures": 0,
        "exit_route_attempt_steps": 5,
        "exit_route_reached_steps": 1,
        "exit_route_failed_steps": 0,
        "exit_route_progress_units": 96.0,
    }
    episode.update(overrides)
    return episode
