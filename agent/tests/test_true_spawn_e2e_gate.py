import json

from restfuldoom_agent.schemas import PPO_SKILL_ACTIONS
from restfuldoom_agent.true_spawn_e2e_gate import (
    DEFAULT_ALLOWED_SKILLS,
    DEFAULT_MIN_KILL_GAIN,
    main as gate_main,
    validate_true_spawn_e2e_gate,
)


def test_true_spawn_e2e_gate_accepts_strict_episode_transition():
    report = validate_true_spawn_e2e_gate(_payload([_episode()]))

    assert report["ok"] is True
    assert report["summary"]["episode_count"] == 1
    assert report["summary"]["passed_episodes"] == 1
    assert report["summary"]["true_spawn_episode_count"] == 1
    assert report["summary"]["level_transition_episode_count"] == 1
    assert report["summary"]["level_complete_episode_count"] == 1
    assert report["summary"]["reset_source_counts"] == {"episode": 1}
    assert report["summary"]["max_kill_gain_total"] == DEFAULT_MIN_KILL_GAIN
    assert report["summary"]["damage_delta_total"] == 42
    assert report["summary"]["route_attempt_steps"] == 8
    assert report["summary"]["exit_route_attempt_steps"] == 3
    assert report["summary"]["skill_counts"] == {
        "fire": 2,
        "press_exit": 1,
        "route_progression": 5,
    }


def test_true_spawn_e2e_gate_rejects_snapshot_or_warmup_reset():
    snapshot = _episode(reset_source="snapshot_restore")
    warmup = _episode(reset_source="warmup")

    report = validate_true_spawn_e2e_gate(_payload([snapshot, warmup]))
    reasons = {failure["reason"] for failure in report["failures"]}

    assert report["ok"] is False
    assert "not_true_spawn_episode_reset" in reasons
    assert "non_episode_reset_source_in_aggregate" in reasons
    assert report["summary"]["snapshot_reset_episode_count"] == 1
    assert report["summary"]["non_episode_reset_episode_count"] == 2


def test_true_spawn_e2e_gate_rejects_missing_chain_links_and_classifies_bottleneck():
    no_route = _episode(
        first_visible_contacts=0,
        first_shootable_contacts=0,
        visible_enemy_steps=0,
        route_attempt_steps=0,
        exit_route_attempt_steps=0,
        level_completed=False,
        done_reason="max_steps",
        level_transition_delta=0,
        max_kills=0,
        max_kill_gain=0,
        kill_delta=0,
        skill_counts={"seek_enemy": 5},
    )
    no_kill = _episode(
        max_kills=0,
        max_kill_gain=0,
        kill_delta=0,
        level_completed=False,
        done_reason="max_steps",
        level_transition_delta=0,
    )
    no_exit = _episode(
        exit_route_attempt_steps=0,
        level_completed=False,
        done_reason="max_steps",
        level_transition_delta=0,
    )

    report = validate_true_spawn_e2e_gate(_payload([no_route, no_kill, no_exit]))
    reasons = {failure["reason"] for failure in report["failures"]}

    assert report["ok"] is False
    assert "first_visible_contacts_below_threshold" in reasons
    assert "first_shootable_contacts_below_threshold" in reasons
    assert "kill_gain_below_threshold" in reasons
    assert "exit_route_attempt_steps_below_threshold" in reasons
    assert "missing_level_transition" in reasons
    assert report["summary"]["bottleneck_counts"] == {
        "combat": 1,
        "post_combat_route": 1,
        "spawn_route": 1,
    }


def test_true_spawn_e2e_gate_rejects_illegal_mask_or_snapshot_semantics():
    bad = _episode(
        start_kills=1,
        start_items=1,
        start_secrets=1,
        invalid_action_steps=1,
        selected_disallowed_steps=1,
        action_mask_fallback_steps=1,
        allowed_skill_filter_fallback_steps=1,
        strict_allowed_skill_fallback_steps=1,
        snapshot_verification_failures=1,
        skill_counts={"route_progression": 2, "unknown_skill": 1},
    )

    report = validate_true_spawn_e2e_gate(_payload([bad]))
    reasons = {failure["reason"] for failure in report["failures"]}

    assert report["ok"] is False
    assert "start_kills_not_fresh" in reasons
    assert "start_items_not_fresh" in reasons
    assert "start_secrets_not_fresh" in reasons
    assert "invalid_action_steps_present" in reasons
    assert "selected_disallowed_steps_present" in reasons
    assert "action_mask_fallback_steps_present" in reasons
    assert "allowed_skill_filter_fallback_steps_present" in reasons
    assert "strict_allowed_skill_fallback_steps_present" in reasons
    assert "snapshot_verification_failures_present" in reasons
    assert "disallowed_skill_executed" in reasons
    assert report["summary"]["disallowed_skill_counts"] == {"unknown_skill": 1}


def test_true_spawn_e2e_gate_requires_strict_filter_evidence():
    missing_strict = _payload(
        [
            _episode(
                allowed_skill_filter_steps=0,
                strict_allowed_skill_filter_steps=0,
            )
        ],
        strict_allowed_skills=False,
    )

    strict_report = validate_true_spawn_e2e_gate(missing_strict)
    relaxed_report = validate_true_spawn_e2e_gate(
        missing_strict,
        require_strict_skill_filter=False,
    )

    assert strict_report["ok"] is False
    assert {
        failure["reason"] for failure in strict_report["failures"]
    } == {"allowed_skill_filter_missing", "strict_allowed_skill_filter_missing"}
    assert relaxed_report["ok"] is True


def test_true_spawn_e2e_gate_allows_diagnostic_lower_kill_threshold():
    default_report = validate_true_spawn_e2e_gate(
        _payload([_episode(max_kills=1, kill_delta=1, max_kill_gain=1)])
    )
    relaxed_report = validate_true_spawn_e2e_gate(
        _payload([_episode(max_kills=1, kill_delta=1, max_kill_gain=1)]),
        min_kill_gain=1,
    )

    assert default_report["ok"] is False
    assert any(
        failure["reason"] == "kill_gain_below_threshold"
        for failure in default_report["failures"]
    )
    assert relaxed_report["ok"] is True


def test_true_spawn_e2e_gate_cli_exits_nonzero_on_failure(tmp_path, capsys):
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


def test_true_spawn_e2e_gate_default_allowed_skills_match_ppo_schema():
    assert DEFAULT_ALLOWED_SKILLS == tuple(PPO_SKILL_ACTIONS)


def test_true_spawn_e2e_gate_default_kill_threshold_matches_post_combat():
    assert DEFAULT_MIN_KILL_GAIN == 5


def _payload(
    episodes,
    *,
    allowed_skills=DEFAULT_ALLOWED_SKILLS,
    strict_allowed_skills=True,
):
    return {
        "schema": "restfuldoom.ppo_eval.v1",
        "checkpoint_path": "agent_models/ppo/current.pt",
        "candidate": {
            "result": {
                "policy_id": "ppo:current.pt",
                "level_completion_rate": 1.0,
                "snapshot_verification_failures": sum(
                    episode.get("snapshot_verification_failures", 0)
                    for episode in episodes
                ),
                "invalid_action_steps": sum(
                    episode.get("invalid_action_steps", 0) for episode in episodes
                ),
                "selected_disallowed_steps": sum(
                    episode.get("selected_disallowed_steps", 0)
                    for episode in episodes
                ),
                "strict_allowed_skill_fallback_steps": sum(
                    episode.get("strict_allowed_skill_fallback_steps", 0)
                    for episode in episodes
                ),
                "reset_source_breakdown": {
                    source: {"episode_count": count}
                    for source, count in _reset_counts(episodes).items()
                },
            },
            "episodes": episodes,
            "allowed_skills": list(allowed_skills),
            "strict_allowed_skills": strict_allowed_skills,
        },
    }


def _reset_counts(episodes):
    counts = {}
    for episode in episodes:
        source = episode.get("reset_source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _episode(**overrides):
    episode = {
        "seed": 7,
        "total_reward": 30.0,
        "level_completed": True,
        "death": False,
        "max_kills": DEFAULT_MIN_KILL_GAIN,
        "min_health": 72,
        "steps": 128,
        "steps_to_exit": 128,
        "stuck_events": 0,
        "done_reason": "level_complete",
        "start_kills": 0,
        "kill_delta": DEFAULT_MIN_KILL_GAIN,
        "max_kill_gain": DEFAULT_MIN_KILL_GAIN,
        "max_items": 0,
        "start_items": 0,
        "item_delta": 0,
        "max_item_gain": 0,
        "max_secrets": 0,
        "start_secrets": 0,
        "secret_delta": 0,
        "max_secret_gain": 0,
        "start_episode": 1,
        "start_map": 1,
        "end_episode": 1,
        "end_map": 2,
        "level_transition_delta": 1,
        "reset_source": "episode",
        "skill_counts": {"route_progression": 5, "fire": 2, "press_exit": 1},
        "visible_enemy_steps": 12,
        "first_visible_contacts": 1,
        "first_shootable_contacts": 1,
        "shootable_target_steps": 4,
        "fire_on_shootable_steps": 2,
        "missed_shootable_fire_steps": 2,
        "damage_delta": 42,
        "invalid_action_steps": 0,
        "selected_disallowed_steps": 0,
        "action_mask_fallback_steps": 0,
        "allowed_skill_filter_steps": 8,
        "allowed_skill_filter_fallback_steps": 0,
        "strict_allowed_skill_filter_steps": 8,
        "strict_allowed_skill_fallback_steps": 0,
        "snapshot_verification_failures": 0,
        "route_action_reward": 1.0,
        "route_attempt_steps": 8,
        "route_reached_steps": 4,
        "route_failed_steps": 0,
        "route_progress_units": 320.0,
        "exit_route_attempt_steps": 3,
        "exit_route_reached_steps": 1,
        "exit_route_failed_steps": 0,
        "exit_route_progress_units": 128.0,
    }
    episode.update(overrides)
    return episode
