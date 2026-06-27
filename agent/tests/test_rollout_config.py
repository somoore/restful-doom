from pathlib import Path

import pytest

from restfuldoom_agent.rollout_config import RolloutConfig, safe_endpoint_host


def test_rollout_config_binds_goal_and_metadata():
    config = RolloutConfig.from_mapping(
        {
            "endpoint": "hellbox.example:50051",
            "token": "opaque-token",
            "agent_port": 50051,
            "goal_preset": "navigation",
            "mission": "reach the marked exit without dying",
            "target_x_fp": 12345,
            "target_y_fp": 67890,
            "max_states": 500,
            "trajectory_jsonl": "trajectories/navigation.jsonl",
            "policy": "cycle",
            "capsule_name": "agent-doom",
            "capsule_id": "microvm-123",
            "auth_lease_id": "lease-123",
        }
    )

    goal = config.goal()
    metadata = config.to_metadata()

    assert goal.name == "navigation"
    assert goal.target_x_fp == 12345
    assert goal.target_y_fp == 67890
    assert config.max_states == 500
    assert config.trajectory_jsonl == Path("trajectories/navigation.jsonl")
    assert config.use_tls() is True
    assert metadata["endpoint_host"] == "hellbox.example:50051"
    assert metadata["token_present"] is True
    assert "token" not in metadata
    assert metadata["goal_preset"] == "navigation"
    assert metadata["mission"] == "reach the marked exit without dying"
    assert metadata["trajectory_jsonl"] == "trajectories/navigation.jsonl"
    assert metadata["capsule_name"] == "agent-doom"
    assert metadata["capsule_id"] == "microvm-123"
    assert metadata["auth_lease_id"] == "lease-123"


def test_rollout_config_cli_overrides_json_values(tmp_path):
    path = tmp_path / "rollout.json"
    path.write_text(
        '{"goal_preset":"survival","max_states":100,"reconnect":true}',
        encoding="utf-8",
    )

    config = RolloutConfig.from_json_file(path).with_overrides(
        goal_preset="combat",
        max_states=25,
        reconnect=False,
    )

    assert config.goal().name == "combat"
    assert config.max_states == 25
    assert config.reconnect is False


def test_rollout_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown rollout config field"):
        RolloutConfig.from_mapping({"goal_preset": "survival", "budget_tokens": 1000})


def test_rollout_config_rejects_malformed_types():
    with pytest.raises(ValueError, match="goal_preset must be a string"):
        RolloutConfig.from_mapping({"goal_preset": 7})

    with pytest.raises(ValueError, match="agent_port must be between"):
        RolloutConfig.from_mapping({"agent_port": 70000})


def test_safe_endpoint_host_strips_scheme_and_path():
    assert (
        safe_endpoint_host("grpcs://abc.lambda-microvm.us-east-2.on.aws:443/anything")
        == "abc.lambda-microvm.us-east-2.on.aws:443"
    )
