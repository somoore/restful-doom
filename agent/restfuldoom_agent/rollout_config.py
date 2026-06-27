"""Rollout configuration for Doom agent demos."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .reward import goal_preset

DEFAULT_AGENT_PORT = 50051


_CONFIG_FIELDS = {
    "endpoint",
    "token",
    "agent_port",
    "tls",
    "authority",
    "goal_preset",
    "mission",
    "target_x_fp",
    "target_y_fp",
    "max_states",
    "trajectory_jsonl",
    "reconnect",
    "max_reconnects",
    "backoff_initial",
    "backoff_max",
    "policy",
    "bedrock_model_id",
    "bedrock_timeout",
    "bedrock_max_tokens",
    "run_id",
    "capsule_name",
    "capsule_id",
    "auth_lease_id",
}


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for one observe-act rollout."""

    endpoint: str = "127.0.0.1:50051"
    token: str | None = None
    agent_port: int = DEFAULT_AGENT_PORT
    tls: bool | None = None
    authority: str | None = None
    goal_preset: str = "custom"
    mission: str | None = None
    target_x_fp: int | None = None
    target_y_fp: int | None = None
    max_states: int | None = 35
    trajectory_jsonl: Path | None = None
    reconnect: bool = True
    max_reconnects: int = 5
    backoff_initial: float = 0.25
    backoff_max: float = 5.0
    policy: str = "cycle"
    bedrock_model_id: str | None = None
    bedrock_timeout: float = 3.0
    bedrock_max_tokens: int = 80
    run_id: str | None = None
    capsule_name: str | None = None
    capsule_id: str | None = None
    auth_lease_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str):
            raise ValueError("endpoint must be a string")
        if not self.endpoint:
            raise ValueError("endpoint must not be empty")
        if self.token is not None and not isinstance(self.token, str):
            raise ValueError("token must be a string or null")
        if self.token == "":
            raise ValueError("token must not be empty")
        if not _is_json_int(self.agent_port):
            raise ValueError("agent_port must be an integer")
        if self.agent_port < 1 or self.agent_port > 65535:
            raise ValueError("agent_port must be between 1 and 65535")
        if self.tls is not None and not isinstance(self.tls, bool):
            raise ValueError("tls must be true, false, or null")
        if self.authority is not None and not isinstance(self.authority, str):
            raise ValueError("authority must be a string or null")
        if not isinstance(self.goal_preset, str):
            raise ValueError("goal_preset must be a string")
        object.__setattr__(self, "goal_preset", self.goal_preset.replace("-", "_").lower())
        if self.mission is not None and not isinstance(self.mission, str):
            raise ValueError("mission must be a string or null")
        if self.target_x_fp is not None and not _is_json_int(self.target_x_fp):
            raise ValueError("target_x_fp must be an integer or null")
        if self.target_y_fp is not None and not _is_json_int(self.target_y_fp):
            raise ValueError("target_y_fp must be an integer or null")
        if self.max_states is not None and not _is_json_int(self.max_states):
            raise ValueError("max_states must be an integer or null")
        if self.max_states is not None and self.max_states <= 0:
            raise ValueError("max_states must be positive or null")
        if not isinstance(self.reconnect, bool):
            raise ValueError("reconnect must be true or false")
        if not _is_json_int(self.max_reconnects):
            raise ValueError("max_reconnects must be an integer")
        if self.max_reconnects < 0:
            raise ValueError("max_reconnects must be zero or greater")
        if not isinstance(self.backoff_initial, (int, float)):
            raise ValueError("backoff_initial must be a number")
        if self.backoff_initial <= 0:
            raise ValueError("backoff_initial must be positive")
        if not isinstance(self.backoff_max, (int, float)):
            raise ValueError("backoff_max must be a number")
        if self.backoff_max <= 0:
            raise ValueError("backoff_max must be positive")
        if not isinstance(self.policy, str):
            raise ValueError("policy must be a string")
        object.__setattr__(self, "policy", self.policy.replace("-", "_").lower())
        if self.policy not in {"cycle", "bedrock"}:
            raise ValueError("policy must be cycle or bedrock")
        if self.bedrock_model_id is not None and not isinstance(self.bedrock_model_id, str):
            raise ValueError("bedrock_model_id must be a string or null")
        if not isinstance(self.bedrock_timeout, (int, float)):
            raise ValueError("bedrock_timeout must be a number")
        if self.bedrock_timeout <= 0:
            raise ValueError("bedrock_timeout must be positive")
        if not _is_json_int(self.bedrock_max_tokens):
            raise ValueError("bedrock_max_tokens must be an integer")
        if self.bedrock_max_tokens <= 0:
            raise ValueError("bedrock_max_tokens must be positive")
        for field_name in ("run_id", "capsule_name", "capsule_id", "auth_lease_id"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or null")
        if self.goal_preset != "custom":
            goal_preset(
                self.goal_preset,
                target_x_fp=self.target_x_fp,
                target_y_fp=self.target_y_fp,
            )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RolloutConfig":
        """Loads rollout configuration from a JSON file."""
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{config_path} must contain a JSON object")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RolloutConfig":
        """Builds a config from a JSON-compatible mapping."""
        unknown = sorted(set(data) - _CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"unknown rollout config field(s): {', '.join(unknown)}")

        values = dict(data)
        if values.get("trajectory_jsonl") is not None:
            values["trajectory_jsonl"] = Path(values["trajectory_jsonl"])
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "RolloutConfig":
        """Returns a config with CLI overrides applied."""
        clean = {key: value for key, value in overrides.items() if value is not None}
        if clean.get("trajectory_jsonl") is not None:
            clean["trajectory_jsonl"] = Path(clean["trajectory_jsonl"])
        return replace(self, **clean)

    def goal(self) -> Any | None:
        """Returns the bound reward goal for this rollout."""
        if self.goal_preset == "custom":
            return None
        return goal_preset(
            self.goal_preset,
            target_x_fp=self.target_x_fp,
            target_y_fp=self.target_y_fp,
        )

    def backoff_config(self) -> Any:
        """Returns the reconnect backoff config without importing gRPC at module import time."""
        from .client import BackoffConfig

        return BackoffConfig(
            initial_seconds=self.backoff_initial,
            max_seconds=self.backoff_max,
            max_attempts=self.max_reconnects,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Returns JSON-safe config metadata for trajectory records."""
        return {
            "run_id": self.run_id,
            "capsule_name": self.capsule_name,
            "capsule_id": self.capsule_id,
            "auth_lease_id": self.auth_lease_id,
            "endpoint_host": safe_endpoint_host(self.endpoint),
            "token_present": self.token is not None,
            "agent_port": self.agent_port,
            "tls": self.use_tls(),
            "authority": self.authority,
            "policy": self.policy,
            "goal_preset": self.goal_preset,
            "mission": self.mission,
            "target_x_fp": self.target_x_fp,
            "target_y_fp": self.target_y_fp,
            "max_states": self.max_states,
            "trajectory_jsonl": (
                str(self.trajectory_jsonl) if self.trajectory_jsonl is not None else None
            ),
            "reconnect": self.reconnect,
            "max_reconnects": self.max_reconnects,
            "backoff_initial": self.backoff_initial,
            "backoff_max": self.backoff_max,
            "bedrock_model_id": self.bedrock_model_id,
            "bedrock_timeout": self.bedrock_timeout,
            "bedrock_max_tokens": self.bedrock_max_tokens,
        }

    def use_tls(self) -> bool:
        """Returns whether the rollout should use a TLS gRPC channel."""
        if self.tls is not None:
            return self.tls
        return self.token is not None and not _is_loopback_endpoint(self.endpoint)


def safe_endpoint_host(endpoint: str) -> str:
    """Returns a secret-safe endpoint host label for trajectory metadata."""
    parsed = urlsplit(endpoint if "://" in endpoint else f"grpc://{endpoint}")
    if parsed.hostname:
        if parsed.port is not None:
            return f"{parsed.hostname}:{parsed.port}"
        return parsed.hostname
    return (parsed.netloc or parsed.path.split("/")[0]).split("@")[-1]


def _is_loopback_endpoint(endpoint: str) -> bool:
    host = safe_endpoint_host(endpoint).lower()
    if host.startswith("["):
        host = host.split("]", 1)[0].strip("[]")
    else:
        host = host.split(":", 1)[0]
    return host in {"localhost", "127.0.0.1", "::1"}


def _is_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
