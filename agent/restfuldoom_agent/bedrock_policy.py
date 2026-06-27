"""Optional AWS Bedrock policy for high-level Doom actions."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from .client import agent_pb2, semantic_action, summarize_state


class BedrockPolicy:
    """Uses Bedrock text reasoning to choose Doom actions."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        timeout_seconds: float = 3.0,
        max_tokens: int = 80,
        mission: str | None = None,
        goal_metadata: dict[str, Any] | None = None,
        fallback_action: int | None = None,
        fallback_amount: int = 10,
        raise_on_error: bool = False,
    ) -> None:
        import boto3

        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
        )
        self.client = boto3.client("bedrock-runtime")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.mission = mission
        self.goal_metadata = dict(goal_metadata or {})
        self.fallback_action = fallback_action or agent_pb2.ACTION_FORWARD
        self.fallback_amount = fallback_amount
        self.raise_on_error = raise_on_error
        self.last_error: str | None = None
        self.error_count = 0
        self.fallback_count = 0
        self.last_latency_ms: float | None = None
        self.last_llm_latency_ms: float | None = None
        self.last_token_usage: dict[str, int] = {}
        self.total_token_usage: dict[str, int] = {}

    async def next_action(self, state: Any) -> Any:
        """Returns the next high-level Doom action."""
        started = time.perf_counter()
        try:
            action = await asyncio.wait_for(
                asyncio.to_thread(self._invoke_action, state),
                timeout=self.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.last_latency_ms = latency_ms
            self.last_llm_latency_ms = latency_ms
            self.last_error = None
            return action
        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.last_latency_ms = latency_ms
            self.last_llm_latency_ms = latency_ms
            self.error_count += 1
            self.last_error = str(error)
            if self.raise_on_error:
                raise
            self.fallback_count += 1
            return semantic_action(
                self.fallback_action,
                amount=self.fallback_amount,
                duration_tics=1,
            )

    def _invoke_action(self, state: Any) -> Any:
        prompt = (
            "You control Doom through one JSON action. "
            "Choose one of forward, backward, turn-left, turn-right, "
            "strafe-left, strafe-right, shoot, use, switch-weapon. "
            "Return only JSON like {\"action\":\"forward\",\"amount\":25}.\n"
            f"MISSION={self.mission or self.goal_metadata.get('goal_preset', 'survive')}\n"
            f"GOAL={json.dumps(_public_goal_metadata(self.goal_metadata), separators=(',', ':'))}\n"
            f"STATE={json.dumps(summarize_state(state), separators=(',', ':'))}"
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        self._record_usage(payload.get("usage"))
        text = payload["content"][0]["text"]
        return action_from_json(json.loads(text))

    def _record_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            self.last_token_usage = {}
            return
        normalized = {
            key: int(value)
            for key, value in usage.items()
            if isinstance(key, str) and isinstance(value, int)
        }
        self.last_token_usage = normalized
        for key, value in normalized.items():
            self.total_token_usage[key] = self.total_token_usage.get(key, 0) + value


def action_from_json(data: dict[str, Any]) -> Any:
    """Maps a Bedrock JSON decision to a protobuf action."""
    name = str(data.get("action", "")).replace("_", "-").lower()
    amount = int(data.get("amount", 10) or 10)
    mapping = {
        "forward": agent_pb2.ACTION_FORWARD,
        "backward": agent_pb2.ACTION_BACKWARD,
        "turn-left": agent_pb2.ACTION_TURN_LEFT,
        "turn-right": agent_pb2.ACTION_TURN_RIGHT,
        "strafe-left": agent_pb2.ACTION_STRAFE_LEFT,
        "strafe-right": agent_pb2.ACTION_STRAFE_RIGHT,
        "shoot": agent_pb2.ACTION_SHOOT,
        "use": agent_pb2.ACTION_USE,
        "switch-weapon": agent_pb2.ACTION_SWITCH_WEAPON,
    }
    return semantic_action(mapping.get(name, agent_pb2.ACTION_FORWARD), amount=amount, duration_tics=1)


def _public_goal_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "goal_preset",
        "mission",
        "target_x_fp",
        "target_y_fp",
        "max_states",
        "policy",
    }
    return {key: value for key, value in metadata.items() if key in allowed}
