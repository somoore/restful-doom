"""Optional AWS Bedrock policy for high-level Doom actions."""

from __future__ import annotations

import json
import os
from typing import Any

from .client import agent_pb2, semantic_action, summarize_state


class BedrockPolicy:
    """Uses Bedrock text reasoning to choose Doom actions."""

    def __init__(self, model_id: str | None = None) -> None:
        import boto3

        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
        )
        self.client = boto3.client("bedrock-runtime")

    async def next_action(self, state: Any) -> Any:
        """Returns the next high-level Doom action."""
        prompt = (
            "You control Doom through one JSON action. "
            "Choose one of forward, backward, turn-left, turn-right, "
            "strafe-left, strafe-right, shoot, use. "
            "Return only JSON like {\"action\":\"forward\",\"amount\":25}.\n"
            f"STATE={json.dumps(summarize_state(state), separators=(',', ':'))}"
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 80,
            "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        text = payload["content"][0]["text"]
        return action_from_json(json.loads(text))


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
    }
    return semantic_action(mapping.get(name, agent_pb2.ACTION_FORWARD), amount=amount, duration_tics=1)
