"""Agent-side helpers for RESTful Doom gRPC mode."""

from .reward import Goal, RewardEngine, TransitionReward
from .rollout_config import RolloutConfig

__all__ = [
    "DoomAgentClient",
    "Goal",
    "RewardEngine",
    "RolloutConfig",
    "SnapshotCommand",
    "TransitionReward",
    "semantic_action",
]


def __getattr__(name):
    if name in {"DoomAgentClient", "SnapshotCommand", "semantic_action"}:
        from .client import DoomAgentClient, SnapshotCommand, semantic_action

        return {
            "DoomAgentClient": DoomAgentClient,
            "SnapshotCommand": SnapshotCommand,
            "semantic_action": semantic_action,
        }[name]
    raise AttributeError(name)
