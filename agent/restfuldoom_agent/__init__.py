"""Agent-side helpers for RESTful Doom gRPC mode."""

from .reward import Goal, RewardEngine, TransitionReward

__all__ = [
    "DoomAgentClient",
    "Goal",
    "RewardEngine",
    "TransitionReward",
    "semantic_action",
]


def __getattr__(name):
    if name in {"DoomAgentClient", "semantic_action"}:
        from .client import DoomAgentClient, semantic_action

        return {
            "DoomAgentClient": DoomAgentClient,
            "semantic_action": semantic_action,
        }[name]
    raise AttributeError(name)
