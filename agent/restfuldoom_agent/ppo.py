"""PPO training primitives for high-level Doom skill learning."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .env import DoomAgentEnv, EnvStep
from .schemas import ACTION_SCHEMA, DECISION_CYCLE_SCHEMA, MEMORY_CONTRACT, OBSERVATION_SCHEMA

PPO_CHECKPOINT_SCHEMA = "restfuldoom.ppo_checkpoint.v1"
ROLLOUT_BUFFER_SCHEMA = "restfuldoom.ppo_rollout.v1"

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs.
    torch = None
    nn = None
    Categorical = None
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for one PPO trainer."""

    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128
    hidden_size: int = 128
    rollout_steps: int = 512
    seed: int = 7


@dataclass
class RolloutRecord:
    """One PPO training transition."""

    obs: list[float]
    action_mask: list[bool]
    action: int
    reward: float
    done: bool
    value: float
    logprob: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBuffer:
    """Stores one PPO batch before advantage estimation."""

    records: list[RolloutRecord] = field(default_factory=list)
    last_value: float = 0.0

    def add(
        self,
        *,
        obs: list[float],
        action_mask: list[bool] | None = None,
        action: int,
        reward: float,
        done: bool,
        value: float,
        logprob: float,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Adds one transition."""
        self.records.append(
            RolloutRecord(
                obs=list(obs),
                action_mask=list(action_mask or []),
                action=int(action),
                reward=float(reward),
                done=bool(done),
                value=float(value),
                logprob=float(logprob),
                info=dict(info or {}),
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def save_jsonl(self, path: str | Path) -> Path:
        """Saves rollout records incrementally-readable JSONL."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema": ROLLOUT_BUFFER_SCHEMA,
                        "created_at": _iso_now(),
                        "count": len(self.records),
                        "observation_schema": OBSERVATION_SCHEMA,
                        "action_schema": ACTION_SCHEMA,
                        "decision_cycle_schema": DECISION_CYCLE_SCHEMA,
                        "memory_contract": MEMORY_CONTRACT,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for index, record in enumerate(self.records):
                payload = asdict(record)
                payload["index"] = index
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return output

    def tensors(self, *, config: PPOConfig, device: str = "cpu") -> dict[str, Any]:
        """Returns tensors with GAE advantages and returns."""
        require_torch()
        if not self.records:
            raise ValueError("cannot convert an empty rollout buffer")

        rewards = [record.reward for record in self.records]
        dones = [record.done for record in self.records]
        values = [record.value for record in self.records] + [self.last_value]
        advantages = [0.0 for _ in self.records]
        gae = 0.0
        for index in reversed(range(len(self.records))):
            nonterminal = 0.0 if dones[index] else 1.0
            delta = rewards[index] + config.gamma * values[index + 1] * nonterminal - values[index]
            gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
            advantages[index] = gae
        returns = [advantage + value for advantage, value in zip(advantages, values[:-1])]

        obs = torch.tensor([record.obs for record in self.records], dtype=torch.float32, device=device)
        action_masks = _action_mask_tensor(
            [record.action_mask for record in self.records],
            action_dim=max((len(record.action_mask) for record in self.records), default=0),
            device=device,
        )
        actions = torch.tensor([record.action for record in self.records], dtype=torch.long, device=device)
        old_logprobs = torch.tensor(
            [record.logprob for record in self.records],
            dtype=torch.float32,
            device=device,
        )
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std(unbiased=False) + 1e-8
        )
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        return {
            "obs": obs,
            "action_masks": action_masks,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "advantages": advantages_tensor,
            "returns": returns_tensor,
        }


if TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        """Small MLP actor-critic over protobuf-derived skill features."""

        def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_size, action_dim)
            self.critic = nn.Linear(hidden_size, 1)

        def forward(self, obs: Any) -> tuple[Any, Any]:
            """Returns action logits and value estimates."""
            hidden = self.shared(obs)
            return self.actor(hidden), self.critic(hidden).squeeze(-1)

        def act(
            self,
            obs: list[float],
            *,
            deterministic: bool = False,
            action_mask: list[bool] | None = None,
        ) -> tuple[int, float, float]:
            """Samples or selects a skill action for one observation."""
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, value = self(obs_tensor)
                if action_mask is not None:
                    mask_tensor = _action_mask_tensor([action_mask], logits.shape[-1], logits.device)
                    logits = _masked_logits(logits, mask_tensor)
                dist = Categorical(logits=logits)
                action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                logprob = dist.log_prob(action)
            return int(action.item()), float(logprob.item()), float(value.item())


else:

    class ActorCritic:  # type: ignore[no-redef]
        """Unavailable placeholder when PyTorch is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_torch()


class PPOTrainer:
    """Collects rollouts and updates an actor-critic policy."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        require_torch()
        self.config = config or PPOConfig()
        self.device = device
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.model = model or ActorCritic(
            obs_dim,
            action_dim,
            hidden_size=self.config.hidden_size,
        )
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.eval_history: list[dict[str, Any]] = []
        self.update_index = 0
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    async def collect_rollout(
        self,
        env: DoomAgentEnv,
        *,
        steps: int | None = None,
        seed: int | None = None,
    ) -> RolloutBuffer:
        """Collects on-policy transitions from the Doom environment."""
        target_steps = steps or self.config.rollout_steps
        buffer = RolloutBuffer()
        obs = await env.reset(seed=seed)
        while len(buffer) < target_steps:
            action_mask = env.action_mask()
            action, logprob, value = self.model.act(obs, action_mask=action_mask)
            transition: EnvStep = await env.step(action)
            buffer.add(
                obs=obs,
                action_mask=action_mask,
                action=action,
                reward=transition.reward,
                done=transition.done,
                value=value,
                logprob=logprob,
                info=transition.info,
            )
            obs = transition.observation
            if transition.done and len(buffer) < target_steps:
                obs = await env.reset(seed=None if seed is None else seed + len(buffer))
        with torch.no_grad():
            _, last_value = self.model(
                torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
        buffer.last_value = float(last_value.item())
        return buffer

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Runs PPO epochs over a collected rollout buffer."""
        tensors = buffer.tensors(config=self.config, device=self.device)
        count = len(buffer)
        indices = torch.arange(count, device=self.device)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        batches = 0
        for _ in range(self.config.update_epochs):
            permutation = indices[torch.randperm(count, device=self.device)]
            for start in range(0, count, self.config.minibatch_size):
                batch = permutation[start : start + self.config.minibatch_size]
                logits, values = self.model(tensors["obs"][batch])
                logits = _masked_logits(logits, tensors["action_masks"][batch])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(tensors["actions"][batch])
                entropy = dist.entropy().mean()
                ratio = (new_logprobs - tensors["old_logprobs"][batch]).exp()
                unclipped = ratio * tensors["advantages"][batch]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * tensors["advantages"][batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (tensors["returns"][batch] - values).pow(2).mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (tensors["old_logprobs"][batch] - new_logprobs).mean()
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += float(approx_kl.item())
                batches += 1

        self.update_index += 1
        return {key: value / max(1, batches) for key, value in metrics.items()}

    def pretrain_actor(
        self,
        samples: list[tuple[list[float], int]],
        *,
        epochs: int = 3,
        minibatch_size: int = 128,
        learning_rate: float | None = None,
    ) -> dict[str, float | int]:
        """Warm-starts the policy head from expert skill labels."""
        require_torch()
        if not samples:
            raise ValueError("cannot pretrain PPO actor without samples")
        rng = random.Random(self.config.seed)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate or self.config.learning_rate,
        )
        metrics = {
            "bc_loss": 0.0,
            "bc_accuracy": 0.0,
            "bc_samples": len(samples),
            "bc_epochs": epochs,
        }
        batches = 0
        for _ in range(epochs):
            rng.shuffle(samples)
            for start in range(0, len(samples), minibatch_size):
                batch = samples[start : start + minibatch_size]
                obs = torch.tensor(
                    [features for features, _action in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.tensor(
                    [action for _features, action in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                logits, _values = self.model(obs)
                loss = nn.functional.cross_entropy(logits, actions)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    accuracy = (predictions == actions).float().mean()
                metrics["bc_loss"] += float(loss.item())
                metrics["bc_accuracy"] += float(accuracy.item())
                batches += 1
        metrics["bc_loss"] = float(metrics["bc_loss"]) / max(1, batches)
        metrics["bc_accuracy"] = float(metrics["bc_accuracy"]) / max(1, batches)
        return metrics

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        reward_config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Saves model weights, optimizer state, schemas, and eval history."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": PPO_CHECKPOINT_SCHEMA,
                "created_at": _iso_now(),
                "config": asdict(self.config),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_index": self.update_index,
                "observation_schema": OBSERVATION_SCHEMA,
                "action_schema": ACTION_SCHEMA,
                "decision_cycle_schema": DECISION_CYCLE_SCHEMA,
                "memory_contract": MEMORY_CONTRACT,
                "reward_config": reward_config or {},
                "eval_history": list(self.eval_history),
                "extra": dict(extra or {}),
            },
            output,
        )
        return output

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "PPOTrainer":
        """Loads a trainer from a PPO checkpoint."""
        require_torch()
        checkpoint = torch.load(Path(path), map_location=device)
        if checkpoint.get("schema") != PPO_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"expected {PPO_CHECKPOINT_SCHEMA}, got {checkpoint.get('schema')!r}"
            )
        config = PPOConfig(**checkpoint["config"])
        obs_dim = int(checkpoint.get("obs_dim") or len(checkpoint["observation_schema"]["feature_names"]))
        action_dim = int(checkpoint.get("action_dim") or len(checkpoint["action_schema"]["actions"]))
        trainer = cls(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.update_index = int(checkpoint.get("update_index", 0))
        trainer.eval_history = list(checkpoint.get("eval_history", []))
        return trainer


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated evaluation result for a policy candidate."""

    policy_id: str
    level_completion_rate: float
    mean_kills: float
    survival_rate: float
    mean_steps_to_exit: float
    mean_stuck_events: float
    episode_count: int
    mean_reward: float = 0.0


@dataclass(frozen=True)
class PromotionDecision:
    """Decision from the PPO promotion gate."""

    promote: bool
    reasons: list[str]


class PromotionGate:
    """Compares PPO candidates against the current deterministic baseline."""

    def __init__(
        self,
        *,
        min_completion_delta: float = 0.0,
        min_kill_delta: float = 0.0,
        min_reward_delta: float = 0.0,
        min_completion_rate: float = 1.0,
        min_mean_kills: float = 1.0,
    ) -> None:
        self.min_completion_delta = min_completion_delta
        self.min_kill_delta = min_kill_delta
        self.min_reward_delta = min_reward_delta
        self.min_completion_rate = min_completion_rate
        self.min_mean_kills = min_mean_kills

    def decide(
        self,
        *,
        candidate: EvaluationResult,
        baseline: EvaluationResult,
    ) -> PromotionDecision:
        """Returns whether the candidate should replace the baseline."""
        reasons: list[str] = []
        if (
            candidate.level_completion_rate
            < baseline.level_completion_rate + self.min_completion_delta
        ):
            reasons.append("completion rate did not beat baseline")
        if candidate.level_completion_rate < self.min_completion_rate:
            reasons.append("completion rate below promotion minimum")
        if candidate.mean_kills < baseline.mean_kills + self.min_kill_delta:
            reasons.append("mean kills did not beat baseline")
        if candidate.mean_kills < self.min_mean_kills:
            reasons.append("mean kills below promotion minimum")
        if candidate.mean_reward < baseline.mean_reward + self.min_reward_delta:
            reasons.append("mean reward did not beat baseline")
        if candidate.survival_rate < baseline.survival_rate:
            reasons.append("survival rate regressed")
        if (
            baseline.mean_steps_to_exit > 0
            and candidate.mean_steps_to_exit > baseline.mean_steps_to_exit
        ):
            reasons.append("time to exit regressed")
        if candidate.mean_stuck_events > baseline.mean_stuck_events:
            reasons.append("stuck events regressed")
        improved = (
            candidate.level_completion_rate > baseline.level_completion_rate
            or candidate.mean_kills > baseline.mean_kills
            or candidate.mean_reward > baseline.mean_reward
            or (
                baseline.mean_steps_to_exit > 0
                and candidate.mean_steps_to_exit < baseline.mean_steps_to_exit
            )
            or candidate.mean_stuck_events < baseline.mean_stuck_events
        )
        if not reasons and not improved:
            reasons.append("candidate did not improve any gate metric")
        return PromotionDecision(promote=not reasons, reasons=reasons)


def require_torch() -> None:
    """Raises a clear error when PPO is used without PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for PPO. Install the agent dependencies with "
            "`pip install -e agent` in a Python version supported by torch."
        )


def _action_mask_tensor(
    masks: list[list[bool]],
    action_dim: int,
    device: str | Any = "cpu",
) -> Any:
    """Return a boolean action-mask tensor, defaulting empty rows to all actions."""
    require_torch()
    width = max(1, int(action_dim))
    normalized: list[list[bool]] = []
    for mask in masks:
        row = [bool(value) for value in mask[:width]]
        if len(row) < width:
            row.extend([True] * (width - len(row)))
        if not any(row):
            row = [True for _ in range(width)]
        normalized.append(row)
    return torch.tensor(normalized, dtype=torch.bool, device=device)


def _masked_logits(logits: Any, action_masks: Any) -> Any:
    """Apply an action mask to logits while preserving tensor shape."""
    if action_masks.shape[-1] != logits.shape[-1]:
        action_masks = _action_mask_tensor(
            action_masks.detach().cpu().tolist(),
            logits.shape[-1],
            logits.device,
        )
    return logits.masked_fill(~action_masks, -1.0e9)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
