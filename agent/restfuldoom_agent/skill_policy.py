"""Trainable skill-selection policy for the structured Doom brain.

This module intentionally avoids heavyweight ML dependencies.  It trains a
small multiclass softmax model over compact protobuf-derived features and saves
the checkpoint as JSON.  The schema is stable enough to export with training
jobs and can later be replaced by a PyTorch implementation without changing the
brain/export contract.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKILL_POLICY_SCHEMA = "restfuldoom.skill_policy.v1"

FEATURE_NAMES = [
    "health_norm",
    "ammo_norm",
    "kills_norm",
    "items_norm",
    "x_units_norm",
    "y_units_norm",
    "angle_sin",
    "angle_cos",
    "visible_enemies_norm",
    "known_enemies_norm",
    "remembered_enemies_norm",
    "enemy_count_norm",
    "has_enemy",
    "enemy_distance_norm",
    "enemy_angle_sin",
    "enemy_angle_cos",
    "enemy_threat_norm",
    "enemy_health_norm",
    "combat_has_target",
    "combat_target_enemy",
    "combat_target_distance_norm",
    "nav_forward_open",
    "nav_back_open",
    "nav_left_open",
    "nav_right_open",
    "nav_use_line_ahead",
    "nav_front_distance_norm",
    "nav_front_special_manual",
    "nav_front_special_exit",
    "nav_open_probe_ratio",
    "nav_use_probe_ratio",
    "nav_best_open_angle_norm",
    "topology_frontier_count_norm",
    "has_use_line",
    "use_line_distance_norm",
    "use_line_angle_sin",
    "use_line_angle_cos",
    "use_line_manual",
    "use_line_exit",
    "use_line_side",
    "use_line_front_distance_norm",
    "stuck",
    "blocked_targets_norm",
    "sector_damaging",
    "sector_damage_norm",
    "sector_exit_damage",
    "sector_floor_height_norm",
    "sector_ceiling_height_norm",
    "route_has_waypoint",
    "route_waypoint_distance_norm",
    "route_waypoint_angle_sin",
    "route_waypoint_angle_cos",
    "route_waypoint_priority_norm",
    "route_waypoint_exit",
    "route_waypoint_walk_trigger",
]

MANUAL_USE_LINE_SPECIALS = frozenset(
    {
        1,
        7,
        9,
        11,
        14,
        15,
        18,
        20,
        21,
        23,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        41,
        42,
        43,
        45,
        49,
        50,
        51,
        55,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        101,
        102,
        103,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        122,
        123,
        127,
        131,
        132,
        133,
        135,
        137,
        140,
    }
)
EXIT_LINE_SPECIALS = frozenset({11, 51})


@dataclass(frozen=True)
class SkillPolicyTrainConfig:
    """Configuration for behavior-cloning skill decisions from trajectories."""

    epochs: int = 12
    learning_rate: float = 0.08
    l2: float = 0.0001
    min_count: int = 4
    max_samples: int = 20000
    max_records_per_file: int = 6000
    seed: int = 7


class SkillPolicyModel:
    """Small softmax skill classifier loaded from a JSON checkpoint."""

    def __init__(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("schema") != SKILL_POLICY_SCHEMA:
            raise ValueError(
                f"expected {SKILL_POLICY_SCHEMA}, got {checkpoint.get('schema')!r}"
            )
        self.checkpoint = checkpoint
        self.feature_names = list(checkpoint["feature_names"])
        self.classes = list(checkpoint["classes"])
        self.weights = [
            [float(value) for value in row] for row in checkpoint["weights"]
        ]
        self.biases = [float(value) for value in checkpoint["biases"]]

    @classmethod
    def load(cls, path: str | Path) -> "SkillPolicyModel":
        """Load a model checkpoint."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_vector(self, vector: list[float], *, top_k: int = 3) -> dict[str, Any]:
        """Predict the most likely skill for an already encoded feature vector."""
        probs = _softmax(_logits(self.weights, self.biases, vector))
        ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        top = [
            {"skill": self.classes[index], "probability": round(probability, 4)}
            for index, probability in ranked[:top_k]
        ]
        return {
            "skill": top[0]["skill"] if top else None,
            "confidence": top[0]["probability"] if top else 0.0,
            "top": top,
        }

    def predict_record(self, record: dict[str, Any], *, top_k: int = 3) -> dict[str, Any]:
        """Predict from a stored trajectory JSONL record."""
        return self.predict_vector(features_from_record(record), top_k=top_k)

    def predict_tactical(self, features: Any, *, top_k: int = 3) -> dict[str, Any]:
        """Predict from the live `TacticalFeatures` object without importing brain."""
        return self.predict_vector(features_from_tactical(features), top_k=top_k)


def train_skill_policy(
    trajectory_paths: list[str | Path],
    output_path: str | Path,
    *,
    config: SkillPolicyTrainConfig | None = None,
) -> dict[str, Any]:
    """Train and write a portable skill-policy checkpoint."""
    train_config = config or SkillPolicyTrainConfig()
    samples, label_counts, source_paths = _load_samples(trajectory_paths, train_config)
    if not samples:
        raise ValueError("no usable policy_decision samples found in trajectories")

    kept_labels = {
        label for label, count in label_counts.items() if count >= train_config.min_count
    }
    samples = [(features, label) for features, label in samples if label in kept_labels]
    if len(kept_labels) < 2:
        raise ValueError(
            "need at least two skill labels after min_count filtering; "
            f"got {sorted(kept_labels)}"
        )
    if not samples:
        raise ValueError("no samples remain after label filtering")

    rng = random.Random(train_config.seed)
    rng.shuffle(samples)
    split_index = max(1, int(len(samples) * 0.8))
    train_samples = samples[:split_index]
    eval_samples = samples[split_index:] or samples[:]

    classes = sorted(kept_labels)
    class_index = {label: index for index, label in enumerate(classes)}
    weights = [[0.0 for _ in FEATURE_NAMES] for _ in classes]
    biases = [0.0 for _ in classes]
    filtered_counts = Counter(label for _, label in samples)
    class_weights = {
        label: math.sqrt(len(samples) / (len(classes) * filtered_counts[label]))
        for label in classes
    }

    for _ in range(max(1, train_config.epochs)):
        rng.shuffle(train_samples)
        for vector, label in train_samples:
            target = class_index[label]
            probs = _softmax(_logits(weights, biases, vector))
            sample_weight = class_weights[label]
            for class_id in range(len(classes)):
                error = (probs[class_id] - (1.0 if class_id == target else 0.0)) * sample_weight
                biases[class_id] -= train_config.learning_rate * error
                row = weights[class_id]
                for feature_index, value in enumerate(vector):
                    gradient = error * value + train_config.l2 * row[feature_index]
                    row[feature_index] -= train_config.learning_rate * gradient

    train_accuracy = _accuracy(weights, biases, classes, train_samples)
    eval_accuracy = _accuracy(weights, biases, classes, eval_samples)
    checkpoint = {
        "schema": SKILL_POLICY_SCHEMA,
        "created_at": _iso_now(),
        "model_type": "softmax_skill_classifier",
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "weights": [[round(value, 8) for value in row] for row in weights],
        "biases": [round(value, 8) for value in biases],
        "train": {
            "sample_count": len(samples),
            "train_sample_count": len(train_samples),
            "eval_sample_count": len(eval_samples),
            "train_accuracy": round(train_accuracy, 4),
            "eval_accuracy": round(eval_accuracy, 4),
            "epochs": train_config.epochs,
            "learning_rate": train_config.learning_rate,
            "l2": train_config.l2,
            "min_count": train_config.min_count,
            "max_samples": train_config.max_samples,
            "max_records_per_file": train_config.max_records_per_file,
            "seed": train_config.seed,
            "label_counts": dict(sorted(filtered_counts.items())),
            "source_trajectories": [str(path) for path in source_paths],
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    return {
        "schema": SKILL_POLICY_SCHEMA,
        "checkpoint_path": str(output),
        "sample_count": len(samples),
        "class_count": len(classes),
        "train_accuracy": round(train_accuracy, 4),
        "eval_accuracy": round(eval_accuracy, 4),
        "classes": classes,
        "label_counts": dict(sorted(filtered_counts.items())),
    }


def features_from_record(record: dict[str, Any]) -> list[float]:
    """Encode one trajectory JSONL row."""
    decision = (
        record.get("metadata", {}).get("policy_decision", {})
        if isinstance(record.get("metadata"), dict)
        else {}
    )
    state = record.get("state", {}) if isinstance(record.get("state"), dict) else {}
    return features_from_decision(decision, state)


def features_from_decision(
    decision: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> list[float]:
    """Encode the compact policy-decision metadata saved in trajectories."""
    state = state or {}
    navigation = decision.get("navigation")
    if not isinstance(navigation, dict):
        navigation = state.get("navigation") if isinstance(state.get("navigation"), dict) else {}
    combat = decision.get("combat")
    if not isinstance(combat, dict):
        combat = state.get("combat") if isinstance(state.get("combat"), dict) else {}
    enemy = decision.get("enemy") if isinstance(decision.get("enemy"), dict) else {}
    use_line = decision.get("use_line") if isinstance(decision.get("use_line"), dict) else {}
    current_sector = (
        navigation.get("current_sector")
        if isinstance(navigation.get("current_sector"), dict)
        else {}
    )
    route_waypoint = (
        navigation.get("route_waypoint")
        if isinstance(navigation.get("route_waypoint"), dict)
        else {}
    )
    route_line = (
        route_waypoint.get("line")
        if isinstance(route_waypoint.get("line"), dict)
        else {}
    )

    direction_probes = navigation.get("direction_probes", [])
    if not isinstance(direction_probes, list):
        direction_probes = []
    open_probes = [probe for probe in direction_probes if probe.get("open")]
    use_probes = [probe for probe in direction_probes if probe.get("use_line_ahead")]
    best_open_angle = min(
        (abs(_to_float(probe.get("angle_offset_degrees"))) for probe in open_probes),
        default=90.0,
    )

    special = int(_to_float(use_line.get("special")))
    front_special = int(_to_float(navigation.get("front_blocking_line_special")))
    angle = _to_float(use_line.get("angle_delta"))
    enemy_angle = _to_float(enemy.get("angle_delta"))
    x_units, y_units = _position_units(decision, state)
    player_angle = _player_angle(decision, state)
    route_angle = _line_angle_delta(route_line, x_units, y_units, player_angle)
    return [
        _norm(_first_number(decision, state, "health"), 100.0),
        _norm(_first_number(decision, state, "ammo_bullets"), 80.0),
        _norm(_first_number(decision, state, "kills"), 10.0),
        _norm(_first_number(decision, state, "items"), 20.0),
        _norm(x_units, 4096.0),
        _norm(y_units, 4096.0),
        math.sin(math.radians(player_angle)),
        math.cos(math.radians(player_angle)),
        _norm(_first_number(decision, state, "visible_enemies"), 8.0),
        _norm(_first_number(decision, state, "known_enemies"), 16.0),
        _norm(_first_number(decision, state, "remembered_enemies"), 16.0),
        _norm(_first_number(decision, state, "enemy_count"), 32.0),
        1.0 if enemy else 0.0,
        _norm(_to_float(enemy.get("distance")), 2400.0),
        math.sin(math.radians(enemy_angle)),
        math.cos(math.radians(enemy_angle)),
        _norm(_to_float(enemy.get("threat")), 10.0),
        _norm(_to_float(enemy.get("health")), 100.0),
        1.0 if combat.get("has_shootable_target") else 0.0,
        1.0 if combat.get("target_is_enemy") else 0.0,
        _norm(_to_float(combat.get("target_distance_fp")) / 65536.0, 2400.0),
        1.0 if navigation.get("forward_open", True) else 0.0,
        1.0 if navigation.get("back_open", True) else 0.0,
        1.0 if navigation.get("left_open", True) else 0.0,
        1.0 if navigation.get("right_open", True) else 0.0,
        1.0 if navigation.get("use_line_ahead", False) else 0.0,
        _norm(_to_float(navigation.get("front_block_distance_fp")) / 65536.0, 512.0),
        1.0 if front_special in MANUAL_USE_LINE_SPECIALS else 0.0,
        1.0 if front_special in EXIT_LINE_SPECIALS else 0.0,
        _norm(len(open_probes), max(1, len(direction_probes))),
        _norm(len(use_probes), max(1, len(direction_probes))),
        _norm(best_open_angle, 90.0),
        _norm(_to_float(navigation.get("topology_frontier_count")), max(1, len(direction_probes))),
        1.0 if use_line else 0.0,
        _norm(_to_float(use_line.get("distance")), 1600.0),
        math.sin(math.radians(angle)),
        math.cos(math.radians(angle)),
        1.0 if special in MANUAL_USE_LINE_SPECIALS else 0.0,
        1.0 if special in EXIT_LINE_SPECIALS else 0.0,
        1.0 if int(_to_float(use_line.get("side"))) == 1 else 0.0,
        _norm(_to_float(use_line.get("front_distance")), 1600.0),
        1.0 if decision.get("stuck") else 0.0,
        _norm(_first_number(decision, state, "blocked_targets"), 16.0),
        1.0 if current_sector.get("damaging") else 0.0,
        _norm(_to_float(current_sector.get("damage_per_32_tics")), 20.0),
        1.0 if current_sector.get("exit_damage") else 0.0,
        _norm(_to_float(current_sector.get("floor_height_fp")) / 65536.0, 1024.0),
        _norm(_to_float(current_sector.get("ceiling_height_fp")) / 65536.0, 1024.0),
        1.0 if route_waypoint else 0.0,
        _norm(_line_distance_units(route_line, x_units, y_units), 2600.0),
        math.sin(math.radians(route_angle)),
        math.cos(math.radians(route_angle)),
        _norm(_to_float(route_waypoint.get("priority")), 4.0),
        1.0 if route_waypoint.get("exit") else 0.0,
        1.0 if route_waypoint.get("walk_trigger") else 0.0,
    ]


def features_from_tactical(features: Any) -> list[float]:
    """Encode live `TacticalFeatures` without creating a dependency cycle."""
    enemy = None
    if getattr(features, "visible_enemies", None):
        enemy = features.visible_enemies[0]
    elif getattr(features, "known_enemies", None):
        enemy = features.known_enemies[0]
    use_lines = getattr(features, "navigation", {}).get("use_lines", [])
    use_line = use_lines[0] if use_lines else {}
    decision = {
        "health": getattr(features, "health", 0),
        "ammo_bullets": getattr(features, "ammo_bullets", 0),
        "kills": getattr(features, "kills", 0),
        "items": getattr(features, "items", 0),
        "x_units": getattr(features, "x_units", 0.0),
        "y_units": getattr(features, "y_units", 0.0),
        "angle": getattr(features, "angle", 0.0),
        "visible_enemies": len(getattr(features, "visible_enemies", [])),
        "known_enemies": len(getattr(features, "known_enemies", [])),
        "remembered_enemies": len(getattr(features, "remembered_enemies", [])),
        "enemy_count": getattr(features, "enemy_count", 0),
        "enemy": enemy or {},
        "navigation": getattr(features, "navigation", {}),
        "combat": getattr(features, "combat", {}),
        "use_line": use_line,
    }
    return features_from_decision(decision)


def _position_units(primary: dict[str, Any], fallback: dict[str, Any]) -> tuple[float, float]:
    x = _first_number(primary, fallback, "x_units")
    y = _first_number(primary, fallback, "y_units")
    if x or y:
        return x, y

    position_fp = fallback.get("position_fp")
    if isinstance(position_fp, list) and len(position_fp) >= 2:
        return _to_float(position_fp[0]) / 65536.0, _to_float(position_fp[1]) / 65536.0

    cell = primary.get("cell") or fallback.get("cell")
    if isinstance(cell, str):
        parts = cell.split(":", 1)
        if len(parts) == 2:
            return _to_float(parts[0]) * 128.0, _to_float(parts[1]) * 128.0
    return 0.0, 0.0


def _player_angle(primary: dict[str, Any], fallback: dict[str, Any]) -> float:
    for key in ("angle", "angle_degrees", "player_angle"):
        if key in primary:
            return _to_float(primary.get(key)) % 360.0
        if key in fallback:
            return _to_float(fallback.get(key)) % 360.0
    return 0.0


def _line_distance_units(line: dict[str, Any], x_units: float, y_units: float) -> float:
    for key in ("distance", "nearest_distance", "front_distance"):
        if key in line:
            return _to_float(line.get(key))
    if "nearest_distance_fp" in line:
        return _to_float(line.get("nearest_distance_fp")) / 65536.0
    point = line.get("nearest_point_fp")
    if isinstance(point, list) and len(point) >= 2:
        return math.dist((x_units, y_units), (_to_float(point[0]) / 65536.0, _to_float(point[1]) / 65536.0))
    return 0.0


def _line_angle_delta(
    line: dict[str, Any],
    x_units: float,
    y_units: float,
    player_angle: float,
) -> float:
    if "angle_delta" in line:
        return _to_float(line.get("angle_delta"))
    if "nearest_x_units" in line and "nearest_y_units" in line:
        return _angle_to_target_units(
            x_units,
            y_units,
            player_angle,
            _to_float(line.get("nearest_x_units")),
            _to_float(line.get("nearest_y_units")),
        )
    point = line.get("nearest_point_fp")
    if isinstance(point, list) and len(point) >= 2:
        return _angle_to_target_units(
            x_units,
            y_units,
            player_angle,
            _to_float(point[0]) / 65536.0,
            _to_float(point[1]) / 65536.0,
        )
    return 0.0


def _angle_to_target_units(
    x_units: float,
    y_units: float,
    player_angle: float,
    target_x_units: float,
    target_y_units: float,
) -> float:
    target_angle = math.degrees(math.atan2(target_y_units - y_units, target_x_units - x_units))
    return ((target_angle - player_angle + 540.0) % 360.0) - 180.0


def _load_samples(
    trajectory_paths: list[str | Path],
    config: SkillPolicyTrainConfig,
) -> tuple[list[tuple[list[float], str]], Counter[str], list[Path]]:
    samples: list[tuple[list[float], str]] = []
    counts: Counter[str] = Counter()
    source_paths: list[Path] = []
    for raw_path in trajectory_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        source_paths.append(path)
        seen = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(samples) >= config.max_samples:
                    return samples, counts, source_paths
                if seen >= config.max_records_per_file:
                    break
                seen += 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision = record.get("metadata", {}).get("policy_decision", {})
                if not isinstance(decision, dict):
                    continue
                label = decision.get("skill")
                if not isinstance(label, str) or not label:
                    continue
                vector = features_from_record(record)
                if len(vector) != len(FEATURE_NAMES):
                    continue
                counts[label] += 1
                samples.append((vector, label))
    return samples, counts, source_paths


def _logits(
    weights: list[list[float]],
    biases: list[float],
    vector: list[float],
) -> list[float]:
    return [
        bias + sum(weight * value for weight, value in zip(row, vector))
        for row, bias in zip(weights, biases)
    ]


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    pivot = max(logits)
    exps = [math.exp(max(-60.0, min(60.0, value - pivot))) for value in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _accuracy(
    weights: list[list[float]],
    biases: list[float],
    classes: list[str],
    samples: list[tuple[list[float], str]],
) -> float:
    if not samples:
        return 0.0
    correct = 0
    for vector, label in samples:
        probs = _softmax(_logits(weights, biases, vector))
        prediction = classes[max(range(len(probs)), key=lambda index: probs[index])]
        if prediction == label:
            correct += 1
    return correct / len(samples)


def _first_number(primary: dict[str, Any], fallback: dict[str, Any], key: str) -> float:
    if key in primary:
        return _to_float(primary.get(key))
    return _to_float(fallback.get(key))


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(-1.0, min(1.0, value / scale))


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
