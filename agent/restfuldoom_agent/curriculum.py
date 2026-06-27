"""Curriculum schedules for PPO reset starts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

CURRICULUM_SCHEMA = "restfuldoom.ppo_curriculum.v1"


@dataclass(frozen=True)
class CurriculumStage:
    """One reset-start stage in a PPO curriculum."""

    index: int
    name: str
    reset_start: dict[str, object]
    note: str = ""
    validated: bool = True
    requires_progressed_state: bool = False
    evidence: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Returns a JSON-serializable stage descriptor."""
        return {
            "index": self.index,
            "name": self.name,
            "reset_start": dict(self.reset_start),
            "note": self.note,
            "validated": self.validated,
            "requires_progressed_state": self.requires_progressed_state,
            "evidence": dict(self.evidence or {}),
        }


E1M1_SPAWN_TO_COMBAT_STAGES: list[CurriculumStage] = [
    CurriculumStage(
        index=0,
        name="combat_start",
        reset_start={
            "x_fp": 212860928,
            "y_fp": -214958080,
            "health": 100,
            "armor": 0,
            "ammo_bullets": 50,
            "face_nearest_enemy": True,
        },
        note="Known legal combat-start point with immediate enemy line-of-sight.",
        evidence={
            "fresh_reset_validated": True,
            "shootable_target_on_reset": True,
            "damage_observed_with_heuristic": True,
            "doom_units": {"x": 3248, "y": -3280},
        },
    ),
    CurriculumStage(
        index=1,
        name="combat_wide_left",
        reset_start={
            "x_fp": 193331200,
            "y_fp": -214958080,
            "health": 100,
            "armor": 0,
            "ammo_bullets": 50,
            "face_nearest_enemy": True,
        },
        note="Fresh-reset combat variant that broadens start position without losing immediate target affordances.",
        evidence={
            "fresh_reset_validated": True,
            "shootable_target_on_reset": True,
            "damage_observed_with_heuristic": True,
            "doom_units": {"x": 2950, "y": -3280},
        },
    ),
    CurriculumStage(
        index=2,
        name="combat_back_left",
        reset_start={
            "x_fp": 199884800,
            "y_fp": -221511680,
            "health": 100,
            "armor": 0,
            "ammo_bullets": 50,
            "face_nearest_enemy": True,
        },
        note="Fresh-reset combat variant with a deeper approach angle and immediate shootable enemy.",
        evidence={
            "fresh_reset_validated": True,
            "shootable_target_on_reset": True,
            "damage_observed_with_heuristic": True,
            "doom_units": {"x": 3050, "y": -3380},
        },
    ),
    CurriculumStage(
        index=3,
        name="fresh_spawn",
        reset_start={},
        note="Real E1M1 spawn; no teleport override. Current PPO shows route progress here but not reliable combat contact.",
        evidence={
            "fresh_reset_validated": True,
            "shootable_target_on_reset": False,
            "latest_spawn_trend": "route progress without shootable target, damage, or kills",
        },
    ),
]

CURRICULUM_PRESETS: dict[str, list[CurriculumStage]] = {
    "e1m1-spawn-to-combat": E1M1_SPAWN_TO_COMBAT_STAGES,
}

CURRICULUM_MODES = {"round_robin", "progressive", "random"}


def curriculum_names() -> list[str]:
    """Returns available named curricula."""
    return sorted(CURRICULUM_PRESETS)


def build_curriculum(
    *,
    name: str | None,
    manual_reset_start: dict[str, object],
    mode: str = "round_robin",
    start_index: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Builds a serializable curriculum descriptor."""
    if mode not in CURRICULUM_MODES:
        available = ", ".join(sorted(CURRICULUM_MODES))
        raise ValueError(f"unknown curriculum mode {mode!r}; choose one of: {available}")

    if not name or name == "none":
        return {
            "schema": CURRICULUM_SCHEMA,
            "name": "none",
            "mode": "single",
            "start_index": 0,
            "seed": int(seed),
            "stages": [
                CurriculumStage(
                    index=0,
                    name="manual_reset_start" if manual_reset_start else "fresh_spawn",
                    reset_start=dict(manual_reset_start),
                    note="Single reset start resolved from CLI arguments.",
                ).to_dict()
            ],
        }

    try:
        stages = CURRICULUM_PRESETS[name]
    except KeyError as error:
        available = ", ".join(["none", *curriculum_names()])
        raise ValueError(f"unknown curriculum {name!r}; choose one of: {available}") from error

    if manual_reset_start:
        raise ValueError("--curriculum cannot be combined with explicit --reset-start-* options")

    return {
        "schema": CURRICULUM_SCHEMA,
        "name": name,
        "mode": mode,
        "start_index": max(0, int(start_index)),
        "seed": int(seed),
        "stages": [stage.to_dict() for stage in stages],
    }


def stage_for_update(curriculum: dict[str, Any], update_index: int) -> dict[str, Any]:
    """Returns the active curriculum stage for an update."""
    stages = curriculum.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum has no stages")

    mode = str(curriculum.get("mode", "single"))
    start_index = int(curriculum.get("start_index", 0))
    update_index = max(0, int(update_index))

    if mode == "single":
        index = 0
    elif mode == "round_robin":
        index = (start_index + update_index) % len(stages)
    elif mode == "progressive":
        index = min(len(stages) - 1, start_index + update_index)
    elif mode == "random":
        rng = random.Random(int(curriculum.get("seed", 0)) + update_index)
        index = rng.randrange(len(stages))
    else:
        raise ValueError(f"unsupported curriculum mode {mode!r}")

    stage = dict(stages[index])
    stage["selected_index"] = index
    return stage
