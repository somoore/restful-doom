"""Snapshot-backed PPO curriculum manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .curriculum import CURRICULUM_SCHEMA, CURRICULUM_MODES

SNAPSHOT_CURRICULUM_SCHEMA = "restfuldoom.snapshot_curriculum.v1"


def load_snapshot_curriculum(
    path: str | Path,
    *,
    mode: str = "round_robin",
    start_index: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Loads a progressed-state curriculum from a versioned JSON manifest."""
    if mode not in CURRICULUM_MODES:
        available = ", ".join(sorted(CURRICULUM_MODES))
        raise ValueError(f"unknown curriculum mode {mode!r}; choose one of: {available}")

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SNAPSHOT_CURRICULUM_SCHEMA:
        raise ValueError(
            f"expected {SNAPSHOT_CURRICULUM_SCHEMA}, got {manifest.get('schema')!r}"
        )
    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("snapshot curriculum must include at least one stage")

    name = str(manifest.get("name") or manifest_path.stem)
    stage_records = [
        _snapshot_stage(index, stage, manifest_path=manifest_path)
        for index, stage in enumerate(stages)
    ]
    return {
        "schema": CURRICULUM_SCHEMA,
        "source_schema": SNAPSHOT_CURRICULUM_SCHEMA,
        "name": name,
        "mode": mode,
        "start_index": max(0, int(start_index)),
        "seed": int(seed),
        "snapshot_curriculum": {
            "schema": SNAPSHOT_CURRICULUM_SCHEMA,
            "name": name,
            "manifest_path": str(manifest_path),
            "stage_count": len(stage_records),
            "source": dict(manifest.get("source", {}))
            if isinstance(manifest.get("source"), dict)
            else {},
        },
        "stages": stage_records,
    }


def _snapshot_stage(
    index: int,
    raw_stage: object,
    *,
    manifest_path: Path,
) -> dict[str, object]:
    if not isinstance(raw_stage, dict):
        raise ValueError(f"snapshot curriculum stage {index} must be an object")
    snapshot = raw_stage.get("snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError(f"snapshot curriculum stage {index} must include snapshot metadata")
    reset_start = raw_stage.get("reset_start", {})
    if not isinstance(reset_start, dict):
        raise ValueError(f"snapshot curriculum stage {index} reset_start must be an object")
    expected = raw_stage.get("expected_state", raw_stage.get("expected", {}))
    if not isinstance(expected, dict):
        raise ValueError(f"snapshot curriculum stage {index} expected_state must be an object")

    evidence = (
        dict(raw_stage.get("evidence", {}))
        if isinstance(raw_stage.get("evidence"), dict)
        else {}
    )
    evidence.update(
        {
            "snapshot_backed": True,
            "manifest_path": str(manifest_path),
        }
    )
    if expected:
        evidence["expected_state"] = dict(expected)

    name = str(raw_stage.get("name") or snapshot.get("id") or f"snapshot_stage_{index}")
    return {
        "index": int(raw_stage.get("index", index)),
        "name": name,
        "reset_start": dict(reset_start),
        "note": str(raw_stage.get("note", "Progressed-state snapshot curriculum stage.")),
        "validated": bool(raw_stage.get("validated", False)),
        "requires_progressed_state": True,
        "reset_mode": "snapshot",
        "snapshot": dict(snapshot),
        "expected_state": dict(expected),
        "evidence": evidence,
    }
