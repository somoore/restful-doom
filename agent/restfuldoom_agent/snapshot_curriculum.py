"""Snapshot-backed PPO curriculum manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .curriculum import CURRICULUM_SCHEMA, CURRICULUM_MODES

SNAPSHOT_CURRICULUM_SCHEMA = "restfuldoom.snapshot_curriculum.v1"
SNAPSHOT_CURRICULUM_VALIDATION_SCHEMA = "restfuldoom.snapshot_curriculum_validation.v1"


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


def validate_snapshot_curriculum(
    path: str | Path,
    *,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate a snapshot curriculum manifest and any local artifacts it names."""
    manifest_path = Path(path)
    report: dict[str, Any] = {
        "schema": SNAPSHOT_CURRICULUM_VALIDATION_SCHEMA,
        "manifest_path": str(manifest_path),
        "valid": False,
        "stage_count": 0,
        "artifact_count": 0,
        "missing_artifacts": [],
        "digest_mismatches": [],
        "warnings": [],
        "errors": [],
        "stages": [],
    }
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except OSError as error:
        report["errors"].append(f"could not read manifest: {error}")
        return report
    except json.JSONDecodeError as error:
        report["errors"].append(f"could not parse manifest JSON: {error}")
        return report

    if manifest.get("schema") != SNAPSHOT_CURRICULUM_SCHEMA:
        report["errors"].append(
            f"expected {SNAPSHOT_CURRICULUM_SCHEMA}, got {manifest.get('schema')!r}"
        )
        return report

    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        report["errors"].append("snapshot curriculum must include at least one stage")
        return report
    report["stage_count"] = len(stages)

    for index, raw_stage in enumerate(stages):
        stage_report = _validate_stage(
            index,
            raw_stage,
            manifest_path=manifest_path,
            require_artifacts=require_artifacts,
        )
        report["stages"].append(stage_report)
        if stage_report.get("artifact_exists"):
            report["artifact_count"] += 1
        if stage_report.get("missing_artifact"):
            report["missing_artifacts"].append(stage_report["name"])
        if stage_report.get("digest_mismatch"):
            report["digest_mismatches"].append(stage_report["name"])
        report["warnings"].extend(stage_report.get("warnings", []))
        report["errors"].extend(stage_report.get("errors", []))

    report["valid"] = not report["errors"]
    return report


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


def _validate_stage(
    index: int,
    raw_stage: object,
    *,
    manifest_path: Path,
    require_artifacts: bool,
) -> dict[str, Any]:
    stage_report: dict[str, Any] = {
        "index": index,
        "name": f"snapshot_stage_{index}",
        "valid": False,
        "artifact_exists": False,
        "missing_artifact": False,
        "digest_mismatch": False,
        "warnings": [],
        "errors": [],
    }
    if not isinstance(raw_stage, dict):
        stage_report["errors"].append(f"stage {index} must be an object")
        return stage_report

    snapshot = raw_stage.get("snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        stage_report["errors"].append(f"stage {index} must include snapshot metadata")
        return stage_report

    name = str(raw_stage.get("name") or snapshot.get("id") or f"snapshot_stage_{index}")
    stage_report["name"] = name
    stage_report["validated"] = bool(raw_stage.get("validated", False))
    stage_report["snapshot_id"] = snapshot.get("id")
    stage_report["snapshot_ref"] = snapshot.get("ref")
    stage_report["snapshot_path"] = snapshot.get("path")
    stage_report["snapshot_slot"] = _snapshot_slot(snapshot)
    stage_report["expected_digest"] = snapshot.get("digest")

    reset_start = raw_stage.get("reset_start", {})
    if reset_start is not None and not isinstance(reset_start, dict):
        stage_report["errors"].append(f"stage {index} reset_start must be an object")
    expected = raw_stage.get("expected_state", raw_stage.get("expected", {}))
    if expected is not None and not isinstance(expected, dict):
        stage_report["errors"].append(f"stage {index} expected_state must be an object")

    if not any(snapshot.get(key) for key in ("id", "path", "ref", "slot")):
        stage_report["errors"].append(
            f"stage {index} snapshot must include at least one of id, path, ref, or slot"
        )

    raw_path = snapshot.get("path")
    if isinstance(raw_path, str) and raw_path:
        artifact_path = _resolve_snapshot_artifact_path(raw_path, manifest_path)
        stage_report["resolved_path"] = str(artifact_path)
        if artifact_path.exists():
            stage_report["artifact_exists"] = True
            digest = f"sha256:{_sha256_path(artifact_path)}"
            stage_report["actual_digest"] = digest
            expected_digest = snapshot.get("digest")
            if _is_real_sha256_digest(expected_digest) and expected_digest != digest:
                stage_report["digest_mismatch"] = True
                stage_report["errors"].append(
                    f"stage {index} digest mismatch for {raw_path}: "
                    f"expected {expected_digest}, got {digest}"
                )
            elif not _is_real_sha256_digest(expected_digest):
                stage_report["warnings"].append(
                    f"stage {index} artifact exists but digest is missing or placeholder"
                )
        else:
            stage_report["missing_artifact"] = True
            message = f"stage {index} snapshot artifact is missing: {raw_path}"
            if require_artifacts:
                stage_report["errors"].append(message)
            else:
                stage_report["warnings"].append(message)
    elif stage_report["snapshot_slot"] is not None:
        message = (
            f"stage {index} uses native save slot {stage_report['snapshot_slot']} "
            "without a bundled snapshot.path; it is portable only on the "
            "originating Doom server"
        )
        if require_artifacts:
            stage_report["errors"].append(message)
        else:
            stage_report["warnings"].append(message)
    elif require_artifacts:
        stage_report["errors"].append(
            f"stage {index} has no local snapshot.path to validate"
        )

    stage_report["valid"] = not stage_report["errors"]
    return stage_report


def _resolve_snapshot_artifact_path(raw_path: str, manifest_path: Path) -> Path:
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute():
        return artifact_path
    if artifact_path.exists():
        return artifact_path
    manifest_relative = manifest_path.parent / artifact_path
    if manifest_relative.exists():
        return manifest_relative
    return artifact_path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_real_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(character in "0123456789abcdef" for character in suffix)


def _snapshot_slot(snapshot: dict[str, Any]) -> int | None:
    value = snapshot.get("slot")
    if value is None:
        ref = snapshot.get("ref")
        if isinstance(ref, str) and ref.startswith("save_slot:"):
            value = ref.removeprefix("save_slot:")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--validate", action="store_true", help="validate the manifest")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="fail validation unless every stage has a local artifact with a matching digest",
    )
    args = parser.parse_args(argv)

    if not args.validate:
        parser.error("only --validate is currently supported")
    report = validate_snapshot_curriculum(
        args.manifest,
        require_artifacts=args.require_artifacts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
