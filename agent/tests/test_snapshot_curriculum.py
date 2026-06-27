import json
import sys
from pathlib import Path

from restfuldoom_agent.snapshot_builder import (
    _redact_command,
    build_snapshot_curriculum_from_trajectory,
)
from restfuldoom_agent.snapshot_curriculum import validate_snapshot_curriculum


def test_snapshot_builder_generates_manifest_from_trajectory_rows(tmp_path):
    trajectory = tmp_path / "run.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=False, shootable=False),
            _trajectory_row(1, visible=True, shootable=False, skill="seek_enemy"),
            _trajectory_row(
                2,
                visible=True,
                shootable=True,
                skill="fire",
                damage_delta=20,
            ),
        ],
    )

    manifest = build_snapshot_curriculum_from_trajectory(
        trajectory,
        output_path=tmp_path / "snapshot-curriculum.json",
        name="e1m1-progressed",
        indexes=[1],
        auto_selectors=["first-shootable", "first-damage"],
        snapshot_dir=Path("snapshots"),
    )

    assert manifest["schema"] == "restfuldoom.snapshot_curriculum.v1"
    assert manifest["source"]["selection"] == {
        "indexes": [1],
        "auto": ["first-shootable", "first-damage"],
    }
    assert [stage["evidence"]["source_record_index"] for stage in manifest["stages"]] == [
        1,
        2,
    ]
    first, second = manifest["stages"]
    assert first["name"] == "0001-explicit_snapshot"
    assert first["reset_start"]["x_fp"] == 101 * 65536
    assert first["expected_state"]["visible_enemy"] is True
    assert first["expected_state"]["shootable_target"] is False
    assert second["evidence"]["selectors"] == ["first-shootable", "first-damage"]
    assert second["expected_state"]["shootable_target"] is True
    assert second["expected_state"]["damage_delta"] == 20

    validation = validate_snapshot_curriculum(tmp_path / "snapshot-curriculum.json")
    assert validation["valid"] is True
    assert validation["stage_count"] == 2
    assert validation["missing_artifacts"] == [
        "0001-explicit_snapshot",
        "0002-first-shootable_snapshot",
    ]


def test_snapshot_validation_checks_required_artifacts_and_digests(tmp_path):
    snapshot = tmp_path / "first-contact.snap"
    snapshot.write_bytes(b"snapshot")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.snapshot_curriculum.v1",
                "name": "bad-digest",
                "stages": [
                    {
                        "name": "first_contact_snapshot",
                        "snapshot": {
                            "id": "snap-1",
                            "path": str(snapshot),
                            "digest": "sha256:" + "0" * 64,
                        },
                        "expected_state": {"episode": 1, "map": 1, "tick": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    validation = validate_snapshot_curriculum(manifest, require_artifacts=True)

    assert validation["valid"] is False
    assert validation["digest_mismatches"] == ["first_contact_snapshot"]
    assert "digest mismatch" in validation["errors"][0]


def test_snapshot_builder_capture_command_populates_digest(tmp_path):
    trajectory = tmp_path / "run.jsonl"
    _write_jsonl(trajectory, [_trajectory_row(0, visible=True, shootable=True)])
    output = tmp_path / "manifest.json"
    capture_command = (
        f"{sys.executable} -c "
        "\"from pathlib import Path; "
        "p=Path({snapshot_path_py}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_bytes(b'snapshot-bytes')\""
    )

    manifest = build_snapshot_curriculum_from_trajectory(
        trajectory,
        output_path=output,
        indexes=[0],
        snapshot_dir=Path("snapshots"),
        capture_command=capture_command,
        capture_cwd=tmp_path,
        require_capture_artifacts=True,
    )

    stage = manifest["stages"][0]
    assert stage["validated"] is True
    assert stage["capture"]["returncode"] == 0
    assert stage["snapshot"]["digest"].startswith("sha256:")
    assert len(stage["snapshot"]["digest"]) == len("sha256:") + 64

    validation = validate_snapshot_curriculum(output, require_artifacts=True)
    assert validation["valid"] is True
    assert validation["artifact_count"] == 1


def test_snapshot_capture_command_redacts_tokens():
    redacted = _redact_command(
        "shrink snapshot --token secret --x-aws-proxy-auth=abc --output snap"
    )

    assert "secret" not in redacted
    assert "abc" not in redacted
    assert "<redacted>" in redacted


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _trajectory_row(
    index: int,
    *,
    visible: bool,
    shootable: bool,
    skill: str = "route_progression",
    damage_delta: int = 0,
    kill_delta: int = 0,
) -> dict:
    return {
        "index": index,
        "state": {
            "tick": 100 + index,
            "episode": 1,
            "map": 1,
            "health": 100,
            "armor": 0,
            "ammo_bullets": 50,
            "kills": kill_delta,
            "items": 0,
            "position_fp": [(100 + index) * 65536, -200 * 65536, 0],
            "combat": {
                "has_shootable_target": shootable,
                "target_is_enemy": shootable,
            },
        },
        "reward": {
            "reward": float(damage_delta + kill_delta * 100),
            "damage_delta": damage_delta,
            "kill_delta": kill_delta,
        },
        "metadata": {
            "policy_decision": {
                "skill": skill,
                "visible_enemies": 1 if visible else 0,
                "combat": {"has_shootable_target": shootable},
            }
        },
    }
