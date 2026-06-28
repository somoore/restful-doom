import json
import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from restfuldoom_agent.snapshot_capture import (
    SnapshotCaptureConfig,
    SnapshotMilestoneTracker,
    _attempt_trajectory_path,
    _capture_stage,
    _main as snapshot_capture_main,
)
from restfuldoom_agent.env import _verify_snapshot_restored_state
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
                target_is_enemy=False,
                skill="open_use_line",
            ),
            _trajectory_row(
                3,
                visible=True,
                shootable=True,
                target_is_enemy=True,
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
        auto_selectors=["first-enemy-shootable", "first-damage"],
        snapshot_dir=Path("snapshots"),
        save_slot_base=3,
    )

    assert manifest["schema"] == "restfuldoom.snapshot_curriculum.v1"
    assert manifest["source"]["selection"] == {
        "indexes": [1],
        "auto": ["first-enemy-shootable", "first-damage"],
    }
    assert [stage["evidence"]["source_record_index"] for stage in manifest["stages"]] == [
        1,
        3,
    ]
    first, second = manifest["stages"]
    assert first["name"] == "0001-explicit_snapshot"
    assert first["reset_start"]["x_fp"] == 101 * 65536
    assert first["expected_state"]["visible_enemy"] is True
    assert first["expected_state"]["shootable_target"] is False
    assert first["expected_state"]["target_is_enemy"] is False
    assert first["snapshot"]["slot"] == 3
    assert first["snapshot"]["ref"] == "save_slot:3"
    assert second["evidence"]["selectors"] == ["first-enemy-shootable", "first-damage"]
    assert second["expected_state"]["shootable_target"] is True
    assert second["expected_state"]["target_is_enemy"] is True
    assert second["expected_state"]["damage_delta"] == 20
    assert second["snapshot"]["slot"] == 4

    validation = validate_snapshot_curriculum(tmp_path / "snapshot-curriculum.json")
    assert validation["valid"] is True
    assert validation["stage_count"] == 2
    assert validation["missing_artifacts"] == [
        "0001-explicit_snapshot",
        "0003-first-enemy-shootable_snapshot",
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


def test_snapshot_validation_flags_native_slots_when_requiring_artifacts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "restfuldoom.snapshot_curriculum.v1",
                "name": "native-slot",
                "stages": [
                    {
                        "name": "slot_snapshot",
                        "snapshot": {
                            "id": "slot-3",
                            "slot": 3,
                            "ref": "save_slot:3",
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
    assert validation["stages"][0]["snapshot_slot"] == 3
    assert validation["missing_artifacts"] == []
    assert "portable only on the originating Doom server" in validation["errors"][0]


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


def test_native_snapshot_capture_tracker_groups_same_record_selectors():
    tracker = SnapshotMilestoneTracker(
        ("first-visible", "first-enemy-shootable", "first-damage")
    )

    first = tracker.observe(_trajectory_row(0, visible=False, shootable=False))
    non_enemy = tracker.observe(
        _trajectory_row(
            1,
            visible=True,
            shootable=True,
            target_is_enemy=False,
        )
    )
    tracker.mark_captured(non_enemy)
    second = tracker.observe(
        _trajectory_row(
            2,
            visible=True,
            shootable=True,
            target_is_enemy=True,
            damage_delta=20,
        )
    )
    tracker.mark_captured(second)
    third = tracker.observe(
        _trajectory_row(
            2,
            visible=True,
            shootable=True,
            target_is_enemy=True,
            damage_delta=20,
        )
    )

    assert first == []
    assert non_enemy == ["first-visible"]
    assert second == ["first-enemy-shootable", "first-damage"]
    assert tracker.complete is True
    assert third == []


def test_snapshot_builder_first_enemy_shootable_skips_non_enemy_target(tmp_path):
    trajectory = tmp_path / "run.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=True, shootable=True, target_is_enemy=False),
            _trajectory_row(1, visible=True, shootable=True, target_is_enemy=True),
        ],
    )

    manifest = build_snapshot_curriculum_from_trajectory(
        trajectory,
        output_path=tmp_path / "snapshot-curriculum.json",
        name="enemy-shootable",
        auto_selectors=["first-shootable", "first-enemy-shootable"],
        save_slot_base=1,
    )

    assert [stage["evidence"]["source_record_index"] for stage in manifest["stages"]] == [
        0,
        1,
    ]
    assert manifest["stages"][0]["evidence"]["selectors"] == ["first-shootable"]
    assert manifest["stages"][0]["expected_state"]["target_is_enemy"] is False
    assert manifest["stages"][1]["evidence"]["selectors"] == ["first-enemy-shootable"]
    assert manifest["stages"][1]["expected_state"]["target_is_enemy"] is True


def test_native_snapshot_capture_stage_queues_save_slot(tmp_path):
    record = _trajectory_row(
        2,
        visible=True,
        shootable=True,
        skill="fire",
        damage_delta=20,
    )
    config = SnapshotCaptureConfig(
        output_path=tmp_path / "manifest.json",
        name="native-capture",
        snapshot_dir=tmp_path / "snapshots",
        save_slot_base=4,
    )
    client = _FakeSnapshotClient()

    stage = asyncio.run(
        _capture_stage(
            client,
            config,
            record,
            selectors=["first-shootable", "first-damage"],
            line_index=2,
            order=0,
            trajectory=tmp_path / "run.jsonl",
            run_id="capture-test",
        )
    )

    assert client.save_requests == [
        {
            "slot": 4,
            "description": "first-shootable-2",
            "run_id": "capture-test-slot-4",
        }
    ]
    assert stage["snapshot"]["slot"] == 4
    assert stage["snapshot"]["ref"] == "save_slot:4"
    assert stage["capture"]["method"] == "grpc_save_snapshot"
    assert stage["capture"]["save_queued"] is True
    assert stage["capture"]["attempt"] == 1
    assert stage["evidence"]["selectors"] == ["first-shootable", "first-damage"]
    assert stage["evidence"]["capture_attempt"] == 1
    assert stage["evidence"]["attempt_record_index"] == 2
    assert stage["expected_state"]["damage_delta"] == 20


def test_native_snapshot_capture_stage_records_attempt_metadata(tmp_path):
    record = _trajectory_row(
        42,
        visible=True,
        shootable=False,
        skill="seek_enemy",
    )
    config = SnapshotCaptureConfig(
        output_path=tmp_path / "manifest.json",
        name="native-capture",
        snapshot_dir=tmp_path / "snapshots",
        save_slot_base=4,
    )
    client = _FakeSnapshotClient()

    stage = asyncio.run(
        _capture_stage(
            client,
            config,
            record,
            selectors=["first-visible"],
            line_index=42,
            order=2,
            trajectory=tmp_path / "run-attempt-002.jsonl",
            run_id="capture-test",
            attempt=2,
            global_record_index=1042,
        )
    )

    assert stage["snapshot"]["slot"] == 6
    assert stage["evidence"]["capture_attempt"] == 2
    assert stage["evidence"]["attempt_record_index"] == 42
    assert stage["evidence"]["global_record_index"] == 1042
    assert stage["capture"]["attempt"] == 2
    assert stage["capture"]["attempt_record_index"] == 42


def test_snapshot_capture_attempt_trajectory_paths_do_not_overwrite_base():
    path = Path("trajectories/snapshot-capture.jsonl")

    assert _attempt_trajectory_path(path, attempt=1, attempts=1) == path
    assert _attempt_trajectory_path(path, attempt=2, attempts=3) == Path(
        "trajectories/snapshot-capture-attempt-002.jsonl"
    )
    assert _attempt_trajectory_path(Path("capture"), attempt=1, attempts=2) == Path(
        "capture-attempt-001.jsonl"
    )
    assert _attempt_trajectory_path(None, attempt=1, attempts=2) is None


def test_snapshot_capture_rejects_native_slot_range_exhaustion(tmp_path, capsys):
    with pytest.raises(SystemExit):
        snapshot_capture_main(
            [
                "--endpoint",
                "127.0.0.1:50051",
                "--output",
                str(tmp_path / "manifest.json"),
                "--save-slot-base",
                "8",
                "--auto",
                "first-visible",
                "--auto",
                "first-enemy-shootable",
                "--auto",
                "first-damage",
            ]
        )

    err = capsys.readouterr().err
    assert "must fit native slots 0..9" in err
    assert "requested range 8..10" in err


def test_native_snapshot_load_verification_matches_progressed_state():
    expected = {
        "episode": 1,
        "map": 1,
        "tick": 100,
        "level_time": 409,
        "position_fp": [100 * 65536, -200 * 65536, 0],
        "shootable_target": True,
        "target_is_enemy": False,
    }
    observed = {
        "episode": 1,
        "map": 1,
        "tick": 5000,
        "level_time": 410,
        "position_fp": [101 * 65536, -200 * 65536, 0],
        "combat": {"has_shootable_target": True, "target_is_enemy": False},
    }
    wrong_map = {**observed, "map": 2}
    wrong_enemy_flag = {
        **observed,
        "combat": {"has_shootable_target": True, "target_is_enemy": True},
    }

    assert _verify_snapshot_restored_state(
        actual=observed,
        expected=expected,
        raw_state=_state(combat=False),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )["valid"]
    assert not _verify_snapshot_restored_state(
        actual=wrong_map,
        expected=expected,
        raw_state=_state(combat=True),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )["valid"]
    assert not _verify_snapshot_restored_state(
        actual=wrong_enemy_flag,
        expected=expected,
        raw_state=_state(combat=False),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )["valid"]


class _FakeSnapshotClient:
    def __init__(self):
        self.save_requests = []

    async def save_snapshot(self, *, slot, description="", run_id=""):
        self.save_requests.append(
            {"slot": slot, "description": description, "run_id": run_id}
        )
        return SimpleNamespace(
            accepted=True,
            message="queued",
            slot=slot,
            save_queued=True,
            load_queued=False,
        )


def _state(*, combat=False):
    return SimpleNamespace(
        enemies=[],
        combat=SimpleNamespace(has_shootable_target=combat, target_is_enemy=combat),
    )


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
    target_is_enemy: bool | None = None,
    skill: str = "route_progression",
    damage_delta: int = 0,
    kill_delta: int = 0,
) -> dict:
    if target_is_enemy is None:
        target_is_enemy = bool(shootable)
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
                "target_is_enemy": target_is_enemy,
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
                "combat": {
                    "has_shootable_target": shootable,
                    "target_is_enemy": target_is_enemy,
                },
            }
        },
    }
