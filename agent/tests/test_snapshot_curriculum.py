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
        "post_combat_kills": 5,
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


def test_snapshot_builder_selects_post_combat_exit_route(tmp_path):
    trajectory = tmp_path / "post-combat.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=False, shootable=False, kills=4),
            _trajectory_row(1, visible=True, shootable=True, kills=5),
            _trajectory_row(2, visible=False, shootable=False, kills=5),
            _trajectory_row(
                3,
                visible=False,
                shootable=False,
                kills=5,
                route_exit=True,
                route_line_id=330,
            ),
        ],
    )

    manifest = build_snapshot_curriculum_from_trajectory(
        trajectory,
        output_path=tmp_path / "post-combat-snapshot.json",
        name="e1m1-post-combat",
        auto_selectors=["post-combat", "post-combat-exit-route"],
        snapshot_dir=Path("snapshots"),
        save_slot_base=7,
    )

    assert [stage["evidence"]["source_record_index"] for stage in manifest["stages"]] == [
        2,
        3,
    ]
    post_combat, exit_route = manifest["stages"]
    assert post_combat["name"] == "0002-post-combat_snapshot"
    assert post_combat["expected_state"]["kills"] == 5
    assert post_combat["snapshot"]["slot"] == 7
    assert exit_route["name"] == "0003-post-combat-exit-route_snapshot"
    assert exit_route["expected_state"]["route_waypoint_exit"] is True
    assert exit_route["expected_state"]["route_waypoint_line_id"] == 330
    assert exit_route["snapshot"]["ref"] == "save_slot:8"


def test_snapshot_builder_selects_post_combat_exit_route_from_decision_line(tmp_path):
    trajectory = tmp_path / "post-combat-decision-exit.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=False, shootable=False, kills=4),
            _trajectory_row(1, visible=False, shootable=False, kills=5),
            _trajectory_row(
                2,
                visible=False,
                shootable=False,
                kills=5,
                route_exit=False,
                route_line_id=195,
                skill="approach_progression_line",
                decision_use_line={"line_id": 330, "special": 11, "distance": 768.0},
            ),
        ],
    )

    manifest = build_snapshot_curriculum_from_trajectory(
        trajectory,
        output_path=tmp_path / "post-combat-decision-exit.json",
        name="e1m1-post-combat-decision-exit",
        auto_selectors=["post-combat-exit-route"],
        snapshot_dir=Path("snapshots"),
        save_slot_base=8,
    )

    assert manifest["stages"][0]["evidence"]["source_record_index"] == 2
    assert manifest["stages"][0]["expected_state"]["route_waypoint_exit"] is False
    assert manifest["stages"][0]["expected_state"]["route_waypoint_line_id"] == 195
    assert manifest["stages"][0]["expected_state"]["exit_use_line_id"] == 330
    assert manifest["stages"][0]["expected_state"]["exit_use_line_special"] == 11
    assert (
        manifest["stages"][0]["expected_state"]["exit_use_line_max_distance_units"]
        == 900.0
    )


def test_snapshot_builder_rejects_exit_control_without_exit_route_waypoint(tmp_path):
    trajectory = tmp_path / "post-combat-exit-control.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=False, shootable=False, kills=4),
            _trajectory_row(1, visible=False, shootable=False, kills=5),
            _trajectory_row(
                2,
                visible=False,
                shootable=False,
                kills=5,
                skill="use_exit_assist_door",
            ),
        ],
    )

    with pytest.raises(ValueError, match="post-combat-exit-route"):
        build_snapshot_curriculum_from_trajectory(
            trajectory,
            output_path=tmp_path / "post-combat-exit-control.json",
            name="e1m1-post-combat-exit-control",
            auto_selectors=["post-combat-exit-route"],
            snapshot_dir=Path("snapshots"),
            save_slot_base=8,
        )


def test_snapshot_builder_rejects_exit_route_without_line_id(tmp_path):
    trajectory = tmp_path / "post-combat-exit-no-line.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=False, shootable=False, kills=5),
            _trajectory_row(
                1,
                visible=False,
                shootable=False,
                kills=5,
                route_exit=True,
                route_line_id=0,
            ),
        ],
    )

    with pytest.raises(ValueError, match="post-combat-exit-route"):
        build_snapshot_curriculum_from_trajectory(
            trajectory,
            output_path=tmp_path / "post-combat-exit-no-line.json",
            name="e1m1-post-combat-exit-no-line",
            auto_selectors=["post-combat-exit-route"],
            snapshot_dir=Path("snapshots"),
            save_slot_base=8,
        )


def test_snapshot_builder_respects_post_combat_kill_threshold(tmp_path):
    trajectory = tmp_path / "post-combat-threshold.jsonl"
    _write_jsonl(
        trajectory,
        [
            _trajectory_row(0, visible=False, shootable=False, kills=5),
            _trajectory_row(1, visible=False, shootable=False, kills=6),
        ],
    )

    manifest = build_snapshot_curriculum_from_trajectory(
        trajectory,
        output_path=tmp_path / "post-combat-threshold.json",
        name="e1m1-post-combat-threshold",
        auto_selectors=["post-combat"],
        snapshot_dir=Path("snapshots"),
        post_combat_kills=6,
    )

    assert manifest["source"]["selection"]["post_combat_kills"] == 6
    assert manifest["stages"][0]["evidence"]["source_record_index"] == 1


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


def test_native_snapshot_capture_tracker_matches_post_combat_selectors():
    tracker = SnapshotMilestoneTracker(
        ("post-combat", "post-combat-exit-route"),
        post_combat_kills=6,
    )

    below_threshold = tracker.observe(
        _trajectory_row(0, visible=False, shootable=False, kills=5)
    )
    visible_combat = tracker.observe(
        _trajectory_row(1, visible=True, shootable=True, kills=6)
    )
    post_combat = tracker.observe(
        _trajectory_row(2, visible=False, shootable=False, kills=6)
    )
    tracker.mark_captured(post_combat)
    exit_control = tracker.observe(
        _trajectory_row(
            3,
            visible=False,
            shootable=False,
            kills=6,
            use_line_specials=[11],
        )
    )
    exit_route = tracker.observe(
        _trajectory_row(
            4,
            visible=False,
            shootable=False,
            kills=6,
            route_exit=True,
            route_line_id=330,
        )
    )
    tracker.mark_captured(exit_route)

    assert below_threshold == []
    assert visible_combat == []
    assert post_combat == ["post-combat"]
    assert exit_control == []
    assert exit_route == ["post-combat-exit-route"]
    assert tracker.complete is True


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


def test_native_snapshot_load_verification_checks_exit_route_waypoint():
    expected = {
        "episode": 1,
        "map": 1,
        "route_waypoint_exit": True,
        "route_waypoint_line_id": 330,
    }
    actual = {
        "episode": 1,
        "map": 1,
        "navigation": {
            "route_waypoint": {
                "exit": True,
                "line": {"line_id": 330},
            }
        },
    }
    wrong_route = {
        "episode": 1,
        "map": 1,
        "navigation": {
            "route_waypoint": {
                "exit": False,
                "line": {"line_id": 325},
            }
        },
    }

    ok = _verify_snapshot_restored_state(
        actual=actual,
        expected=expected,
        raw_state=_state(combat=False),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )
    bad = _verify_snapshot_restored_state(
        actual=wrong_route,
        expected=expected,
        raw_state=_state(combat=False),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )

    assert ok["valid"] is True
    assert "route_waypoint_exit" in ok["compared_fields"]
    assert "route_waypoint_line_id" in ok["compared_fields"]
    assert bad["valid"] is False
    assert any("route_waypoint_exit" in error for error in bad["errors"])


def test_native_snapshot_load_verification_uses_raw_route_waypoint_fallback():
    expected = {
        "episode": 1,
        "map": 1,
        "route_waypoint_exit": True,
        "route_waypoint_line_id": 330,
    }
    actual = {"episode": 1, "map": 1}

    ok = _verify_snapshot_restored_state(
        actual=actual,
        expected=expected,
        raw_state=_state(combat=False, route_exit=True, route_line_id=330),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )
    bad = _verify_snapshot_restored_state(
        actual=actual,
        expected=expected,
        raw_state=_state(combat=False, route_exit=True, route_line_id=325),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )

    assert ok["valid"] is True
    assert bad["valid"] is False
    assert any("route_waypoint_line_id" in error for error in bad["errors"])


def test_native_snapshot_load_verification_checks_exit_use_line():
    expected = {
        "episode": 1,
        "map": 1,
        "exit_use_line_id": 330,
        "exit_use_line_special": 11,
        "exit_use_line_max_distance_units": 896.0,
    }
    actual = {
        "episode": 1,
        "map": 1,
        "navigation": {
            "use_lines": [
                {
                    "line_id": 330,
                    "special": 11,
                    "nearest_distance_fp": 512 * 65536,
                }
            ]
        },
    }
    wrong_line = {
        "episode": 1,
        "map": 1,
        "navigation": {
            "use_lines": [
                {
                    "line_id": 330,
                    "special": 88,
                    "nearest_distance_fp": 512 * 65536,
                }
            ]
        },
    }

    ok = _verify_snapshot_restored_state(
        actual=actual,
        expected=expected,
        raw_state=_state(combat=False),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )
    bad = _verify_snapshot_restored_state(
        actual=wrong_line,
        expected=expected,
        raw_state=_state(combat=False),
        enabled=True,
        tick_tolerance=35,
        verify_stream_tick=False,
        position_tolerance_fp=160 * 65536,
    )

    assert ok["valid"] is True
    assert "exit_use_line_id" in ok["compared_fields"]
    assert "exit_use_line_special" in ok["compared_fields"]
    assert "exit_use_line_max_distance_units" in ok["compared_fields"]
    assert bad["valid"] is False
    assert any("exit_use_line_special" in error for error in bad["errors"])


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


def _state(*, combat=False, route_exit: bool | None = None, route_line_id: int | None = None):
    line = SimpleNamespace(line_id=route_line_id) if route_line_id is not None else None
    route = (
        SimpleNamespace(exit=route_exit, line=line)
        if route_exit is not None or line is not None
        else None
    )
    return SimpleNamespace(
        enemies=[],
        combat=SimpleNamespace(has_shootable_target=combat, target_is_enemy=combat),
        navigation=SimpleNamespace(route_waypoint=route),
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
    kills: int | None = None,
    route_exit: bool = False,
    route_line_id: int = 0,
    use_line_specials: list[int] | None = None,
    decision_use_line: dict | None = None,
) -> dict:
    if target_is_enemy is None:
        target_is_enemy = bool(shootable)
    state_kills = kill_delta if kills is None else kills
    route_waypoint = (
        {
            "exit": route_exit,
            "line": {
                "line_id": route_line_id,
                "distance_fp": 128 * 65536,
                "nearest_distance_fp": 128 * 65536,
            },
        }
        if route_exit or route_line_id
        else {}
    )
    use_lines = [
        {
            "line_id": 1000 + offset,
            "special": special,
            "distance_fp": 128 * 65536,
            "nearest_distance_fp": 128 * 65536,
        }
        for offset, special in enumerate(use_line_specials or [])
    ]
    policy_decision = {
        "skill": skill,
        "visible_enemies": 1 if visible else 0,
        "combat": {
            "has_shootable_target": shootable,
            "target_is_enemy": target_is_enemy,
        },
    }
    if decision_use_line is not None:
        policy_decision["use_line"] = dict(decision_use_line)
    return {
        "index": index,
        "state": {
            "tick": 100 + index,
            "episode": 1,
            "map": 1,
            "health": 100,
            "armor": 0,
            "ammo_bullets": 50,
            "kills": state_kills,
            "items": 0,
            "position_fp": [(100 + index) * 65536, -200 * 65536, 0],
            "navigation": {
                "route_waypoint": route_waypoint,
                "use_lines": use_lines,
            },
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
            "policy_decision": policy_decision
        },
    }
