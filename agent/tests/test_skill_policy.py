import json
from types import SimpleNamespace

from restfuldoom_agent.brain import (
    AgentMemory,
    BrainPolicy,
    BrainPolicyParams,
    export_training_job,
    import_training_job,
    train_skill_policy_from_memory,
)
from restfuldoom_agent.skill_policy import (
    FEATURE_NAMES,
    SKILL_POLICY_SCHEMA,
    SkillPolicyModel,
    SkillPolicyTrainConfig,
    features_from_record,
    train_skill_policy,
)


def test_skill_policy_trains_and_predicts_from_trajectory(tmp_path):
    trajectory = tmp_path / "skill.jsonl"
    _write_skill_rows(trajectory)
    checkpoint = tmp_path / "skill-policy.json"

    summary = train_skill_policy(
        [trajectory],
        checkpoint,
        config=SkillPolicyTrainConfig(
            epochs=20,
            learning_rate=0.15,
            min_count=2,
            max_samples=100,
            seed=3,
        ),
    )

    assert summary["schema"] == SKILL_POLICY_SCHEMA
    assert summary["class_count"] == 2
    assert summary["eval_accuracy"] >= 0.5

    model = SkillPolicyModel.load(checkpoint)
    fire_record = _row("fire_on_shootable_target", health=100, enemy_count=1, combat=True)
    press_record = _row("press_exit_switch", health=93, enemy_count=0, line_special=11)

    assert model.predict_record(fire_record)["skill"] == "fire_on_shootable_target"
    assert model.predict_record(press_record)["skill"] == "press_exit_switch"


def test_train_skill_policy_from_memory_updates_export_manifest(tmp_path):
    trajectory = tmp_path / "brain-success.jsonl"
    _write_skill_rows(trajectory)
    memory_path = tmp_path / "agent_memory" / "e1m1.json"
    memory = AgentMemory.load(memory_path)
    memory.data["episodes"].append(
        {
            "trajectory_jsonl": str(trajectory),
            "success": True,
            "level_completed": True,
            "kill_delta": 1,
        }
    )
    memory.save()
    checkpoint = tmp_path / "agent_models" / "skill-policy.json"

    train_summary = train_skill_policy_from_memory(
        checkpoint,
        memory_path=memory_path,
        config=SkillPolicyTrainConfig(epochs=4, min_count=2, max_samples=100),
    )

    assert train_summary["checkpoint_path"] == str(checkpoint)
    learned = AgentMemory.load(memory_path).data["learned_policy"]
    assert learned["schema"] == SKILL_POLICY_SCHEMA
    assert learned["checkpoint_path"] == str(checkpoint)

    bundle = tmp_path / "training.tar.gz"
    export_summary = export_training_job(
        bundle,
        memory_path=memory_path,
        notes_path=tmp_path / "missing-notes.md",
    )

    assert export_summary["model_checkpoint_count"] == 1
    import tarfile

    with tarfile.open(bundle, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        manifest = json.loads(archive.extractfile("manifest.json").read().decode())

    assert "agent_models/skill-policy.json" in names
    assert manifest["learned_policy"]["checkpoint_path"] == str(checkpoint)
    assert manifest["model_checkpoints"] == ["agent_models/skill-policy.json"]


def test_export_training_job_includes_ppo_checkpoint(tmp_path):
    memory_path = tmp_path / "agent_memory" / "e1m1.json"
    memory = AgentMemory.load(memory_path)
    checkpoint = tmp_path / "agent_models" / "ppo" / "ppo.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake torch checkpoint")
    buffer = tmp_path / "trajectories" / "ppo" / "ppo-buffer.jsonl"
    buffer.parent.mkdir(parents=True)
    buffer.write_text('{"schema":"restfuldoom.ppo_rollout.v1"}\n')
    trajectory = tmp_path / "trajectories" / "brain-success.jsonl"
    trajectory.parent.mkdir(exist_ok=True)
    trajectory.write_text('{"schema":"restfuldoom.trajectory.v1"}\n')
    snapshot = tmp_path / "snapshots" / "first-contact.snap"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"fake snapshot")
    memory.data["episodes"].append(
        {
            "trajectory_jsonl": str(trajectory),
            "success": True,
            "level_completed": True,
            "kill_delta": 1,
        }
    )
    memory.data["ppo_policy"] = {
        "schema": "restfuldoom.ppo_policy.v1",
        "checkpoint_path": str(checkpoint),
        "buffer_path": str(buffer),
        "reward_config": {"goal_preset": "combat"},
        "eval_history": [{"policy_id": "ppo", "mean_kills": 1.0}],
        "rollout_summary": {"snapshot_restore_count": 1},
        "curriculum": {
            "schema": "restfuldoom.ppo_curriculum.v1",
            "snapshot_curriculum": {
                "schema": "restfuldoom.snapshot_curriculum.v1",
                "name": "progressed-e1m1",
            },
            "stages": [
                {
                    "name": "first_contact_snapshot",
                    "snapshot": {
                        "id": "snap-1",
                        "path": str(snapshot),
                    },
                }
            ],
        },
    }
    memory.save()

    bundle = tmp_path / "training.tar.gz"
    export_training_job(
        bundle,
        memory_path=memory_path,
        notes_path=tmp_path / "missing-notes.md",
    )

    import tarfile

    with tarfile.open(bundle, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        manifest = json.loads(archive.extractfile("manifest.json").read().decode())

    assert "agent_models/ppo/ppo.pt" in names
    assert "trajectories/ppo/ppo-buffer.jsonl" in names
    assert "trajectories/brain-success.jsonl" in names
    assert "snapshots/first-contact.snap" in names
    assert manifest["ppo_checkpoints"] == ["agent_models/ppo/ppo.pt"]
    assert manifest["ppo_checkpoint_artifacts"] == [
        {
            "source_path": str(checkpoint),
            "bundle_path": "agent_models/ppo/ppo.pt",
        }
    ]
    assert manifest["ppo_rollout_buffers"] == ["trajectories/ppo/ppo-buffer.jsonl"]
    assert manifest["ppo_rollout_buffer_artifacts"] == [
        {
            "source_path": str(buffer),
            "bundle_path": "trajectories/ppo/ppo-buffer.jsonl",
        }
    ]
    assert manifest["ppo_policy"]["checkpoint_path"] == str(checkpoint)
    assert manifest["observation_schema"]["schema"] == "restfuldoom.observation.v1"
    assert manifest["action_schema"]["schema"] == "restfuldoom.skill_action.v1"
    assert manifest["reward_config"] == {"goal_preset": "combat"}
    assert manifest["eval_history"] == [{"policy_id": "ppo", "mean_kills": 1.0}]
    assert manifest["snapshot_curriculum"]["name"] == "progressed-e1m1"
    assert manifest["snapshot_restore_context"]["rollout_summary"]["snapshot_restore_count"] == 1
    assert manifest["trajectory_artifacts"] == [
        {
            "source_path": str(trajectory),
            "bundle_path": "trajectories/brain-success.jsonl",
        }
    ]
    assert manifest["snapshot_artifacts"] == [
        {
            "source_path": str(snapshot),
            "bundle_path": "snapshots/first-contact.snap",
        }
    ]

    import_summary = import_training_job(bundle, destination=tmp_path / "imported")
    imported_memory = json.loads(
        (tmp_path / "imported" / "agent_memory" / "e1m1.json").read_text()
    )

    assert import_summary["ppo_rollout_buffer_count"] == 1
    assert import_summary["path_rewrite_count"] >= 4
    assert imported_memory["ppo_policy"]["checkpoint_path"] == str(
        tmp_path / "imported" / "agent_models" / "ppo" / "ppo.pt"
    )
    assert imported_memory["ppo_policy"]["buffer_path"] == str(
        tmp_path / "imported" / "trajectories" / "ppo" / "ppo-buffer.jsonl"
    )
    assert imported_memory["episodes"][0]["trajectory_jsonl"] == str(
        tmp_path / "imported" / "trajectories" / "brain-success.jsonl"
    )
    imported_snapshot = imported_memory["ppo_policy"]["curriculum"]["stages"][0][
        "snapshot"
    ]
    assert imported_snapshot["path"] == str(
        tmp_path / "imported" / "snapshots" / "first-contact.snap"
    )
    assert imported_memory["training_job_import"]["schema"] == (
        "restfuldoom.training_job_import.v1"
    )


def test_export_training_job_flags_unbundled_native_snapshot_slots(tmp_path):
    memory_path = tmp_path / "agent_memory" / "e1m1.json"
    memory = AgentMemory.load(memory_path)
    memory.data["ppo_policy"] = {
        "schema": "restfuldoom.ppo_policy.v1",
        "curriculum": {
            "schema": "restfuldoom.ppo_curriculum.v1",
            "stages": [
                {
                    "name": "first_visible_native_slot",
                    "reset_mode": "snapshot",
                    "snapshot": {
                        "id": "slot-3",
                        "slot": 3,
                        "ref": "save_slot:3",
                    },
                }
            ],
        },
    }
    memory.save()

    bundle = tmp_path / "training.tar.gz"
    export_summary = export_training_job(
        bundle,
        memory_path=memory_path,
        notes_path=tmp_path / "missing-notes.md",
    )

    import tarfile

    with tarfile.open(bundle, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        manifest = json.loads(archive.extractfile("manifest.json").read().decode())

    assert export_summary["unbundled_snapshot_slot_count"] == 1
    assert "snapshots/agentdoom3.dsg" not in names
    assert manifest["snapshot_artifacts"] == []
    assert manifest["unbundled_snapshot_slots"] == [
        {
            "schema": "restfuldoom.unbundled_snapshot_slot.v1",
            "stage": "first_visible_native_slot",
            "snapshot_id": "slot-3",
            "slot": 3,
            "ref": "save_slot:3",
            "reason": (
                "native Doom save slots live in the game server save directory "
                "and are not bundled"
            ),
        }
    ]


def test_features_include_position_and_facing_from_record():
    record = _row("fire_on_shootable_target", health=100, enemy_count=1, combat=True)

    vector = features_from_record(record)
    values = dict(zip(FEATURE_NAMES, vector))

    assert values["x_units_norm"] != 0.0
    assert values["y_units_norm"] != 0.0
    assert values["angle_sin"] == 0.0
    assert values["angle_cos"] == 1.0


def test_brain_policy_records_learned_skill_prediction(tmp_path):
    trajectory = tmp_path / "skill.jsonl"
    _write_skill_rows(trajectory)
    checkpoint = tmp_path / "skill-policy.json"
    train_skill_policy(
        [trajectory],
        checkpoint,
        config=SkillPolicyTrainConfig(epochs=15, learning_rate=0.15, min_count=2),
    )
    model = SkillPolicyModel.load(checkpoint)
    memory = AgentMemory.load(tmp_path / "memory.json")
    policy = BrainPolicy(
        memory=memory,
        params=BrainPolicyParams(),
        policy_id="test",
        skill_model=model,
    )

    state = SimpleNamespace(
        tick=1,
        player=SimpleNamespace(
            object=SimpleNamespace(
                position=SimpleNamespace(x_fp=0, y_fp=0, z_fp=0),
                angle_degrees=0,
            ),
            health=100,
            ammo=SimpleNamespace(bullets=50),
            kills=0,
            items=0,
            secrets=0,
        ),
        enemies=[],
        level=SimpleNamespace(episode=1, map=1),
        navigation=SimpleNamespace(
            forward_open=True,
            back_open=True,
            left_open=True,
            right_open=True,
            use_line_ahead=False,
            front_blocking_line_special=0,
            front_block_distance_fp=96 * 65536,
            probe_distance_fp=96 * 65536,
            direction_probes=[],
            use_lines=[],
        ),
        combat=SimpleNamespace(
            has_shootable_target=False,
            target_id=0,
            target_health=0,
            target_distance_fp=0,
            aim_slope_fp=0,
            range_fp=0,
            target_is_enemy=False,
        ),
    )

    import asyncio

    asyncio.run(policy.next_action(state))

    assert "learned_skill_prediction" in policy.last_decision
    assert policy.last_decision["learned_skill_prediction"]["skill"] in {
        "fire_on_shootable_target",
        "press_exit_switch",
    }


def _write_skill_rows(path):
    rows = []
    for _ in range(8):
        rows.append(_row("fire_on_shootable_target", health=100, enemy_count=1, combat=True))
        rows.append(_row("press_exit_switch", health=93, enemy_count=0, line_special=11))
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _row(skill, *, health, enemy_count, combat=False, line_special=0):
    use_line = {}
    if line_special:
        use_line = {
            "line_id": 330,
            "special": line_special,
            "distance": 16.0,
            "angle_delta": 0.0,
            "side": 0,
            "front_distance": 16.0,
        }
    enemy = {}
    if enemy_count:
        enemy = {
            "id": 7,
            "distance": 256.0,
            "angle_delta": 0.0,
            "health": 20,
            "threat": 2.0,
        }
    return {
        "state": {
            "health": health,
            "ammo_bullets": 50,
            "kills": 0,
            "items": 0,
            "enemy_count": enemy_count,
            "position_fp": [1024 * 65536, -512 * 65536, 0],
            "angle_degrees": 0,
        },
        "metadata": {
            "policy_decision": {
                "skill": skill,
                "health": health,
                "ammo_bullets": 50,
                "kills": 0,
                "items": 0,
                "visible_enemies": enemy_count,
                "known_enemies": enemy_count,
                "remembered_enemies": 0,
                "enemy_count": enemy_count,
                "enemy": enemy,
                "use_line": use_line,
                "navigation": {
                    "forward_open": not line_special,
                    "back_open": True,
                    "left_open": True,
                    "right_open": True,
                    "use_line_ahead": bool(line_special),
                    "front_block_distance_fp": 16 * 65536,
                    "front_blocking_line_special": line_special,
                    "direction_probes": [],
                    "use_lines": [use_line] if use_line else [],
                },
                "combat": {
                    "has_shootable_target": combat,
                    "target_is_enemy": combat,
                    "target_distance_fp": 256 * 65536 if combat else 0,
                },
            }
        },
    }
