"""Regression tests for the security remediation."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

import restfuldoom_agent.ppo as ppo_module
from restfuldoom_agent.brain import import_training_job
from restfuldoom_agent.ppo import PPOConfig, PPOTrainer
from restfuldoom_agent.snapshot_builder import (
    _capture_uses_shell,
    _render_capture_command,
    _run_capture_command,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stage(**overrides):
    stage = {
        "name": "stage",
        "snapshot": {
            "id": "snap-1",
            "path": "snapshots/snap-1.snap",
            "ref": "save_slot:1",
            "microvm_id": "vm-1",
            "capsule": "agent-doom",
        },
        "expected_state": {"tick": 1234},
        "evidence": {"source_record_index": 7},
    }
    stage.update(overrides)
    return stage


def test_capture_command_quotes_tick_and_record_index():
    rendered = _render_capture_command(
        "echo {tick} {record_index} {tick_sh} {record_index_sh}",
        _stage(),
        trajectory=Path("run.jsonl"),
    )

    assert "1234 7 1234 7" in rendered


def test_capture_command_rejects_non_integer_tick():
    with pytest.raises(ValueError, match="tick"):
        _render_capture_command(
            "echo {tick}",
            _stage(expected_state={"tick": "1; touch /tmp/pwned"}),
            trajectory=Path("run.jsonl"),
        )


def test_capture_command_rejects_non_integer_record_index():
    with pytest.raises(ValueError, match="record_index"):
        _render_capture_command(
            "echo {record_index}",
            _stage(evidence={"source_record_index": "$(id)"}),
            trajectory=Path("run.jsonl"),
        )


def test_capture_uses_shell_only_when_needed():
    assert not _capture_uses_shell("echo --tick 12")
    assert _capture_uses_shell("echo hi > out.txt")
    assert _capture_uses_shell("echo hi | tee out.txt")


def test_capture_command_runs_argv_without_shell(tmp_path):
    result = _run_capture_command(
        "printf '%s' {tick}",
        _stage(),
        trajectory=tmp_path / "run.jsonl",
        cwd=tmp_path,
        timeout_seconds=10,
    )

    assert result["returncode"] == 0
    assert result["stdout_tail"] == "1234"


def test_training_bundle_rejects_memory_path_escape(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    manifest = {
        "schema": "restfuldoom.training_job.v1",
        "memory_path": "../escaped.json",
    }
    with tarfile.open(bundle, "w:gz") as archive:
        payload = json.dumps(manifest).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="escapes"):
        import_training_job(bundle, destination=tmp_path / "imported")


def test_training_bundle_rejects_symlink_member(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    manifest = {
        "schema": "restfuldoom.training_job.v1",
        "memory_path": "agent_memory/e1m1.json",
    }
    with tarfile.open(bundle, "w:gz") as archive:
        payload = json.dumps(manifest).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("agent_memory/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)

    with pytest.raises(ValueError, match="link member"):
        import_training_job(bundle, destination=tmp_path / "imported")


def test_load_checkpoint_requests_weights_only(monkeypatch, tmp_path):
    trainer = PPOTrainer(obs_dim=2, action_dim=2, config=PPOConfig())
    checkpoint = trainer.save_checkpoint(tmp_path / "ppo.pt")

    seen: dict[str, object] = {}
    real_load = ppo_module.torch.load

    def recording_load(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(ppo_module.torch, "load", recording_load)
    loaded = PPOTrainer.load_checkpoint(checkpoint)

    assert seen.get("weights_only") is True
    assert loaded.update_index == trainer.update_index


def test_committed_checkpoints_load_with_weights_only():
    paths = sorted((REPO_ROOT / "agent_models" / "ppo").glob("*.pt"))
    if not paths:
        pytest.skip("no committed PPO checkpoints in this checkout")
    import torch

    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["schema"] == ppo_module.PPO_CHECKPOINT_SCHEMA
