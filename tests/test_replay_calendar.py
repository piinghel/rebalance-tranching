"""A registry outage must not erase a replay's local completion record."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("replay_fails", [False, True])
def test_replay_keeps_local_outcome_when_registry_update_fails(
    tmp_path, monkeypatch, replay_fails
):
    path = Path(__file__).resolve().parents[1] / "scripts/replay_calendar.py"
    spec = importlib.util.spec_from_file_location("replay_calendar", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def register(args, record, revision=0):
        if revision:
            raise RuntimeError("Registry unavailable")
        return {"revision": 1}

    def replay(args):
        if replay_fails:
            raise ValueError("Invalid research inputs")

    monkeypatch.setattr(runner, "register", register)
    monkeypatch.setattr(runner, "replay", replay)
    source = tmp_path / "input"
    source.write_text("frozen fixture")
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--base",
            str(source),
            "--panel",
            str(source),
            "--predictions",
            str(source),
            "--allocation-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--registry",
            "unused-registry",
            "--database",
            str(tmp_path / "registry.sqlite"),
            "--weekday",
            "5",
            "--offset",
            "0",
            "--calibration",
            "1.18",
        ],
    )
    with pytest.raises(RuntimeError, match="Registry unavailable"):
        runner.main()
    record = json.loads((output / "execution.json").read_text())
    assert record["status"] == ("failed" if replay_fails else "completed")
    assert record["provenance"]["finished_at"] != "unknown"
    if replay_fails:
        assert "Invalid research inputs" in record["failure_reason"]
