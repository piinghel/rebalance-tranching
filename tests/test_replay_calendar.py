"""A registry outage must not erase a replay's local completion record."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
def test_replay_keeps_local_outcome_when_registry_update_fails(
    tmp_path, monkeypatch, outcome
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
        if outcome == "failed":
            raise ValueError("Invalid research inputs")
        if outcome == "interrupted":
            raise KeyboardInterrupt

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
            "--experiment-id",
            "rebalancing:exp:matched-specification",
            "--run-prefix",
            "rebalancing:matched:run",
        ],
    )
    with pytest.raises(RuntimeError, match="Registry unavailable"):
        runner.main()
    record = json.loads((output / "execution.json").read_text())
    assert record["status"] == outcome
    assert record["provenance"]["finished_at"] != "unknown"
    assert record["id"] == "rebalancing:matched:run:w5-o0-a1"
    assert record["experiment_id"] == "rebalancing:exp:matched-specification"
    assert record["evidence"][0]["record_id"] == record["experiment_id"]
    if outcome == "failed":
        assert "Invalid research inputs" in record["failure_reason"]
    if outcome == "interrupted":
        assert "interrupted" in record["failure_reason"]
