import json
import sys

import pytest
from scripts.batch_vidosc import run_stage


def stage(tmp_path):
    source = tmp_path / "input"
    source.write_text("v1")
    output = tmp_path / "output"
    return {"name": "test", "command": [sys.executable, "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('result')", str(output)],
        "inputs": [str(source)], "outputs": [str(output)], "receipt": str(tmp_path / "done.json")}


def test_plan_and_modal_gate_do_not_execute(tmp_path):
    s = stage(tmp_path)
    s["modal"] = True
    assert run_stage(s) == "planned"
    with pytest.raises(ValueError, match="Modal"):
        run_stage(s, execute=True)
    assert not (tmp_path / "output").exists()


def test_resume_and_changed_inputs(tmp_path):
    s = stage(tmp_path)
    assert run_stage(s, execute=True) == "completed"
    assert run_stage(s, execute=True) == "reused"
    (tmp_path / "input").write_text("v2")
    with pytest.raises(ValueError, match="Changed"):
        run_stage(s, execute=True)


def test_partial_results_not_silently_reused(tmp_path):
    s = stage(tmp_path)
    (tmp_path / "output").write_text("partial")
    with pytest.raises(ValueError, match="partial"):
        run_stage(s, execute=True)


def test_failed_stage_has_no_receipt(tmp_path):
    import subprocess
    s = stage(tmp_path)
    s["command"] = [sys.executable, "-c", "raise SystemExit(1)"]
    with pytest.raises(subprocess.CalledProcessError):
        run_stage(s, execute=True)
    assert not (tmp_path / "done.json").exists()
