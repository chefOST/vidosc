"""Exercise coordinator routing with fake subprocesses and no model stages."""
import json
import sys
from unittest.mock import Mock, call

import pytest

from scripts import run_cmu30_active as coordinator


VIDEO_ID = "S08_Eggs_7150991-727"
EXPERIMENT = "offline-routing-check"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "ROOT", tmp_path)
    monkeypatch.setattr(coordinator, "WANDB_ENTITY", "chefOST")
    monkeypatch.setattr(coordinator, "WANDB_PROJECT", "vidosc-experiments")
    (tmp_path / "configs").mkdir()
    preparation = tmp_path / "runs/batch-preparation"
    preparation.mkdir(parents=True)
    (tmp_path / "configs/moscato_cmu30_v1.json").write_text(json.dumps({"video_ids": [VIDEO_ID]}))
    (preparation / "modal_rates.json").write_text(json.dumps({"cpu_hour_cost": 0.05, "mem_gib_hour_cost": 0.001}))
    (preparation / "inventory.json").write_text(json.dumps({"videos": [{
        "video_id": VIDEO_ID, "local_video": str(tmp_path / "missing-video.avi"),
        "metadata_frame_count": 30, "fps": 30, "duration_seconds": 1,
    }]}))
    return tmp_path


def run(workspace, monkeypatch, *flags):
    monkeypatch.setattr(sys, "argv", ["run_cmu30_active.py", "--execute", "--experiment", EXPERIMENT, *flags])
    coordinator.main()
    directory = workspace / "runs/batches" / EXPERIMENT
    status = json.loads((directory / "batch_status.json").read_text())
    assert status["experiment"] == EXPERIMENT
    assert not (directory / "batch.lock").exists()
    assert not (workspace / "runs/batches/cmu30-active-object-v1").exists()
    return directory, status


def receipt_path(workspace):
    return workspace / "runs/batches" / EXPERIMENT / VIDEO_ID / "model/wandb_inference_receipt.json"


def uploaded_receipt():
    return {"url": "https://wandb.ai/chefOST/vidosc-experiments/runs/synthetic",
            "run_id": "synthetic", "entity": "chefOST", "project": "vidosc-experiments"}


def test_matching_destination_skips_all_subprocesses_and_model_stages(workspace, monkeypatch):
    path = receipt_path(workspace)
    path.parent.mkdir(parents=True)
    receipt = uploaded_receipt()
    path.write_text(json.dumps(receipt))
    process = Mock(side_effect=AssertionError("No subprocess should run"))
    stage = Mock(side_effect=AssertionError("No model stage should run"))
    monkeypatch.setattr(coordinator.subprocess, "run", process)
    monkeypatch.setattr(coordinator, "run_stage", stage)
    directory, status = run(workspace, monkeypatch)
    process.assert_not_called()
    stage.assert_not_called()
    assert status["state"] == "complete"
    assert status["reserved_compute_usd"] == 0
    assert status["videos"][VIDEO_ID] == {"state": "uploaded", **receipt}
    assert not (directory / f"{VIDEO_ID}.log").exists()


def test_changed_destination_runs_only_logger_and_preserves_outputs(workspace, monkeypatch):
    path = receipt_path(workspace)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "url": "https://forge.coreweave.com/wandb/chefOST/vidosc%20-%20full%20evals/runs/c0ae0bc82d43003c",
    }))
    output = path.parent / "raw_probabilities.npz"
    output.write_bytes(b"existing output must not be regenerated")
    receipt = uploaded_receipt()

    def log_only(command, *, cwd, check):
        assert command == [sys.executable, "scripts/log_inference_wandb.py", "--run", str(path.parent)]
        assert cwd == workspace and check is True
        path.write_text(json.dumps(receipt))

    process = Mock(side_effect=log_only)
    stage = Mock(side_effect=AssertionError("Existing outputs must bypass model stages"))
    monkeypatch.setattr(coordinator.subprocess, "run", process)
    monkeypatch.setattr(coordinator, "run_stage", stage)
    _, status = run(workspace, monkeypatch)
    process.assert_called_once()
    stage.assert_not_called()
    assert output.read_bytes() == b"existing output must not be regenerated"
    assert status["state"] == "complete"
    assert status["reserved_compute_usd"] == 0
    assert status["videos"][VIDEO_ID] == {"state": "uploaded", **receipt}


def test_custom_experiment_reaches_planner_stages_logger_and_status(workspace, monkeypatch):
    path = receipt_path(workspace)
    plan_path = workspace / "runs/batch-preparation" / f"{EXPERIMENT}-{VIDEO_ID}.json"
    stages = [{"name": name, "env": {}} for name in ("model", "mapping")]

    def fake_process(command, *, cwd, check):
        assert cwd == workspace and check is True
        if command[1] == "scripts/prepare_full_video_plan.py":
            assert command == [sys.executable, "scripts/prepare_full_video_plan.py",
                               "--video-id", VIDEO_ID, "--active-object", "--experiment", EXPERIMENT]
            plan_path.write_text(json.dumps({"jobs": [{"stages": stages}]}))
        else:
            assert command == [sys.executable, "scripts/log_inference_wandb.py", "--run", str(path.parent)]
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(uploaded_receipt()))

    process = Mock(side_effect=fake_process)
    stage = Mock(return_value="completed")
    monkeypatch.setattr(coordinator.subprocess, "run", process)
    monkeypatch.setattr(coordinator, "run_stage", stage)
    directory, status = run(workspace, monkeypatch)
    assert process.call_count == 2
    assert stage.call_args_list == [
        call({"name": "model", "env": {"VIDOSC_TIMEOUT": "7200"}}, execute=True, allow_modal=True),
        call({"name": "mapping", "env": {}}, execute=True, allow_modal=True),
    ]
    assert plan_path.is_file()
    assert (directory / f"{VIDEO_ID}.log").is_file()
    assert status["state"] == "complete"
    assert status["reserved_compute_usd"] == pytest.approx(7210 / 3600 * 0.216)
    assert status["videos"][VIDEO_ID]["project"] == "vidosc-experiments"


def test_budget_stop_precedes_modal_stage_and_upload(workspace, monkeypatch):
    plan_path = workspace / "runs/batch-preparation" / f"{EXPERIMENT}-{VIDEO_ID}.json"

    def prepare_only(command, *, cwd, check):
        assert command[1] == "scripts/prepare_full_video_plan.py"
        assert cwd == workspace and check is True
        plan_path.write_text(json.dumps({"jobs": [{"stages": [
            {"name": "model", "env": {"VIDOSC_TIMEOUT": "7200"}},
            {"name": "mapping", "env": {}},
        ]}]}))

    process = Mock(side_effect=prepare_only)
    stage = Mock(side_effect=AssertionError("Insufficient budget must not launch any stage"))
    monkeypatch.setattr(coordinator.subprocess, "run", process)
    monkeypatch.setattr(coordinator, "run_stage", stage)
    _, status = run(workspace, monkeypatch, "--budget-usd", "1")
    process.assert_called_once()
    stage.assert_not_called()
    assert status["state"] == "budget_stop"
    assert status["next_video"] == VIDEO_ID
    assert status["reserved_compute_usd"] == 0
    assert status["videos"] == {}
