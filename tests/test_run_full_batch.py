"""Exercise batch planning, preflight, budgets and routing without cloud calls."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import run_full_batch as runner
from scripts.prepare_full_video_plan import build_plan

EXPERIMENT = "offline-test"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("VIDOSC_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("VIDOSC_STATE_VOCABULARY", str(tmp_path / "data/moscato/vocabulary/state_dict.json"))
    monkeypatch.setattr(runner, "WANDB_ENTITY", "chefOST")
    monkeypatch.setattr(runner, "WANDB_PROJECT", "vidosc-experiments")
    r2 = {"bucket": "fixture", "endpoint": "https://example.invalid", "prefix": "release/",
          "manifest_key": "release/manifest.json", "release_sha256": "a" * 64}
    rows = [{"video_id": dataset + "_video", "dataset": dataset, "available": True,
             "metadata_frame_count": 75600, "fps": 30, "duration_seconds": 2520,
             "r2_key": f"release/{dataset}/video.mp4", "sha256": "b" * 64, "bytes": 1234}
            for dataset in ("CMU", "EGTEA")]
    inventory = {"videos": rows + [{"video_id": "unavailable", "available": False}], "r2": r2}
    preparation = tmp_path / "runs/batch-preparation"
    preparation.mkdir(parents=True)
    (preparation / "inventory.json").write_text(json.dumps(inventory))
    (preparation / "modal_rates.json").write_text(json.dumps({"cpu_hour_cost": 0.05, "mem_gib_hour_cost": 0.001}))
    for row in rows:
        produced = set()
        for stage in build_plan(row, r2, EXPERIMENT, root=tmp_path)["jobs"][0]["stages"]:
            for name in stage["inputs"]:
                path = Path(name)
                if name not in produced and not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
            produced.update(stage["outputs"])
    expected = {name: runner.fingerprint(runner.checkpoint_path(name, tmp_path)) for name in runner.WEIGHT_SHA256}
    monkeypatch.setattr(runner, "WEIGHT_SHA256", expected)
    (tmp_path / "configs/r2_release.json").write_text(json.dumps(r2))
    (tmp_path / "configs/changeit_moscato_alias_mapping.json").write_text(json.dumps({
        "checkpoint_sha256": expected["changeit/state_best.ckpt"],
        "target_vocabulary_sha256": runner.fingerprint(tmp_path / "data/moscato/vocabulary/state_dict.json"),
    }))
    monkeypatch.setattr(runner, "preflight_runtime", Mock())
    monkeypatch.setattr(runner, "execute_plan", Mock())

    def upload(folder, group):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "wandb_inference_receipt.json").write_text(json.dumps(receipt(group)))

    monkeypatch.setattr(runner, "upload", Mock(side_effect=upload))
    return tmp_path, inventory


def receipt(group=EXPERIMENT):
    return {"url": "https://wandb.ai/chefOST/vidosc-experiments/runs/fixture",
            "entity": "chefOST", "project": "vidosc-experiments", "group": group}


def invoke(workspace, monkeypatch, *flags):
    root, _ = workspace
    monkeypatch.setattr(sys, "argv", ["run_full_batch.py", "--experiment", EXPERIMENT, *flags])
    runner.main()
    path = root / "runs/batches" / EXPERIMENT / "batch_status.json"
    assert not (path.parent / "batch.lock").exists()
    return json.loads(path.read_text()) if path.exists() else None


def test_preview_enumerates_all_available_datasets_without_work(workspace, monkeypatch, capsys):
    root, _ = workspace
    assert invoke(workspace, monkeypatch) is None
    output = json.loads(capsys.readouterr().out)
    assert output["videos"] == ["CMU_video", "EGTEA_video"]
    assert output["worker_timeout_seconds"] == 21600
    assert output["budget_usd"] is None
    runner.preflight_runtime.assert_not_called()
    runner.execute_plan.assert_not_called()
    assert not (root / "runs/batches").exists()


def test_explicit_subset_does_not_schedule_other_videos(workspace, monkeypatch):
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100", "--video-id", "EGTEA_video")
    assert state["state"] == "complete"
    assert state["selected_video_ids"] == ["EGTEA_video"]
    stages = runner.execute_plan.call_args.args[0]["jobs"][0]["stages"]
    assert [stage["name"] for stage in stages] == ["model"]
    runner.upload.assert_called_once()
    assert runner.upload.call_args.args[1] == EXPERIMENT


def test_all_plans_preflight_before_any_paid_stage(workspace, monkeypatch):
    checks = []
    real_preflight = runner.preflight_plan

    def preflight(plan, dataset):
        checks.append(dataset)
        real_preflight(plan, dataset)

    def execute(plan, path):
        assert checks == ["CMU", "EGTEA"]
        assert plan["jobs"][0]["stages"][0]["env"]["VIDOSC_TIMEOUT"] == "36000"

    monkeypatch.setattr(runner, "preflight_plan", preflight)
    runner.execute_plan.side_effect = execute
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100", "--timeout", "36000")
    assert state["state"] == "complete"
    assert runner.execute_plan.call_count == 2
    assert state["reserved_compute_usd"] == pytest.approx(2 * 36010 / 3600 * .216)


@pytest.mark.parametrize("flags", [["--execute"], ["--execute", "--budget-usd", "nan"],
    ["--execute", "--budget-usd", "inf"], ["--execute", "--budget-usd", "0"],
    ["--video-id", "unknown"], ["--video-id", "unavailable"], ["--timeout", "86401"]])
def test_invalid_request_never_executes(workspace, monkeypatch, flags):
    with pytest.raises(SystemExit) as error:
        invoke(workspace, monkeypatch, *flags)
    assert error.value.code == 2
    runner.execute_plan.assert_not_called()
    runner.upload.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "0.05"])
def test_invalid_rates_rejected(value):
    with pytest.raises(ValueError, match="finite positive"):
        runner.worker_hourly_rate({"cpu_hour_cost": value, "mem_gib_hour_cost": .001})


def test_budget_stop_precedes_paid_stage_and_upload(workspace, monkeypatch):
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "1")
    assert state["state"] == "budget_stop"
    assert state["next_video"] == "CMU_video"
    assert state["reserved_compute_usd"] == 0
    runner.execute_plan.assert_not_called()
    runner.upload.assert_not_called()


def test_bad_video_metadata_does_not_block_other_dataset(workspace, monkeypatch):
    root, inventory = workspace
    inventory["videos"][0]["metadata_frame_count"] = None
    (root / "runs/batch-preparation/inventory.json").write_text(json.dumps(inventory))
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100")
    assert state["state"] == "finished_with_failures"
    assert state["videos"]["CMU_video"]["state"] == "failed"
    assert state["videos"]["EGTEA_video"]["state"] == "uploaded"
    runner.execute_plan.assert_called_once()


def test_bad_cmu_vocabulary_blocks_cmu_before_paid_stage_only(workspace, monkeypatch):
    root, _ = workspace
    (root / "data/moscato/vocabulary/state_dict.json").write_text("different vocabulary")
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100")
    assert "vocabulary differs" in state["videos"]["CMU_video"]["error"]
    assert state["videos"]["EGTEA_video"]["state"] == "uploaded"
    runner.execute_plan.assert_called_once()


def test_wrong_checkpoint_prevents_every_paid_stage(workspace, monkeypatch):
    root, _ = workspace
    (root / "checkpoints/changeit/state_best.ckpt").write_bytes(b"wrong weights")
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100")
    assert state["state"] == "finished_with_failures"
    assert all("fingerprint mismatch" in value["error"] for value in state["videos"].values())
    runner.execute_plan.assert_not_called()
    runner.upload.assert_not_called()


def test_failed_paid_video_is_not_retried_and_next_video_runs(workspace, monkeypatch):
    runner.execute_plan.side_effect = [RuntimeError("worker failed"), None]
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100")
    assert state["state"] == "finished_with_failures"
    assert state["videos"]["CMU_video"]["state"] == "failed"
    assert state["videos"]["EGTEA_video"]["state"] == "uploaded"
    assert runner.execute_plan.call_count == 2
    runner.upload.assert_called_once()


@pytest.mark.parametrize("old_group,uploads", [(EXPERIMENT, 0), ("cmu30-active-object-v1", 1)])
def test_existing_receipt_reuse_or_relog_never_runs_model(workspace, monkeypatch, old_group, uploads):
    root, inventory = workspace
    folder = root / "runs/batches" / EXPERIMENT / "CMU_video/model"
    folder.mkdir(parents=True)
    (folder / "wandb_inference_receipt.json").write_text(json.dumps(receipt(old_group)))
    row = inventory["videos"][0]
    (folder / "feature_manifest.json").write_text(json.dumps({
        "video_id": row["video_id"], "video_sha256": row["sha256"], "dataset": row["dataset"],
        "clip_start_seconds": 0, "clip_end_seconds": row["metadata_frame_count"] / row["fps"],
    }))
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "1", "--video-id", "CMU_video")
    assert state["state"] == "complete"
    assert state["reserved_compute_usd"] == 0
    assert runner.upload.call_count == uploads
    runner.execute_plan.assert_not_called()


def test_historical_avi_receipt_is_not_reused_for_r2_mp4(workspace, monkeypatch):
    root, inventory = workspace
    row = inventory["videos"][0]
    folder = root / "runs/batches" / EXPERIMENT / "CMU_video/model"
    folder.mkdir(parents=True)
    (folder / "wandb_inference_receipt.json").write_text(json.dumps(receipt()))
    (folder / "feature_manifest.json").write_text(json.dumps({
        "video_id": row["video_id"], "video_sha256": "c" * 64,
        "clip_start_seconds": 0, "clip_end_seconds": row["duration_seconds"],
    }))
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100", "--video-id", "CMU_video")
    assert state["state"] == "finished_with_failures"
    assert "fresh experiment" in state["videos"]["CMU_video"]["error"]
    runner.execute_plan.assert_not_called()
    runner.upload.assert_not_called()


def test_mismatched_release_rejected_before_execution(workspace, monkeypatch):
    root, inventory = workspace
    inventory["r2"]["release_sha256"] = "c" * 64
    (root / "runs/batch-preparation/inventory.json").write_text(json.dumps(inventory))
    with pytest.raises(SystemExit) as error:
        invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100")
    assert error.value.code == 2
    runner.execute_plan.assert_not_called()
    runner.upload.assert_not_called()


def test_runtime_failure_prevents_all_paid_stages(workspace, monkeypatch):
    runner.preflight_runtime.side_effect = RuntimeError("Missing account access")
    state = invoke(workspace, monkeypatch, "--execute", "--budget-usd", "100")
    assert state["state"] == "finished_with_failures"
    assert state["reserved_compute_usd"] == 0
    runner.execute_plan.assert_not_called()
    runner.upload.assert_not_called()


def test_runtime_check_is_imports_then_noninteractive_account_check(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    process = Mock()
    monkeypatch.setattr(runner.subprocess, "run", process)
    runner.preflight_runtime()
    assert process.call_args_list[0].args[0] == [str(python), "-c", "import modal, numpy, wandb, boto3"]
    assert process.call_args_list[1].args[0] == [str(python), "scripts/setup_accounts.py", "--check"]
    assert process.call_args_list[1].kwargs["stdin"] == subprocess.DEVNULL
