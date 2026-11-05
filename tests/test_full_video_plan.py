"""R2 planning contract checks; no source videos, models or cloud calls."""
import json
from pathlib import Path
import sys

import pytest

from scripts import prepare_full_video_plan as planner


@pytest.fixture
def inputs(tmp_path):
    r2 = {"bucket": "fixture-bucket", "endpoint": "https://example.invalid",
          "prefix": "release/v1/", "manifest_key": "release/v1/manifest.json", "release_sha256": "a" * 64}
    row = {"video_id": "S08_Eggs_7150991-727", "dataset": "CMU", "available": True,
           "metadata_frame_count": 6243, "fps": 30, "duration_seconds": 208.1,
           "r2_key": "release/v1/CMU/S08_Eggs_7150991-727.mp4", "sha256": "b" * 64, "bytes": 1234}
    return tmp_path, row, r2


def test_cmu_plan_uses_r2_then_mapping_without_local_extraction(inputs):
    root, row, r2 = inputs
    plan = planner.build_plan(row, r2, root=root)
    stages = plan["jobs"][0]["stages"]
    assert [stage["name"] for stage in stages] == ["model", "mapping"]
    model, mapping = stages
    output = root / "runs/batches/full-r2-v1" / row["video_id"] / "model"
    assert model["command"] == [str(root / ".venv/bin/python"), "-m", "modal", "run", "scripts/modal_clip.py"]
    assert model["env"] == {
        "VIDOSC_R2_KEY": row["r2_key"], "VIDOSC_VIDEO_SHA256": row["sha256"],
        "VIDOSC_VIDEO_BYTES": "1234", "VIDOSC_VIDEO_ID": row["video_id"],
        "VIDOSC_FRAME_COUNT": "6243", "VIDOSC_FPS": "30", "VIDOSC_DATASET": "CMU",
        "VIDOSC_R2_BUCKET": r2["bucket"], "VIDOSC_R2_ENDPOINT": r2["endpoint"],
        "VIDOSC_R2_PREFIX": r2["prefix"], "VIDOSC_OUTPUT": str(output), "VIDOSC_TIMEOUT": "21600",
    }
    assert model["inputs"][:3] == [str(root / name) for name in (
        "runs/batch-preparation/inventory.json", "checkpoints/internvideo/InternVideo-MM-L-14.ckpt",
        "checkpoints/changeit/state_best.ckpt")]
    for source in ("moscato/r2_inputs.py", "scripts/inventory_r2.py", "configs/r2_release.json"):
        assert str(root / source) in model["inputs"]
    assert [Path(path).name for path in model["outputs"]] == [
        "feature_manifest.json", "combined_features.pth.tar", "object_features.pth.tar",
        "raw_probabilities.npz", "raw_logits.npz"]
    assert str(root / "data/moscato/vocabulary/state_dict.json") in mapping["inputs"]
    assert [stage["modal"] for stage in stages] == [True, False]
    assert all("moscato.extract_internvideo" not in stage["command"] for stage in stages)
    assert not list(root.iterdir())


def test_egtea_has_only_raw_model_stage(inputs):
    root, row, r2 = inputs
    row.update(dataset="EGTEA", video_id="OP01-R01-PastaSalad", r2_key="release/v1/EGTEA/video.mp4")
    stages = planner.build_plan(row, r2, root=root)["jobs"][0]["stages"]
    assert len(stages) == 1
    assert stages[0]["env"]["VIDOSC_DATASET"] == "EGTEA"
    assert not any("state_dict.json" in path or "alias_mapping.json" in path for path in stages[0]["inputs"])
    assert not any("predictions" in path for path in stages[0]["outputs"])


@pytest.mark.parametrize("frames,fps", [(27000, 30), (27001, 30), (75600, 30), (150001, 29.97)])
def test_complete_video_metadata_is_preserved_beyond_900_seconds(inputs, frames, fps):
    root, row, r2 = inputs
    row.update(metadata_frame_count=frames, fps=fps, duration_seconds=frames / fps)
    env = planner.build_plan(row, r2, root=root)["jobs"][0]["stages"][0]["env"]
    assert env["VIDOSC_FRAME_COUNT"] == str(frames)
    assert env["VIDOSC_FPS"] == str(fps)


@pytest.mark.parametrize("timeout", [60, 21600, 86400])
def test_timeout_is_explicit_in_worker_identity(inputs, timeout):
    root, row, r2 = inputs
    env = planner.build_plan(row, r2, timeout=timeout, root=root)["jobs"][0]["stages"][0]["env"]
    assert env["VIDOSC_TIMEOUT"] == str(timeout)


@pytest.mark.parametrize("change", [
    {"available": False}, {"metadata_frame_count": None}, {"metadata_frame_count": -1},
    {"fps": float("nan")}, {"fps": 0}, {"bytes": 0}, {"sha256": "not-a-hash"},
    {"r2_key": "other-release/video.mp4"}, {"dataset": "unsupported"}, {"video_id": "../../escape"},
])
def test_invalid_input_metadata_is_rejected(inputs, change):
    root, row, r2 = inputs
    row.update(change)
    with pytest.raises(ValueError):
        planner.build_plan(row, r2, root=root)


@pytest.mark.parametrize("options", [{"timeout": 59}, {"timeout": 86401}, {"experiment": "../escape"}])
def test_invalid_execution_options_are_rejected(inputs, options):
    root, row, r2 = inputs
    with pytest.raises(ValueError):
        planner.build_plan(row, r2, root=root, **options)


def test_cli_writes_selected_r2_plan_only(inputs, monkeypatch):
    root, row, r2 = inputs
    preparation = root / "runs/batch-preparation"
    preparation.mkdir(parents=True)
    (preparation / "inventory.json").write_text(json.dumps({"videos": [row], "r2": r2}))
    monkeypatch.setattr(planner, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["prepare_full_video_plan.py", "--video-id", row["video_id"], "--timeout", "36000"])
    planner.main()
    saved = json.loads((preparation / f"full-r2-v1-{row['video_id']}.json").read_text())
    assert saved["jobs"][0]["stages"][0]["env"]["VIDOSC_TIMEOUT"] == "36000"
    assert not (root / "runs/batches").exists()
