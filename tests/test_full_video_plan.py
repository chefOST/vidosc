"""Check full-video plans using inventory metadata only; no assets or jobs."""
import json
import sys
from pathlib import Path

import pytest

from scripts import prepare_full_video_plan as planner


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    preparation = tmp_path / "runs/batch-preparation"
    preparation.mkdir(parents=True)
    row = {
        "video_id": "S08_Eggs_7150991-727",
        "available": True,
        "metadata_frame_count": 6243,
        "fps": 30,
        "duration_seconds": 208.1,
        "local_video": str(tmp_path / "data/moscato/videos/S08_Eggs_7150991-727.avi"),
    }
    return tmp_path, preparation, row


def prepare(inventory, monkeypatch, *flags):
    _, preparation, row = inventory
    (preparation / "inventory.json").write_text(json.dumps({"videos": [row]}))
    monkeypatch.setattr(sys, "argv", [
        "prepare_full_video_plan.py", "--video-id", row["video_id"], *flags,
    ])
    planner.main()
    path = preparation / f"cmu30-active-object-v1-{row['video_id']}.json"
    return path, json.loads(path.read_text())


def test_default_plan_preserves_active_object_stages_and_outputs(inventory, monkeypatch):
    root, _, row = inventory
    _, plan = prepare(inventory, monkeypatch)
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    assert job["name"] == row["video_id"] + ":active-object"
    assert [stage["name"] for stage in job["stages"]] == ["model", "mapping"]
    model_stage, mapping_stage = job["stages"]
    output = root / "runs/batches/cmu30-active-object-v1" / row["video_id"]
    model_dir = output / "model"
    python = str(root / ".venv/bin/python")
    assert model_stage["command"] == [python, "-m", "modal", "run", "scripts/modal_clip.py"]
    assert model_stage["env"] == {
        "VIDOSC_VIDEO": row["local_video"],
        "VIDOSC_VIDEO_ID": row["video_id"],
        "VIDOSC_FRAME_COUNT": "6243",
        "VIDOSC_FPS": "30",
        "VIDOSC_OUTPUT": str(model_dir),
        "VIDOSC_TIMEOUT": "7200",
    }
    # Inference needs the raw video and weights, never precomputed local features.
    assert model_stage["inputs"][:3] == [
        row["local_video"], str(Path.home() / "Downloads/InternVideo-MM-L-14.ckpt"),
        str(root / "checkpoints/changeit/state_best.ckpt"),
    ]
    assert all(path.endswith(".py") for path in model_stage["inputs"][3:])
    assert mapping_stage["command"] == [
        python, "-m", "moscato.import_modal_predictions", "--run", str(model_dir),
        "--mapping", str(root / "configs/changeit_moscato_alias_mapping.json"),
        "--state-dict", str(root / "data/moscato/vocabulary/state_dict.json"),
        "--output", str(model_dir / "predictions"),
    ]
    assert [stage["modal"] for stage in job["stages"]] == [True, False]
    assert mapping_stage["env"] == {}
    assert model_stage["outputs"] == [str(model_dir / name) for name in (
        "feature_manifest.json", "combined_features.pth.tar", "object_features.pth.tar",
        "raw_probabilities.npz", "raw_logits.npz", "global_features.pth.tar",
    )]
    assert mapping_stage["outputs"] == [str(model_dir / "predictions" / name) for name in (
        "manifest.json", "probabilities.npz", "predictions_by_second.csv",
    )]
    for stage in job["stages"]:
        assert stage["receipt"] == str(output / f"{stage['name']}.done.json")
        assert not {"--annotations", "--timing", "--target-object", "--evaluate"}.intersection(stage["command"])
        assert not any("/annotations/" in path or "timing.json" in path for path in stage["inputs"])
        assert "moscato.extract_internvideo" not in stage["command"]
        assert "--device" not in stage["command"]
    # The fixture supplies no videos, checkpoints, timing or target-object labels.
    assert not Path(row["local_video"]).exists()
    assert not (root / "runs/batches").exists()


def test_explicit_active_object_flag_preserves_default_plan(inventory, monkeypatch):
    _, default = prepare(inventory, monkeypatch)
    _, explicit = prepare(inventory, monkeypatch, "--active-object")
    assert explicit == default


def test_fractional_fps_is_preserved(inventory, monkeypatch):
    _, _, row = inventory
    row.update(fps=29.97, duration_seconds=6243 / 29.97)
    _, plan = prepare(inventory, monkeypatch)
    env = plan["jobs"][0]["stages"][0]["env"]
    assert env["VIDOSC_FPS"] == "29.97"
    assert env["VIDOSC_FRAME_COUNT"] == "6243"


@pytest.mark.parametrize("frames", [27000, 27001, 54015, 150001])
def test_long_videos_preserve_full_frame_count_without_local_extraction(inventory, monkeypatch, frames):
    _, _, row = inventory
    row.update(duration_seconds=frames / row["fps"], metadata_frame_count=frames)
    _, plan = prepare(inventory, monkeypatch)
    stages = plan["jobs"][0]["stages"]
    assert [stage["name"] for stage in stages] == ["model", "mapping"]
    assert stages[0]["env"]["VIDOSC_FRAME_COUNT"] == str(frames)
    assert stages[0]["env"]["VIDOSC_TIMEOUT"] == "7200"
    assert stages[0]["modal"] is True


@pytest.mark.parametrize("change", [{"available": False}, {"metadata_frame_count": None}])
def test_unavailable_or_uninspected_video_has_no_plan(inventory, monkeypatch, change):
    _, preparation, row = inventory
    row.update(change)
    with pytest.raises(SystemExit) as error:
        prepare(inventory, monkeypatch)
    assert error.value.code == 2
    assert not list(preparation.glob("cmu30-active-object-v1-*.json"))
