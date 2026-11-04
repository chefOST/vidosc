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
    assert [stage["name"] for stage in job["stages"]] == ["global", "model", "mapping"]
    global_stage, model_stage, mapping_stage = job["stages"]
    output = root / "runs/batches/cmu30-active-object-v1" / row["video_id"]
    global_dir, model_dir = output / "global", output / "model"
    python = str(root / ".venv/bin/python")
    assert global_stage["command"] == [
        python, "-m", "moscato.extract_internvideo", "--video", row["local_video"],
        "--video-id", row["video_id"], "--object", "active-object",
        "--checkpoint", str(Path.home() / "Downloads/InternVideo-MM-L-14.ckpt"),
        "--source", str(root.parent / "internvideo-upstream"),
        "--output", str(global_dir), "--start-frame", "0", "--end-frame", "6243",
        "--fps", "30", "--device", "cpu",
    ]
    assert model_stage["command"] == [python, "-m", "modal", "run", "scripts/modal_clip.py"]
    assert model_stage["env"] == {
        "VIDOSC_VIDEO": row["local_video"],
        "VIDOSC_GLOBAL_DIR": str(global_dir),
        "VIDOSC_OUTPUT": str(model_dir),
        "VIDOSC_TIMEOUT": "1057",
    }
    assert mapping_stage["command"] == [
        python, "-m", "moscato.import_modal_predictions", "--run", str(model_dir),
        "--mapping", str(root / "configs/changeit_moscato_alias_mapping.json"),
        "--state-dict", str(root / "data/moscato/vocabulary/state_dict.json"),
        "--output", str(model_dir / "predictions"),
    ]
    assert [stage["modal"] for stage in job["stages"]] == [False, True, False]
    assert global_stage["env"] == mapping_stage["env"] == {}
    assert global_stage["outputs"] == [str(global_dir / name) for name in (
        "global_features.pth.tar", "feature_manifest.json",
    )]
    assert model_stage["outputs"] == [str(model_dir / name) for name in (
        "feature_manifest.json", "combined_features.pth.tar", "object_features.pth.tar",
        "raw_probabilities.npz", "raw_logits.npz",
    )]
    assert mapping_stage["outputs"] == [str(model_dir / "predictions" / name) for name in (
        "manifest.json", "probabilities.npz", "predictions_by_second.csv",
    )]
    for stage in job["stages"]:
        assert stage["receipt"] == str(output / f"{stage['name']}.done.json")
        assert not {"--annotations", "--timing", "--target-object", "--evaluate"}.intersection(stage["command"])
        assert not any("/annotations/" in path or "timing.json" in path for path in stage["inputs"])
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
    command = plan["jobs"][0]["stages"][0]["command"]
    assert command[command.index("--fps") + 1] == "29.97"
    assert command[command.index("--end-frame") + 1] == "6243"


def test_exactly_900_seconds_is_supported(inventory, monkeypatch):
    _, _, row = inventory
    row.update(duration_seconds=900, metadata_frame_count=27000)
    _, plan = prepare(inventory, monkeypatch)
    stages = plan["jobs"][0]["stages"]
    assert stages[0]["command"][stages[0]["command"].index("--end-frame") + 1] == "27000"
    assert stages[1]["env"]["VIDOSC_TIMEOUT"] == "3600"


def test_over_900_seconds_requires_chunking_before_plan(inventory, monkeypatch, capsys):
    _, preparation, row = inventory
    row.update(duration_seconds=900.01, metadata_frame_count=27001)
    with pytest.raises(SystemExit) as error:
        prepare(inventory, monkeypatch)
    assert error.value.code == 2
    assert "chunking protocol" in capsys.readouterr().err
    assert not list(preparation.glob("cmu30-active-object-v1-*.json"))


@pytest.mark.parametrize("change", [{"available": False}, {"metadata_frame_count": None}])
def test_unavailable_or_uninspected_video_has_no_plan(inventory, monkeypatch, change):
    _, preparation, row = inventory
    row.update(change)
    with pytest.raises(SystemExit) as error:
        prepare(inventory, monkeypatch)
    assert error.value.code == 2
    assert not list(preparation.glob("cmu30-active-object-v1-*.json"))
