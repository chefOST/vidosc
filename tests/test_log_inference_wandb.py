"""Preserve the active-object output contract without importing W&B or models."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import numpy as np
import pytest

from scripts import log_inference_wandb as logger


ARTIFACT_FILES = [
    "feature_manifest.json",
    "combined_features.pth.tar",
    "object_features.pth.tar",
    "raw_logits.npz",
    "raw_probabilities.npz",
    "predictions/manifest.json",
    "predictions/probabilities.npz",
    "predictions/predictions_by_second.csv",
]

# Recorded feature hashes test run identity; fixture tensors are synthetic.
RECORDED_RUNS = [
    ("S08_Eggs_7150991-727",
     "da8ec70dcd71a9949e9b8e928e757ec07ce9e1bbb45af8b7bc7692fc50cc9d51",
     "c0ae0bc82d43003c"),
    ("S29_Salad_7150991-1",
     "2c6b867cf03d89de4ba055f413740d3fc4a09aec9554d276a2f515680e51f745",
     "886538b4dcf21777"),
]


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "predictions").mkdir()
    for name in ARTIFACT_FILES:
        (tmp_path / name).write_bytes(b"synthetic artifact")
    manifest = {
        "ground_truth_used": False,
        "object": "active-object",
        "video_id": RECORDED_RUNS[0][0],
        "feature_sha256": RECORDED_RUNS[0][1],
        "vidosc_checkpoint_sha256": "synthetic-checkpoint",
        "interval_starts_seconds": [0, 1, 2],
        "clip_start_seconds": 0,
        "clip_end_seconds": 2.2,
        "crop_policy": {"name": "midpoint_hand_associated_union_v1"},
        "object_windows": 2,
        "missing_object_windows": 1,
    }
    (tmp_path / "feature_manifest.json").write_text(json.dumps(manifest))
    logits, probabilities = {}, {}
    for name, width in (("state", 52), ("action", 28)):
        # The large offset also exercises stable softmax validation.
        raw = np.arange(3 * width, dtype=np.float64).reshape(3, width) + 1000
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        logits[name + "_logits"] = raw
        probabilities[name] = exp / exp.sum(axis=1, keepdims=True)
    np.savez(tmp_path / "raw_logits.npz", **logits)
    np.savez(tmp_path / "raw_probabilities.npz", **probabilities)
    (tmp_path / "predictions/manifest.json").write_text("{}")
    np.savez(tmp_path / "predictions/probabilities.npz", mapped=np.zeros((3, 77)))
    (tmp_path / "predictions/predictions_by_second.csv").write_text("clip_second,state_label\n")
    return tmp_path, manifest


def invoke(monkeypatch, folder, *extra):
    monkeypatch.setattr(sys, "argv", [
        "log_inference_wandb.py", "--run", str(folder),
        "--entity", "chefOST", "--project", "vidosc - full evals", *extra,
    ])
    logger.main()


def fake_wandb(monkeypatch, run_id):
    run = MagicMock()
    run.__enter__.return_value = run
    run.id = run_id
    run.url = f"https://wandb.ai/chefOST/vidosc%20-%20full%20evals/runs/{run_id}"
    run.summary = {}
    artifact = Mock()
    client = SimpleNamespace(init=Mock(return_value=run), Artifact=Mock(return_value=artifact))
    monkeypatch.setitem(sys.modules, "wandb", client)
    return client, run, artifact


def test_valid_bundle_and_dry_run_do_not_upload(bundle, monkeypatch, capsys):
    folder, manifest = bundle
    assert logger.validate(folder) == (manifest, ARTIFACT_FILES)
    client, _, _ = fake_wandb(monkeypatch, "unused")
    invoke(monkeypatch, folder, "--dry-run")
    client.init.assert_not_called()
    assert "3 intervals; no metrics" in capsys.readouterr().out
    assert not (folder / "wandb_inference_receipt.json").exists()


@pytest.mark.parametrize("missing", ARTIFACT_FILES)
def test_each_output_is_required(bundle, missing):
    folder, _ = bundle
    (folder / missing).unlink()
    with pytest.raises(FileNotFoundError):
        logger.validate(folder)


@pytest.mark.parametrize("key,value", [("ground_truth_used", True), ("object", "vegetables")])
def test_rejects_gt_or_named_object_stream(bundle, key, value):
    folder, manifest = bundle
    manifest[key] = value
    (folder / "feature_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unnamed active-object"):
        logger.validate(folder)


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "probabilities"])
def test_rejects_invalid_logits_or_probabilities(bundle, fault):
    folder, _ = bundle
    filename = "raw_probabilities.npz" if fault == "probabilities" else "raw_logits.npz"
    with np.load(folder / filename) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if fault == "shape":
        arrays["state_logits"] = arrays["state_logits"][:-1]
    elif fault == "nonfinite":
        arrays["action_logits"][0, 0] = np.nan
    else:
        arrays["state"][0] = np.roll(arrays["state"][0], 1)
    np.savez(folder / filename, **arrays)
    with pytest.raises(ValueError, match="Invalid output|reproduce saved probabilities"):
        logger.validate(folder)


@pytest.mark.parametrize("video_id,feature_sha256,run_id", RECORDED_RUNS)
def test_preserves_recorded_ids_and_upload_contract(bundle, monkeypatch, video_id, feature_sha256, run_id):
    folder, manifest = bundle
    manifest.update(video_id=video_id, feature_sha256=feature_sha256)
    (folder / "feature_manifest.json").write_text(json.dumps(manifest))
    client, run, artifact = fake_wandb(monkeypatch, run_id)
    invoke(monkeypatch, folder)
    client.init.assert_called_once_with(
        entity="chefOST", project="vidosc - full evals", group="cmu30-active-object-v1",
        id=run_id, resume="allow", name=f"{video_id}-active-object", job_type="inference",
        tags=["vidosc", "cmu30", "active-object", "outputs-only"],
        config={
            "video_id": video_id,
            "scope": "one unnamed active-object stream per video",
            "dataset": "CMU",
            "mapping_applied": True,
            "checkpoint_sha256": "synthetic-checkpoint",
            "feature_sha256": feature_sha256,
            "crop_policy": manifest["crop_policy"],
            "metrics_computed": False,
            "temporal_postprocessing": "none",
            "input_timesteps": 3,
            "duration_seconds": 2.2,
            "provenance": "fresh extraction",
        },
    )
    assert run.summary == {
        "coverage/object_windows": 2,
        "coverage/missing_object_windows": 1,
        "completion": "outputs_saved",
    }
    client.Artifact.assert_called_once_with(f"{video_id}-active-object-outputs", type="inference-outputs")
    assert artifact.add_file.call_args_list == [
        call(str(folder / name), name=name) for name in ARTIFACT_FILES
    ]
    run.log_artifact.assert_called_once_with(artifact)
    run.log_artifact.return_value.wait.assert_called_once_with()
    assert json.loads((folder / "wandb_inference_receipt.json").read_text()) == {
        "url": run.url, "run_id": run_id, "group": "cmu30-active-object-v1",
        "entity": "chefOST", "project": "vidosc - full evals",
    }


@pytest.mark.parametrize("kind", ["explicit", "legacy_wandb", "legacy_forge"])
def test_existing_upload_receipt_prevents_duplicate_upload(bundle, monkeypatch, kind):
    folder, _ = bundle
    client, run, _ = fake_wandb(monkeypatch, RECORDED_RUNS[0][2])
    if kind == "legacy_forge":
        run.url = f"https://forge.coreweave.com/wandb/chefOST/vidosc%20-%20full%20evals/runs/{run.id}"
    receipt = {"url": run.url, "run_id": run.id, "group": "cmu30-active-object-v1"}
    if kind == "explicit":
        receipt.update(entity="chefOST", project="vidosc - full evals")
    path = folder / "wandb_inference_receipt.json"
    path.write_text(json.dumps(receipt))
    before = path.read_bytes()
    invoke(monkeypatch, folder)
    client.init.assert_not_called()
    assert path.read_bytes() == before


def test_legacy_forge_receipt_matches_only_its_original_destination():
    receipt = {"url": "https://forge.coreweave.com/wandb/chefOST/vidosc%20-%20full%20evals/runs/c0ae0bc82d43003c"}
    assert logger.receipt_matches(receipt, "chefOST", "vidosc - full evals")
    assert not logger.receipt_matches(receipt, "chefOST", "vidosc-experiments")
    assert not logger.receipt_matches(receipt, "another-team", "vidosc - full evals")


def test_group_matching_is_optional_for_legacy_receipts():
    receipt = {"url": "https://wandb.ai/chefOST/vidosc-experiments/runs/synthetic-run"}
    assert logger.receipt_matches(receipt, "chefOST", "vidosc-experiments")
    assert not logger.receipt_matches(receipt, "chefOST", "vidosc-experiments", "full-r2-v1")
    receipt["group"] = "full-r2-v1"
    assert logger.receipt_matches(receipt, "chefOST", "vidosc-experiments", "full-r2-v1")
    assert not logger.receipt_matches(receipt, "chefOST", "vidosc-experiments", "another-batch")


def test_omitted_destination_flags_use_shared_configuration(bundle, monkeypatch):
    folder, _ = bundle
    monkeypatch.setattr(logger, "WANDB_ENTITY", "configured-team")
    monkeypatch.setattr(logger, "WANDB_PROJECT", "configured-experiments")
    client, run, _ = fake_wandb(monkeypatch, RECORDED_RUNS[0][2])
    run.url = f"https://wandb.ai/configured-team/configured-experiments/runs/{run.id}"
    monkeypatch.setattr(sys, "argv", ["log_inference_wandb.py", "--run", str(folder)])
    logger.main()
    client.init.assert_called_once()
    assert client.init.call_args.kwargs["entity"] == "configured-team"
    assert client.init.call_args.kwargs["project"] == "configured-experiments"
    receipt = json.loads((folder / "wandb_inference_receipt.json").read_text())
    assert receipt["entity"] == "configured-team"
    assert receipt["project"] == "configured-experiments"


@pytest.fixture
def egtea_bundle(bundle):
    folder, manifest = bundle
    manifest.update(dataset="EGTEA", video_id="synthetic-egtea-video")
    (folder / "feature_manifest.json").write_text(json.dumps(manifest))
    for name in ARTIFACT_FILES[5:]:
        (folder / name).unlink()
    (folder / "predictions").rmdir()
    return folder, manifest


def test_explicit_cmu_keeps_all_eight_artifacts(bundle):
    folder, manifest = bundle
    manifest["dataset"] = "CMU"
    (folder / "feature_manifest.json").write_text(json.dumps(manifest))
    assert logger.validate(folder) == (manifest, ARTIFACT_FILES)
    (folder / ARTIFACT_FILES[-1]).unlink()
    with pytest.raises(FileNotFoundError):
        logger.validate(folder)


@pytest.mark.parametrize("dataset", ["egtea", "unknown", "", None])
def test_rejects_unsupported_dataset(bundle, dataset):
    folder, manifest = bundle
    manifest["dataset"] = dataset
    (folder / "feature_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Expected dataset CMU or EGTEA"):
        logger.validate(folder)


@pytest.mark.parametrize("missing", ARTIFACT_FILES[:5])
def test_egtea_requires_every_raw_output(egtea_bundle, missing):
    folder, _ = egtea_bundle
    (folder / missing).unlink()
    with pytest.raises(FileNotFoundError):
        logger.validate(folder)


@pytest.mark.parametrize("mapped_files", [[], ARTIFACT_FILES[5:6], ARTIFACT_FILES[5:]])
def test_egtea_rejects_partial_or_complete_mapping(egtea_bundle, mapped_files):
    folder, _ = egtea_bundle
    (folder / "predictions").mkdir()
    for name in mapped_files:
        (folder / name).write_bytes(b"unexpected mapped output")
    with pytest.raises(ValueError, match="EGTEA has no supported alias mapping"):
        logger.validate(folder)


def test_egtea_logs_only_five_raw_files_without_mapping_or_metrics(egtea_bundle, monkeypatch):
    folder, manifest = egtea_bundle
    assert logger.validate(folder) == (manifest, ARTIFACT_FILES[:5])
    client, run, artifact = fake_wandb(monkeypatch, "synthetic-egtea-run")
    invoke(monkeypatch, folder, "--group", "r2-full-active-object-v1")
    settings = client.init.call_args.kwargs
    assert settings["group"] == "r2-full-active-object-v1"
    assert settings["tags"] == ["vidosc", "egtea", "active-object", "outputs-only"]
    assert settings["config"]["dataset"] == "EGTEA"
    assert settings["config"]["mapping_applied"] is False
    assert settings["config"]["metrics_computed"] is False
    assert artifact.add_file.call_args_list == [
        call(str(folder / name), name=name) for name in ARTIFACT_FILES[:5]
    ]
    assert set(run.summary) == {"coverage/object_windows", "coverage/missing_object_windows", "completion"}
    receipt = json.loads((folder / "wandb_inference_receipt.json").read_text())
    assert receipt["group"] == "r2-full-active-object-v1"


def test_importer_rejects_egtea_before_loading_mapping_or_creating_outputs(egtea_bundle, monkeypatch):
    from moscato import import_modal_predictions as importer

    folder, _ = egtea_bundle
    output = folder / "predictions"
    monkeypatch.setattr(sys, "argv", [
        "import_modal_predictions.py", "--run", str(folder),
        "--mapping", str(folder / "missing-mapping.json"),
        "--state-dict", str(folder / "missing-vocabulary.json"), "--output", str(output),
    ])
    with pytest.raises(ValueError, match="only supported for CMU"):
        importer.main()
    assert not output.exists()


@pytest.mark.parametrize("old_group", [None, "old-group"])
def test_explicit_group_relogs_existing_outputs_with_missing_or_different_group(bundle, monkeypatch, old_group):
    folder, _ = bundle
    client, run, artifact = fake_wandb(monkeypatch, "new-group-run")
    previous = {"url": "https://wandb.ai/chefOST/vidosc%20-%20full%20evals/runs/old-run"}
    if old_group is not None:
        previous["group"] = old_group
    (folder / "wandb_inference_receipt.json").write_text(json.dumps(previous))
    invoke(monkeypatch, folder, "--group", "full-r2-v1")
    assert client.init.call_args.kwargs["group"] == "full-r2-v1"
    assert artifact.add_file.call_count == 8
    assert json.loads((folder / "wandb_inference_receipt.json").read_text())["group"] == "full-r2-v1"
    backups = list(folder.glob("wandb_inference_receipt.*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == previous


def test_explicit_matching_group_reuses_receipt(egtea_bundle, monkeypatch):
    folder, _ = egtea_bundle
    client, run, _ = fake_wandb(monkeypatch, "existing-run")
    receipt = {"url": run.url, "group": "full-r2-v1"}
    path = folder / "wandb_inference_receipt.json"
    path.write_text(json.dumps(receipt))
    invoke(monkeypatch, folder, "--group", "full-r2-v1")
    client.init.assert_not_called()
    assert json.loads(path.read_text()) == receipt
