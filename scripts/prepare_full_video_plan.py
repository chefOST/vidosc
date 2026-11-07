"""Prepare R2 full-video inference and, for CMU only, fixed label mapping."""
import argparse
import json
import math
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moscato.paths import checkpoint_path, state_vocabulary_path
DEFAULT_EXPERIMENT = "full-r2-v1"
DEFAULT_TIMEOUT = 21600


def validate_experiment(value):
    if not value or not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Experiment must be a simple directory name")
    return value


def validate_timeout(value):
    if not isinstance(value, int) or isinstance(value, bool) or not 60 <= value <= 86400:
        raise ValueError("Timeout must be between 60 and 86400 seconds")
    return value


def build_plan(row, r2, experiment=DEFAULT_EXPERIMENT, timeout=DEFAULT_TIMEOUT, root=None):
    root = ROOT if root is None else Path(root)
    validate_experiment(experiment)
    validate_timeout(timeout)
    video_id = row["video_id"]
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", video_id):
        raise ValueError("Video ID must be a simple file name")
    dataset = str(row["dataset"]).upper()
    if dataset not in {"CMU", "EGTEA"}:
        raise ValueError(f"Unsupported release dataset: {dataset}")
    if row.get("available") is not True:
        raise ValueError(f"Video is unavailable: {video_id}")
    frames, fps = row["metadata_frame_count"], row["fps"]
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise ValueError(f"Missing positive frame count: {video_id}")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Missing positive finite FPS: {video_id}")
    size, sha = row["bytes"], row["sha256"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"Missing positive video size: {video_id}")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        raise ValueError(f"Missing SHA-256 fingerprint: {video_id}")
    key = row["r2_key"]
    prefix = r2["prefix"].rstrip("/") + "/"
    if not isinstance(key, str) or not key.startswith(prefix):
        raise ValueError(f"Video key is outside the release prefix: {video_id}")
    if not r2.get("bucket") or not r2.get("endpoint"):
        raise ValueError("Inventory is missing R2 bucket or endpoint")

    out = root / "runs/batches" / experiment / video_id
    model = out / "model"
    python = str(root / ".venv/bin/python")
    checkpoint = checkpoint_path("changeit/state_best.ckpt", root)
    iv = checkpoint_path("internvideo/InternVideo-MM-L-14.ckpt", root)
    mapping = root / "configs/changeit_moscato_alias_mapping.json"
    vocab = state_vocabulary_path(root)
    code = [root / name for name in (
        "moscato/__init__.py", "moscato/paths.py", "moscato/baseline.py", "moscato/checkpoint_adapter.py",
        "moscato/extract_internvideo.py", "moscato/hand_object.py", "moscato/r2_inputs.py",
        "moscato/import_modal_predictions.py", "moscato/object_crops.py",
        "moscato/progress_cache.py", "scripts/modal_clip.py", "scripts/modal_detector.py",
        "scripts/inventory_r2.py", "configs/r2_release.json",
    )]
    stages = []

    def add(name, command, inputs, outputs, env=None, modal=False):
        stages.append(dict(name=name, command=list(map(str, command)),
            inputs=list(map(str, inputs + code)), outputs=list(map(str, outputs)),
            receipt=str(out / f"{name}.done.json"), env=env or {}, modal=modal))

    add("model", [python, "-m", "modal", "run", "scripts/modal_clip.py"],
        [root / "runs/batch-preparation/inventory.json", iv, checkpoint],
        [model / name for name in ("feature_manifest.json", "combined_features.pth.tar",
            "object_features.pth.tar", "raw_probabilities.npz", "raw_logits.npz")],
        {"MODAL_IMAGE_BUILDER_VERSION": "2025.06",
         "VIDOSC_R2_KEY": key, "VIDOSC_VIDEO_SHA256": sha.lower(),
         "VIDOSC_VIDEO_BYTES": str(size), "VIDOSC_VIDEO_ID": video_id,
         "VIDOSC_FRAME_COUNT": str(frames), "VIDOSC_FPS": str(fps), "VIDOSC_DATASET": dataset,
         "VIDOSC_R2_BUCKET": r2["bucket"], "VIDOSC_R2_ENDPOINT": r2["endpoint"],
         "VIDOSC_R2_PREFIX": prefix, "VIDOSC_OUTPUT": str(model), "VIDOSC_TIMEOUT": str(timeout)}, True)
    if dataset == "CMU":
        add("mapping", [python, "-m", "moscato.import_modal_predictions", "--run", model,
            "--mapping", mapping, "--state-dict", vocab, "--output", model / "predictions"],
            [model / "feature_manifest.json", model / "combined_features.pth.tar",
             model / "raw_probabilities.npz", mapping, vocab],
            [model / "predictions" / name for name in
             ("manifest.json", "probabilities.npz", "predictions_by_second.csv")])
    return {"jobs": [{"name": video_id + ":active-object", "stages": stages}],
            "note": "R2 input, feature extraction and inference run on Modal CPU. "
                    "CMU mapping runs locally; EGTEA preserves raw outputs only. "
                    "No W&B uploads, metrics or temporal postprocessing."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = p.parse_args()
    inventory = json.loads((ROOT / "runs/batch-preparation/inventory.json").read_text())
    rows = [row for row in inventory["videos"] if row["video_id"] == args.video_id]
    if len(rows) != 1:
        p.error("Video ID must match exactly one inventory entry")
    try:
        plan = build_plan(rows[0], inventory["r2"], args.experiment, args.timeout)
    except (KeyError, TypeError, ValueError) as error:
        p.error(str(error))
    path = ROOT / "runs/batch-preparation" / f"{args.experiment}-{args.video_id}.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
