"""Prepare full-video active-object extraction, inference and mapping stages."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", default="S29_Salad_7150991-1")
    p.add_argument("--active-object", action="store_true", default=True,
                   help="One unnamed hand-associated stream per video (the only supported mode)")
    p.add_argument("--experiment", default="cmu30-active-object-v1")
    args = p.parse_args()
    if not args.experiment.replace("-", "").replace("_", "").isalnum():
        p.error("Experiment must be a simple directory name")
    rows = json.loads((ROOT / "runs/batch-preparation/inventory.json").read_text())["videos"]
    row = next(r for r in rows if r["video_id"] == args.video_id)
    if not row["available"] or not row.get("metadata_frame_count"):
        p.error("Download and inspect the video before preparing an executable plan")
    out = ROOT / "runs/batches" / args.experiment / args.video_id
    model = out / "model"
    python = str(ROOT / ".venv/bin/python")
    checkpoint = ROOT / "checkpoints/changeit/state_best.ckpt"
    iv = Path.home() / "Downloads/InternVideo-MM-L-14.ckpt"
    mapping = ROOT / "configs/changeit_moscato_alias_mapping.json"
    vocab = ROOT / "data/moscato/vocabulary/state_dict.json"
    code = [ROOT / name for name in (
        "moscato/__init__.py", "moscato/baseline.py", "moscato/checkpoint_adapter.py",
        "moscato/extract_internvideo.py", "moscato/hand_object.py",
        "moscato/import_modal_predictions.py", "moscato/object_crops.py",
        "moscato/progress_cache.py", "scripts/modal_clip.py", "scripts/modal_detector.py",
    )]
    stages = []

    def add(name, command, inputs, outputs, env=None, modal=False):
        stages.append(dict(name=name, command=list(map(str, command)),
            inputs=list(map(str, inputs + code)), outputs=list(map(str, outputs)),
            receipt=str(out / f"{name}.done.json"), env=env or {}, modal=modal))

    add("model", [python, "-m", "modal", "run", "scripts/modal_clip.py"],
        [Path(row["local_video"]), iv, checkpoint],
        [model / n for n in ["feature_manifest.json", "combined_features.pth.tar", "object_features.pth.tar", "raw_probabilities.npz", "raw_logits.npz"]],
        {"VIDOSC_VIDEO": row["local_video"], "VIDOSC_VIDEO_ID": args.video_id,
         "VIDOSC_FRAME_COUNT": str(row["metadata_frame_count"]), "VIDOSC_FPS": str(row["fps"]),
         "VIDOSC_OUTPUT": str(model), "VIDOSC_TIMEOUT": "7200"}, True)
    add("mapping", [python, "-m", "moscato.import_modal_predictions", "--run", model,
        "--mapping", mapping, "--state-dict", vocab, "--output", model / "predictions"],
        [model / "feature_manifest.json", model / "combined_features.pth.tar", model / "raw_probabilities.npz", mapping, vocab],
        [model / "predictions" / n for n in ["manifest.json", "probabilities.npz", "predictions_by_second.csv"]])
    plan = ROOT / "runs/batch-preparation" / f"{args.experiment}-{args.video_id}.json"
    plan.write_text(json.dumps({"jobs": [{"name": args.video_id + ":active-object", "stages": stages}],
        "note": "Global features, detection, object features and inference run on Modal CPU. No W&B uploads, metrics or temporal postprocessing."}, indent=2) + "\n")
    print(plan)


if __name__ == "__main__":
    main()
