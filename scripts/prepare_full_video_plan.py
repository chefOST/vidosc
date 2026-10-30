"""Prepare a full-video pilot plan from local inventory, without running models."""
import argparse
import math
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", default="S29_Salad_7150991-1")
    p.add_argument("--object", default="vegetables")
    p.add_argument("--active-object", action="store_true", help="One unnamed hand-associated stream per video, no GT object selection")
    p.add_argument("--timing", type=Path, default=ROOT / "configs/s29_provisional_timing.json")
    p.add_argument("--experiment", default="full-video-pilot-v1")
    p.add_argument("--with-metrics", action="store_true",
                   help="Opt in to evaluation; full runs default to outputs only")
    args = p.parse_args()
    if not args.experiment.replace("-", "").replace("_", "").isalnum():
        p.error("Experiment must be a simple directory name")
    rows = json.loads((ROOT / "runs/batch-preparation/inventory.json").read_text())["videos"]
    row = next(r for r in rows if r["video_id"] == args.video_id)
    target = None if args.active_object else next(o for o in row["objects"] if o["object"] == args.object)
    if args.active_object:
        args.object = "active-object"
        if args.with_metrics:
            p.error("Active-object streams cannot be scored as a named GT object")
    if not row["available"] or not row.get("metadata_frame_count"):
        p.error("Download and inspect the video before preparing an executable plan")
    if target is not None and row["metadata_frame_count"] != target["annotation_count"]:
        p.error("Video/annotation counts differ; resolve timing first")
    if row["duration_seconds"] > 900:
        p.error("Videos over 900 seconds need an agreed chunking protocol")
    timing = {} if args.active_object else json.loads(args.timing.read_text())
    if target is not None and (timing["video_id"] != args.video_id or timing["annotation_count"] != target["annotation_count"]):
        p.error("Timing file does not match video/object")
    out = ROOT / "runs/batches" / args.experiment / args.video_id
    glob, model = out / "global", out / "model"
    python = str(ROOT / ".venv/bin/python")
    checkpoint = ROOT / "checkpoints/changeit/state_best.ckpt"
    iv = Path.home() / "Downloads/InternVideo-MM-L-14.ckpt"
    mapping = ROOT / "configs/changeit_moscato_alias_mapping.json"
    vocab = ROOT / "data/moscato/vocabulary/state_dict.json"
    annotations = ROOT / "data/moscato/annotations/gt_annotations_cmu.json"
    code = sorted((ROOT / "moscato").glob("*.py")) + [ROOT / "scripts/modal_clip.py", ROOT / "scripts/modal_detector.py"]
    stages = []

    def add(name, command, inputs, outputs, env=None, modal=False):
        stages.append(dict(name=name, command=list(map(str, command)),
            inputs=list(map(str, inputs + code)), outputs=list(map(str, outputs)),
            receipt=str(out / f"{name}.done.json"), env=env or {}, modal=modal))

    add("global", [python, "-m", "moscato.extract_internvideo", "--video", row["local_video"],
        "--video-id", args.video_id, "--object", args.object, "--checkpoint", iv,
        "--source", ROOT.parent / "internvideo-upstream", "--output", glob,
        "--start-frame", "0", "--end-frame", str(row["metadata_frame_count"]),
        "--fps", str(row["fps"]), "--device", "cpu"],
        [Path(row["local_video"]), iv], [glob / "global_features.pth.tar", glob / "feature_manifest.json"])
    add("model", [python, "-m", "modal", "run", "scripts/modal_clip.py"],
        [Path(row["local_video"]), iv, checkpoint, glob / "global_features.pth.tar", glob / "feature_manifest.json"],
        [model / n for n in ["feature_manifest.json", "combined_features.pth.tar", "object_features.pth.tar", "raw_probabilities.npz", "raw_logits.npz"]],
        {"VIDOSC_VIDEO": row["local_video"], "VIDOSC_GLOBAL_DIR": str(glob), "VIDOSC_OUTPUT": str(model), "VIDOSC_TIMEOUT": str(min(3600, math.ceil(row["duration_seconds"] * 4.5 + 120)))}, True)
    add("mapping", [python, "-m", "moscato.import_modal_predictions", "--run", model,
        "--mapping", mapping, "--state-dict", vocab, "--output", model / "predictions"],
        [model / "feature_manifest.json", model / "combined_features.pth.tar", model / "raw_probabilities.npz", mapping, vocab],
        [model / "predictions" / n for n in ["manifest.json", "probabilities.npz", "predictions_by_second.csv"]])
    command = [python, "-m", "moscato.baseline", "evaluate", "--run", model / "predictions",
        "--annotations", annotations, "--state-dict", vocab, "--timing", args.timing.resolve(), "--output", model / "scoring"]
    if timing.get("verified") is not True:
        command.append("--allow-unverified-timing")
    add("scoring", command, [annotations, vocab, args.timing.resolve(), model / "predictions/manifest.json", model / "predictions/probabilities.npz"],
        [model / "scoring/evaluation.json", model / "scoring/per_frame.csv"])
    add("report", [python, "-m", "moscato.summarize_modal_run", "--run", model],
        [model / "feature_manifest.json", model / "predictions/manifest.json", model / "predictions/probabilities.npz", model / "scoring/evaluation.json"], [model / "RESULTS.md"])
    if not args.with_metrics:
        stages = [stage for stage in stages if stage["name"] not in {"scoring", "report"}]
    plan = ROOT / "runs/batch-preparation" / f"{args.experiment}-{args.video_id}.json"
    plan.write_text(json.dumps({"jobs": [{"name": args.video_id + ":" + args.object, "stages": stages}],
        "note": "Full-video CPU pilot; global features run locally, detection/object features/inference on Modal. No uploads. Metrics are opt-in; temporal postprocessing is absent."}, indent=2) + "\n")
    print(plan)


if __name__ == "__main__":
    main()
