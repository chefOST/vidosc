"""Upload saved VidOSC exploratory results, without rerunning inference."""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "RESULTS.md", "feature_manifest.json", "raw_probabilities.npz",
    "predictions/manifest.json", "predictions/probabilities.npz",
    "predictions/predictions_by_second.csv", "scoring/evaluation.json",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity")
    parser.add_argument("--project", default="vidosc-moscato")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/modal-clip/20261011T184132341252Z")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--full-video", action="store_true")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    for filename in FILES:
        if not (folder / filename).is_file():
            raise FileNotFoundError(folder / filename)
    scores = json.loads((folder / "scoring/evaluation.json").read_text())
    metrics = {
        "eval/frame_micro_f1_max": scores["f1_max"]["f1"],
        "eval/f1_max_threshold": scores["f1_max"]["threshold"],
        "eval/mAP": scores["map"],
        "eval/segment_f1_iou_0_5": scores["segment_f1"]["0.5"]["f1"],
        "eval/evaluated_frames": scores["evaluated_frames"],
        "eval/skipped_empty_frames": scores["skipped_empty_frames"],
    }
    with (folder / "predictions/predictions_by_second.csv").open() as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        rows = list(reader)
    config = {
        "model": "VidOSC", "checkpoint_training_dataset": "ChangeIt",
        "evaluation_dataset": "MOSCATO", "training_performed": False, "full_video": args.full_video,
        "prediction_intervals": len(rows), "execution_hardware": "Modal CPU",
        **{key: scores[key] for key in [
            "video_id", "object", "scope", "method", "timing_verified",
            "provisional", "official_evaluator_parity", "unmapped_gt_states",
            "predictions_sha256", "annotation_sha256",
        ]},
    }
    if args.dry_run:
        print(json.dumps({"config": config, "metrics": metrics, "files": FILES}, indent=2))
        return
    if not args.entity:
        parser.error("--entity must specify your W&B team or username")
    receipt = folder / "wandb_upload_receipt.json"
    if receipt.exists():
        raise RuntimeError(f"Already uploaded; see {receipt}")

    import wandb

    with wandb.init(
        entity=args.entity, project=args.project, group=args.group,
        name=f"{scores['video_id']}-{len(rows)}-intervals-preliminary",
        job_type="evaluation-upload", config=config,
        tags=["vidosc", "moscato", "preliminary", "full-video" if args.full_video else "single-clip"],
        notes=("Retrospective upload of saved predictions; no inference runs here. "
               "Not a full MOSCATO benchmark. Timing is provisional; crop parity "
               "is unverified. Unsupported classes and tied scores affect mAP. "
               "F1-Max selects a threshold on this clip. See RESULTS.md."),
    ) as run:
        run.summary.update(metrics)
        run.log({"predictions/by_second": wandb.Table(columns=columns, data=rows)})
        curve = scores["threshold_curve"]
        table = wandb.Table(
            columns=["threshold", "precision", "recall", "f1"],
            data=[[p[k] for k in ["threshold", "precision", "recall", "f1"]] for p in curve],
        )
        run.log({"eval/threshold_curve": table})
        artifact = wandb.Artifact(
            f"vidosc-{scores['video_id']}-evaluation", type="evaluation",
            metadata={"provisional": scores["provisional"], "scope": scores["scope"]},
        )
        for filename in FILES:
            artifact.add_file(str(folder / filename), name=filename)
        run.log_artifact(artifact).wait()
        result = {"run_url": run.url, "run_id": run.id,
                  "entity": args.entity, "project": args.project}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Uploaded: {result['run_url']}")


if __name__ == "__main__":
    main()
