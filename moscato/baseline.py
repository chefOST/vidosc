"""VidOSC-only transfer evaluation. See MOSCATO_BASELINE.md for the protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    text = Path(path).read_text()
    if text.startswith("version https://git-lfs"):
        raise ValueError(f"{path} is a Git LFS pointer; download its contents first")
    return json.loads(text)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def project_scores(raw, mapping, vocabulary):
    """A fixed many-to-many mapping, declared before looking at evaluation GT.

    Source classes are mutually exclusive softmax outcomes. Summing disjoint
    source probabilities gives each target state's marginal probability.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Invalid source probabilities")
    if not np.allclose(raw.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Source probabilities must sum to one")
    if set(mapping) != {str(i) for i in range(raw.shape[1])}:
        raise ValueError("Mapping must explicitly cover every checkpoint output index (use [] for unmapped classes)")
    projected = np.zeros((len(raw), len(vocabulary)))
    indices = {state: i for i, state in enumerate(vocabulary)}
    for index, states in mapping.items():
        if not isinstance(states, list) or any(not isinstance(s, str) for s in states):
            raise ValueError("Each mapping value must be a list of canonical state strings")
        if len(states) != len(set(states)):
            raise ValueError("Duplicate target labels in mapping")
        for state in states:
            if state not in indices:
                raise ValueError(f"Noncanonical mapped state: {state}")
            projected[:, indices[state]] += raw[:, int(index)]
    return projected


def align_scores(scores, starts, clip_start, clip_end, annotation_fps, annotation_count):
    """Sample predictions on the annotation clock; never assume raw video FPS."""
    starts = np.asarray(starts, dtype=float)
    if (len(starts) != len(scores) or len(starts) == 0 or not np.isfinite(starts).all()
            or starts[0] != 0 or (np.diff(starts) <= 0).any()):
        raise ValueError("Feature interval starts must begin at zero and strictly increase")
    if (not np.isfinite([clip_start, clip_end, annotation_fps]).all()
            or annotation_fps <= 0 or clip_start < 0 or clip_end <= clip_start):
        raise ValueError("Invalid clip interval or annotation FPS")
    if starts[-1] >= clip_end - clip_start:
        raise ValueError("Last feature interval starts beyond the clip")
    if clip_end > annotation_count / annotation_fps + 1e-8:
        raise ValueError("Clip extends beyond the annotation timeline")
    # Evaluate annotation samples whose timestamps lie in [clip_start, clip_end).
    first = int(np.ceil(clip_start * annotation_fps - 1e-8))
    end = int(np.ceil(clip_end * annotation_fps - 1e-8))
    indices = np.arange(first, end)
    if not len(indices):
        raise ValueError("Clip contains no annotation samples")
    local_times = indices / annotation_fps - clip_start
    bins = np.searchsorted(starts, local_times + 1e-9, side="right") - 1
    return indices, bins, scores[bins]


def predict_run(args):
    from .checkpoint_adapter import predict
    manifest = read(args.feature_manifest)
    required = {"video_id", "object", "clip_start_seconds", "clip_end_seconds",
                "interval_starts_seconds", "feature_backbone", "feature_layout", "video_sha256"}
    if not required <= manifest.keys():
        raise ValueError(f"Feature manifest missing {sorted(required - manifest.keys())}")
    if "annotations_path" in manifest:
        raise ValueError("Prediction manifest must not carry a ground-truth path")
    mapping = read(args.mapping)
    if mapping.get("ground_truth_used_to_define_mapping") is not False or not mapping.get("rationale"):
        raise ValueError("Mapping must document its rationale and assert no test-GT-derived mapping")
    vocabulary = sorted(read(args.state_dict)["s2i"])
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    state, action, info = predict(args.checkpoint, args.features, args.upstream, args.heads)
    if mapping.get("checkpoint_sha256") != digest(args.checkpoint):
        raise ValueError("Mapping checkpoint digest mismatch; verify output class indices")
    scores = project_scores(state, mapping["source_index_to_states"], vocabulary)
    # Validate intervals independently of GT before committing prediction artifacts.
    duration = float(manifest["clip_end_seconds"]) - float(manifest["clip_start_seconds"])
    starts = np.asarray(manifest["interval_starts_seconds"], dtype=float)
    if (len(starts) != len(state) or starts[0] != 0 or not np.isfinite(starts).all()
            or (np.diff(starts) <= 0).any() or starts[-1] >= duration
            or not np.isfinite(duration) or float(manifest["clip_start_seconds"]) < 0):
        raise ValueError("Feature intervals do not match model output or clip duration")
    out.mkdir(parents=True)
    np.savez(out / "probabilities.npz", state=state, mapped=scores,
             **({"action": action} if action is not None else {}))
    write(out / "manifest.json", {
        "method": "vidosc-fixed-mapping-transfer", "ground_truth_used_for_inference": False,
        "feature_manifest": manifest, "model": info, "transformer_heads": args.heads,
        "mapping": mapping, "vocabulary": vocabulary, "decoder": "raw softmax; no ordering",
        "checkpoint_sha256": digest(args.checkpoint), "features_sha256": digest(args.features),
        "state_dict_sha256": digest(args.state_dict), "model_source_sha256": digest(Path(args.upstream) / "model.py"),
        "probabilities_sha256": digest(out / "probabilities.npz"),
        "feature_compatibility_verified": False,
    })
    print(f"Predictions saved to {out}; ground truth was not loaded.")


def evaluate_run(args):
    from .baseline_metrics import evaluate_scores
    run = Path(args.run)
    manifest = read(run / "manifest.json")
    if manifest.get("ground_truth_used_for_inference") is not False:
        raise ValueError("Missing no-ground-truth inference provenance")
    if digest(run / "probabilities.npz") != manifest["probabilities_sha256"]:
        raise ValueError("Prediction artifact changed after inference")
    if digest(args.state_dict) != manifest["state_dict_sha256"]:
        raise ValueError("Vocabulary changed after inference")
    timing = read(args.timing)
    if timing.get("verified") is not True or not timing.get("evidence"):
        raise ValueError("Timing needs verified=true and evidence from matching video/annotations")
    fm = manifest["feature_manifest"]
    if timing.get("video_id") != fm["video_id"] or timing.get("video_sha256") != fm["video_sha256"]:
        raise ValueError("Timing metadata refers to a different source video")
    annotations = read(args.annotations)
    timeline = annotations[fm["video_id"]]["state"][fm["object"]]
    if len(timeline) != timing["annotation_count"]:
        raise ValueError("Annotation count does not match verified timing")
    scores = np.load(run / "probabilities.npz", allow_pickle=False)["mapped"]
    indices, bins, aligned = align_scores(scores, fm["interval_starts_seconds"],
        fm["clip_start_seconds"], fm["clip_end_seconds"], timing["annotation_fps"], len(timeline))
    vocab = manifest["vocabulary"]
    lookup = {s: i for i, s in enumerate(vocab)}
    aliases = read(args.state_dict).get("all2one", {})
    truth = np.zeros_like(aligned, dtype=int)
    for row, index in enumerate(indices):
        labels = timeline[index]
        if not isinstance(labels, list):
            raise ValueError("Use the string-label ground-truth JSON, not the indexed variant")
        for state in labels:
            canonical = aliases.get(state, state)
            if canonical not in lookup:
                raise ValueError(f"Unknown ground-truth state: {state}")
            truth[row, lookup[canonical]] = 1
    metrics = evaluate_scores(truth, aligned, args.threshold)
    mapped_labels = {s for values in manifest["mapping"]["source_index_to_states"].values() for s in values}
    metrics.update({"method": manifest["method"], "video_id": fm["video_id"], "object": fm["object"],
        "scope": "one video/object clip, fixed-mapping transfer baseline; not full MOSCATO benchmark",
        "state_order": vocab, "mapped_states": sorted(mapped_labels),
        "unmapped_gt_states": [vocab[i] for i in np.flatnonzero(truth.any(axis=0)) if vocab[i] not in mapped_labels],
        "annotation_sha256": digest(args.annotations), "timing": timing,
        "predictions_sha256": digest(run / "probabilities.npz"),
        "prediction_manifest_sha256": digest(run / "manifest.json")})
    # Unmapped GT states remain negatives in predictions, and count as misses.
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a fresh scoring output directory: {output}")
    output.mkdir(parents=True)
    write(output / "evaluation.json", metrics)
    with (output / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["annotation_index", "timestamp_seconds", "vidosc_timestep", "evaluated", "gt_states", "predicted_states"])
        for row, index in enumerate(indices):
            writer.writerow([int(index), float(index / timing["annotation_fps"]), int(bins[row]),
                int(truth[row].any()), "|".join(vocab[i] for i in np.flatnonzero(truth[row])),
                "|".join(vocab[i] for i in np.flatnonzero(aligned[row] >= args.threshold))])
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-checkpoint")
    inspect.add_argument("--checkpoint", required=True)
    predict = commands.add_parser("predict")
    for name in ("checkpoint", "features", "feature-manifest", "mapping", "state-dict", "upstream", "output"):
        predict.add_argument("--" + name, required=True)
    predict.add_argument("--heads", type=int, default=4)
    evaluate = commands.add_parser("evaluate")
    for name in ("run", "annotations", "state-dict", "timing", "output"):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "predict":
        predict_run(args)
    elif args.command == "evaluate":
        evaluate_run(args)
    else:
        from .checkpoint_adapter import read_weights, inspect_weights
        info = inspect_weights(read_weights(args.checkpoint))
        info["checkpoint_sha256"] = digest(args.checkpoint)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
