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
    state, action, info = predict(args.checkpoint, args.features, args.upstream, args.heads, pad_to=getattr(args, "pad_to", 0))
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




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-checkpoint")
    inspect.add_argument("--checkpoint", required=True)
    predict = commands.add_parser("predict")
    for name in ("checkpoint", "features", "feature-manifest", "mapping", "state-dict", "upstream", "output"):
        predict.add_argument("--" + name, required=True)
    predict.add_argument("--heads", type=int, default=4)
    predict.add_argument("--pad-to", type=int, default=0, help="Explicit temporal padding; original ChangeIt loader uses 900")
    args = parser.parse_args()
    if args.command == "predict":
        predict_run(args)
    else:
        from .checkpoint_adapter import read_weights, inspect_weights
        info = inspect_weights(read_weights(args.checkpoint))
        info["checkpoint_sha256"] = digest(args.checkpoint)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
