#!/usr/bin/env python3
"""MOSCATO per-frame evaluation -- ported verbatim from the TubeletGraph branch.

Source: chefOST-ml, branch feat/tubeletgraph-moscato-cmu-s12,
``scripts/evaluate.py``.

`densify`, `f1_threshold_curve`, `evaluate_events`, `normalize_state`,
`canonical_states` and `write_csv` below are copied unchanged. Do not "improve"
them: TubeletGraph and VidOSC+Claude are only comparable because they are scored
by identical code under identical policy --

  * evaluation unit .... one raw MOSCATO frame
  * multi-label GT ..... a prediction is correct if it appears in the GT list
  * empty GT `[]` ...... frame skipped, counted in `skipped_empty_gt_frames`
  * temporal ........... initial state propagated, transitions forward-filled
  * vocabulary ......... state_dict `s2i`, with `all2one` alias folding
  * f1_max ............. best micro-F1 over global thresholds 0.1 .. 0.9

VidOSC+Claude reaches this module by expressing its segment predictions in the
same event schema TubeletGraph used (`initial_state` + `transitions`), so this
file needed no VidOSC-specific branch at all. See moscato/segments_to_events.py.

The VIDOSC-specific additions live at the bottom, after the ported section, and
only *add* columns to the CSV; they do not touch how any metric is computed.

This is the ONLY module in the package permitted to read gt_annotations_cmu.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import precision_recall_fscore_support


F1_MAX_THRESHOLDS = tuple(round(index / 10, 1) for index in range(1, 10))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_states(state_dict: dict[str, Any]) -> list[str]:
    return sorted(state.strip().lower() for state in state_dict["s2i"])


def normalize_state(state: str, state_dict: dict[str, Any]) -> str:
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"State must be a non-empty string; got {state!r}")
    normalized = state.strip().lower()
    aliases = {
        key.strip().lower(): value.strip().lower()
        for key, value in state_dict.get("all2one", {}).items()
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in set(canonical_states(state_dict)):
        raise ValueError(
            f"State {state!r} is not a canonical MOSCATO state or known alias"
        )
    return normalized


def densify(events: dict[str, Any], state_dict: dict[str, Any]) -> list[str]:
    frame_count = int(events["num_local_frames"])
    if frame_count <= 0:
        raise ValueError("num_local_frames must be positive")

    current_state = normalize_state(events.get("initial_state"), state_dict)
    dense: list[str] = []
    cursor = 0
    previous_transition = -1
    transitions = sorted(
        events.get("transitions", []), key=lambda item: int(item["local_frame"])
    )
    for transition in transitions:
        frame = int(transition["local_frame"])
        if not 0 <= frame < frame_count:
            raise ValueError(f"Transition frame {frame} is outside [0, {frame_count})")
        if frame == previous_transition:
            raise ValueError(f"Duplicate transition frame: {frame}")
        dense.extend([current_state] * (frame - cursor))
        current_state = normalize_state(transition.get("state"), state_dict)
        cursor = frame
        previous_transition = frame
    dense.extend([current_state] * (frame_count - cursor))
    if len(dense) != frame_count:
        raise AssertionError("Densification produced the wrong number of frames")
    return dense


def f1_threshold_curve(
    y_true: list[list[int]],
    y_score: list[list[float]],
    thresholds: tuple[float, ...] = F1_MAX_THRESHOLDS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Return MOSCATO's global-threshold micro-F1 sweep and its maximum."""
    if not y_true or len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must contain the same non-zero frames")
    if not thresholds:
        raise ValueError("At least one F1-max threshold is required")

    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        thresholded = [
            [int(score >= threshold) for score in frame_scores]
            for frame_scores in y_score
        ]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            thresholded,
            average="micro",
            zero_division=0,
        )
        curve.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    best = max(curve, key=lambda point: (point["f1"], -point["threshold"]))
    return curve, best


def evaluate_events(
    annotations: dict[str, Any],
    state_dict: dict[str, Any],
    events: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_id = events["video_id"]
    object_name = events["object"]
    clip_start = int(events.get("clip_start_global", 0))
    if video_id not in annotations:
        raise KeyError(f"{video_id} not found in MOSCATO annotations")
    video_gt = annotations[video_id]
    if object_name not in video_gt["state"]:
        raise KeyError(
            f"{object_name!r} is unavailable; choose from {sorted(video_gt['state'])}"
        )

    gt_timeline = video_gt["state"][object_name]
    predictions = densify(events, state_dict)
    if clip_start < 0 or clip_start >= len(gt_timeline):
        raise ValueError(f"clip_start_global {clip_start} is outside the GT timeline")
    if clip_start + len(predictions) > len(gt_timeline):
        raise ValueError(
            "Predicted clip extends beyond MOSCATO: "
            f"{clip_start} + {len(predictions)} > {len(gt_timeline)}"
        )

    canonical = canonical_states(state_dict)
    label_to_idx = {label: index for index, label in enumerate(canonical)}
    rows: list[dict[str, Any]] = []
    y_true: list[list[int]] = []
    y_pred: list[list[int]] = []
    hits = 0

    for local_frame, prediction in enumerate(predictions):
        global_frame = clip_start + local_frame
        gt_states = [
            normalize_state(state, state_dict)
            for state in gt_timeline[global_frame]
        ]
        if not gt_states:
            rows.append(
                {
                    "local_frame": local_frame,
                    "global_frame": global_frame,
                    "prediction": prediction,
                    "gt_states": "",
                    "evaluated": 0,
                    "correct": "",
                }
            )
            continue

        correct = prediction in gt_states
        hits += int(correct)
        true_vector = [0] * len(canonical)
        pred_vector = [0] * len(canonical)
        for gt_state in gt_states:
            true_vector[label_to_idx[gt_state]] = 1
        pred_vector[label_to_idx[prediction]] = 1
        y_true.append(true_vector)
        y_pred.append(pred_vector)
        rows.append(
            {
                "local_frame": local_frame,
                "global_frame": global_frame,
                "prediction": prediction,
                "gt_states": "|".join(gt_states),
                "evaluated": 1,
                "correct": int(correct),
            }
        )

    evaluated = len(y_true)
    if evaluated == 0:
        raise RuntimeError("No annotated frames were evaluated")
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # The adapter emits one hard canonical state, not calibrated probabilities.
    # Treat its one-hot labels as scores so the benchmark's official global
    # threshold sweep remains reproducible and its limitation stays explicit.
    f1_curve, f1_best = f1_threshold_curve(
        y_true,
        [[float(value) for value in frame] for frame in y_pred],
    )
    accuracy = hits / evaluated
    summary = {
        "video_id": video_id,
        "object": object_name,
        "clip_start_global": clip_start,
        "predicted_frames": len(predictions),
        "evaluated_frames": evaluated,
        "skipped_empty_gt_frames": len(predictions) - evaluated,
        "correct_frames": hits,
        "precision": float(p_micro),
        "accuracy": accuracy,
        "f1": float(f1_micro),
        "f1_max": f1_best["f1"],
        "f1_max_threshold": f1_best["threshold"],
        "frame_hit_accuracy": accuracy,
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "f1_threshold_curve": f1_curve,
        "policy": {
            "primary_metric_namespace": "word_accuracy",
            "word_unit": "exact canonical MOSCATO state label per frame",
            "multi_label": "prediction is correct if present in GT state list",
            "empty_gt": "skip",
            "temporal": "initial propagation plus transition forward fill",
            "f1_max": "maximum micro-F1 at one global threshold in [0.1, ..., 0.9]",
            "prediction_scores": "hard one-hot labels; no calibrated confidence available",
        },
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "local_frame",
                "global_frame",
                "prediction",
                "gt_states",
                "evaluated",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# VIDOSC-pipeline additions.
#
# Everything above is the ported TubeletGraph evaluator and computes the
# metrics. Everything below only wires a VIDOSC run directory into it and joins
# the scores back onto the prediction CSV for inspection. No metric is redefined
# here.
# ---------------------------------------------------------------------------

SCORED_CSV_COLUMNS = [
    "local_frame",
    "global_moscato_frame",
    "timestamp",
    "vidosc_timestep",
    "vidosc_state",
    "claude_description",
    "mapped_moscato_state",
    "ground_truth_state",
    "evaluated",
    "correct",
]


def read_prediction_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    """Evaluate a saved VIDOSC+Claude run against MOSCATO ground truth.

    Ground truth is opened here and nowhere else. The predictions being scored
    were written to disk by an earlier stage that never had access to it; their
    digests are recorded in the summary so the scored artifacts can be traced
    back to the exact prediction files.
    """
    from .config import load_config, resolve, sha256, write_json

    config = load_config(config_path)
    run_dir = Path(run_dir)

    events_path = run_dir / "moscato_predictions.json"
    predictions_csv = run_dir / "per_frame_predictions.csv"
    for required in (events_path, predictions_csv):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing; run the predict stage before evaluating."
            )

    events = load_json(events_path)
    if events.get("provenance", {}).get("ground_truth_used_for_inference") is not False:
        raise RuntimeError(
            "Refusing to evaluate predictions that do not assert "
            "ground_truth_used_for_inference=false"
        )

    annotations = load_json(resolve(config, "annotations_path"))
    state_dict = load_json(resolve(config, "state_dict_path"))
    rows, summary = evaluate_events(annotations, state_dict, events)

    # Join the scores onto the untouched prediction rows.
    by_frame = {int(row["local_frame"]): row for row in rows}
    prediction_rows = read_prediction_csv(predictions_csv)
    if len(prediction_rows) != len(rows):
        raise ValueError(
            f"per_frame_predictions.csv has {len(prediction_rows)} rows but the "
            f"evaluator produced {len(rows)}; the run is inconsistent."
        )

    scored: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        local_frame = int(prediction["local_frame"])
        score = by_frame[local_frame]
        if prediction["mapped_moscato_state"] != score["prediction"]:
            raise ValueError(
                f"Frame {local_frame}: CSV says "
                f"{prediction['mapped_moscato_state']!r} but densified events say "
                f"{score['prediction']!r}"
            )
        if int(prediction["global_moscato_frame"]) != int(score["global_frame"]):
            raise ValueError(
                f"Frame {local_frame}: global frame mismatch between the frame "
                "mapping and the evaluator"
            )
        scored.append(
            {
                "local_frame": local_frame,
                "global_moscato_frame": prediction["global_moscato_frame"],
                "timestamp": prediction["timestamp"],
                "vidosc_timestep": prediction["vidosc_timestep"],
                "vidosc_state": prediction["vidosc_state"],
                "claude_description": prediction["claude_description"],
                "mapped_moscato_state": prediction["mapped_moscato_state"],
                "ground_truth_state": score["gt_states"],
                "evaluated": score["evaluated"],
                "correct": score["correct"],
            }
        )

    scored_path = run_dir / "per_frame_predictions_scored.csv"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(scored)

    summary["method"] = "vidosc+claude"
    summary["run_dir"] = str(run_dir)
    summary["evaluated_artifacts"] = {
        "moscato_predictions.json": sha256(events_path),
        "per_frame_predictions.csv": sha256(predictions_csv),
    }
    summary["ground_truth_used_for_inference"] = False
    write_json(run_dir / "evaluation.json", summary)
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path)
    args = parser.parse_args()

    rows, summary = evaluate_events(
        load_json(args.annotations),
        load_json(args.state_dict),
        load_json(args.events),
    )
    write_csv(args.out_csv, rows)
    summary_path = args.out_summary or args.out_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Comparison CSV: {args.out_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
