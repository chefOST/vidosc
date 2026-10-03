"""Explicit multi-label metrics for a VidOSC-to-MOSCATO transfer baseline.

These are independently implemented metrics, not a claim of official evaluator
parity. Empty annotation rows are excluded; gaps split temporal segments.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def segments(binary, valid):
    result = []
    start = None
    for i in range(len(binary) + 1):
        active = i < len(binary) and valid[i] and binary[i]
        if active and start is None:
            start = i
        if not active and start is not None:
            result.append((start, i))
            start = None
    return result


def segment_counts(truth, predictions, valid, overlap):
    """One-to-one same-label temporal IoU matching, independently per video."""
    tp = fp = fn = 0
    for c in range(truth.shape[1]):
        gt = segments(truth[:, c], valid)
        pred = segments(predictions[:, c], valid)
        used = set()
        for a, b in pred:
            candidates = []
            for j, (x, y) in enumerate(gt):
                if j not in used:
                    intersection = max(0, min(b, y) - max(a, x))
                    union = max(b, y) - min(a, x)
                    candidates.append((intersection / union, j))
            best, index = max(candidates, default=(0, -1))
            if best >= overlap:
                used.add(index)
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0}


def evaluate_scores(truth, scores, threshold=0.5):
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if truth.ndim != 2 or truth.shape != scores.shape or not truth.shape[1]:
        raise ValueError("Expected matching nonempty [time, state] matrices")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Ground truth must be binary multi-label indicators")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must lie strictly between 0 and 1")
    valid = truth.any(axis=1)
    if not valid.any():
        raise ValueError("No annotated frames to evaluate")
    y, p = truth[valid], scores[valid]
    curve = []
    for t in np.arange(1, 10) / 10:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        curve.append({"threshold": float(t), "precision": float(precision),
                      "recall": float(recall), "f1": float(f1)})
    positive = np.flatnonzero(y.any(axis=0))
    ap = {int(c): float(average_precision_score(y[:, c], p[:, c])) for c in positive}
    return {
        "evaluated_frames": int(valid.sum()), "skipped_empty_frames": int((~valid).sum()),
        "f1_max": max(curve, key=lambda point: point["f1"]), "threshold_curve": curve,
        "map": float(np.mean(list(ap.values()))), "per_class_ap": ap,
        "map_policy": "macro mean over classes with positive GT in this evaluation",
        "decision_threshold": threshold,
        "segment_f1": {str(t): segment_counts(truth, scores >= threshold, valid, t)
                       for t in (0.1, 0.25, 0.5)},
        "policy": "empty GT skipped; unknown gaps split segments; half-open intervals; scores in [0,1]",
        "official_evaluator_parity": False,
    }
