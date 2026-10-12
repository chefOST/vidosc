"""README precision: frames with >=1 matching predicted label / evaluated frames.
Evaluated frames = sampled frames where the object has >=1 ground-truth state.
Detail is pending a definition and is not computed."""


def score(pred, gt_sets, candidates):
    """pred: predicted state per frame; gt_sets: GT state set per frame.

    Returns counts plus two diagnostics (not README metrics):
      oracle: share of evaluated frames where some GT state is among the candidates (upper bound
              for any method choosing from these candidates);
      chance: expected precision of a uniform random pick from the candidates.
    """
    cand = set(candidates)
    ev = [(p, g) for p, g in zip(pred, gt_sets) if g]
    n = len(ev)
    return {
        "evaluated_frames": n,
        "correct_frames": sum(p in g for p, g in ev),
        "oracle_frames": sum(bool(g & cand) for _, g in ev),
        "chance_frames": sum(len(g & cand) / len(cand) for _, g in ev) if cand else 0.0,
    }


def summarize(counts):
    n = counts["evaluated_frames"]
    return {
        "precision": counts["correct_frames"] / n if n else float("nan"),
        "oracle": counts["oracle_frames"] / n if n else float("nan"),
        "chance": counts["chance_frames"] / n if n else float("nan"),
        "evaluated_frames": n,
    }
