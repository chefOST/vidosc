"""Attribute view of object states (proposal; README precision is unchanged).

A state is a set of attribute values, e.g. 'chopped' -> form=cut, 'in-pan' -> location=in_pan.
state_attributes/<dataset>.json defines the attributes (each a list of values, optionally ordered,
with a phrase per value for prompting) and maps every MOSCATO state to the values it asserts.
A frame's ground truth for an attribute is the union over its GT states; MOSCATO only asserts
states, so an attribute is unknown (not scored) on frames where no GT state sets it.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_attributes")


class Schema:
    def __init__(self, dataset):
        cfg = json.load(open(os.path.join(DIR, f"{dataset}.json")))
        self.attrs = {a: list(c["values"]) for a, c in cfg["attributes"].items()}
        self.phrases = {a: c["values"] for a, c in cfg["attributes"].items()}
        self.ordered = {a: c["ordered"] for a, c in cfg["attributes"].items()}
        self.states, self.overrides = cfg["states"], cfg.get("overrides", {})
        self.unmapped = set(cfg.get("unmapped", []))
        for s, av in self.states.items():
            for a, v in av.items():
                assert v in self.attrs[a], f"{s}: {a}={v} is not a value of {a}"

    def values_of(self, obj, state):
        """Attribute -> value asserted by one state of `obj` ({} for unmapped states)."""
        return self.overrides.get(obj, {}).get(state, self.states.get(state, {}))

    def frame_values(self, obj, states):
        """GT state set of one frame -> {attribute: set of values}."""
        out = defaultdict(set)
        for s in states:
            for a, v in self.values_of(obj, s).items():
                out[a].add(v)
        return out


def gt_values(schema, obj, gt_sets):
    """{attribute: [set of GT values per frame]} (empty set = not scored at that frame)."""
    per_frame = [schema.frame_values(obj, g) for g in gt_sets]
    return {a: [f.get(a, set()) for f in per_frame] for a in schema.attrs}


def value_counts(schema, ds, vid, obj):
    """{attribute: Counter(value -> frames)} from training pseudo-labels: the video's recipe for this
    object, else every recipe (as data.Moscato.state_prior)."""
    out = defaultdict(Counter)
    for s, n in ds.state_prior(vid, obj).items():
        for a, v in schema.values_of(obj, s).items():
            out[a][v] += n
    return out


def candidates(schema, ds, vid, obj, mode):
    """{attribute: candidate values} for one object.
    'seen': attributes and values the object takes in training pseudo-labels.
    'all': every attribute with all its values (the schema is the only prior)."""
    if mode == "all":
        return {a: list(vs) for a, vs in schema.attrs.items()}
    counts = value_counts(schema, ds, vid, obj)
    return {a: [v for v in schema.attrs[a] if counts[a][v]] for a in schema.attrs if counts[a]}


def from_states(schema, obj, probs, states, attr, values):
    """[T, K] state probabilities -> [T, V] probabilities over `values` of `attr` (sum over the
    states asserting each value, renormalized), or None when no candidate state sets `attr`."""
    asserted = [schema.values_of(obj, s).get(attr) for s in states]
    cols = [probs[:, [k for k, v2 in enumerate(asserted) if v2 == v]].sum(1) for v in values]
    p = np.stack(cols, 1)
    if not p.any():
        return None
    return (p + 1e-12) / (p + 1e-12).sum(1, keepdims=True)


def score(pred, gt, cands, order=None):
    """One attribute of one (video, object). pred: value or None per frame; gt: set per frame.
    Scored frames: GT sets the attribute. Correct: the prediction is one of the GT values.
    dist (ordered attributes, `order` = values low to high): rank gap to the nearest GT value over
    (len(order) - 1); a missing prediction counts as 1. oracle/chance: as metrics.score, over `cands`."""
    ev = [(p, g) for p, g in zip(pred, gt) if g]
    cand = set(cands or [])
    out = {"frames": len(ev), "correct": sum(p in g for p, g in ev),
           "oracle": sum(bool(g & cand) for _, g in ev),
           "chance": sum(len(g & cand) / len(cand) for _, g in ev) if cand else 0.0}
    if order:
        rank = {v: i for i, v in enumerate(order)}
        out["dist"] = sum(1.0 if p is None else min(abs(rank[p] - rank[v]) for v in g) / (len(order) - 1)
                          for p, g in ev)
    return out
