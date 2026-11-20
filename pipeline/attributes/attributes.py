'Attribute vocabularies and prediction conversion for object state inference.'
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


