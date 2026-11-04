"""Artifact IO, fingerprints and the fixed ChangeIt-to-CMU probability mapping."""
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
