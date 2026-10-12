"""Near-synonym state clusters (prediction side only; README precision is unchanged).

MOSCATO's state_dict.json already maps raw synonyms to canonical states. Clusters group canonical
states that still look alike (state_clusters/<dataset>.json, hand-drafted, each state in at most one
cluster, unlisted states are singletons). The pipeline decodes over an object's clusters, then picks
one member per frame: 'prior' = the member most frequent for this recipe and object in training
pseudo-labels, 'model' = the member the zero-shot scores prefer at that frame.
"""
import json
import os

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_clusters")


def load(dataset):
    clusters = json.load(open(os.path.join(DIR, f"{dataset}.json")))
    seen = [s for members in clusters.values() for s in members]
    assert len(seen) == len(set(seen)), "a state is listed in more than one cluster"
    return {s: name for name, members in clusters.items() for s in members}


def group(states, state_to_cluster):
    """The object's candidates grouped by cluster, as lists of indices into `states`."""
    groups = {}
    for k, s in enumerate(states):
        groups.setdefault(state_to_cluster.get(s, s), []).append(k)
    return list(groups.values())


def aggregate(p, groups):
    """[T, K] state probabilities -> [T, C] cluster probabilities (sum over members)."""
    return np.stack([p[:, g].sum(1) for g in groups], 1)


def expand(cluster_idx, groups, states, zs, prior, mode):
    """One member state per frame for a decoded cluster sequence."""
    out = []
    for t, c in enumerate(cluster_idx):
        g = groups[c]
        if mode == "prior":
            k = max(g, key=lambda j: (prior.get(states[j], 0), zs[t, j]))
        else:
            k = max(g, key=lambda j: zs[t, j])
        out.append(states[k])
    return out
