"""State discovery from an object's embedding track (proposal, not the README metric).

Input: per-frame embeddings X [T, D] of one object (frame or crop features) and visibility weights w [T].
Output: any number of states, each a unit centroid in the encoder's space plus the frame spans it covers.
No vocabulary: a state is a region of embedding space the object stays in, found by
  1. change points: exact optimal partition (DP) minimizing visibility-weighted within-segment squared
     error + `penalty` x noise per segment (noise = half the mean squared frame-to-frame step), segments >= `min_len`;
  2. merging segments whose centroids are within cosine distance `merge` (average linkage), so a state can recur.
"""
import numpy as np


def noise(X, w):
    d = ((X[1:] - X[:-1]) ** 2).sum(1)
    ww = np.minimum(w[1:], w[:-1])
    return 0.5 * float((ww * d).sum() / max(ww.sum(), 1e-6))


def changepoints(X, w, penalty, min_len=5):
    """Segment starts (including 0) of the optimal weighted piecewise-constant fit."""
    T = len(X)
    w = np.maximum(w, 1e-3)
    Pw = np.concatenate([[0], np.cumsum(w)])
    Pq = np.concatenate([[0], np.cumsum(w * (X ** 2).sum(1))])
    Px = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(w[:, None] * X, 0)])
    G = Px @ Px.T
    beta = penalty * noise(X, w) * 1.0
    F = np.full(T + 1, np.inf)
    F[0], back = -beta, np.zeros(T + 1, int)
    for j in range(min_len, T + 1):
        i = np.arange(0, j - min_len + 1)
        if j < T and T - j < min_len:  # the remainder could not form a segment
            continue
        sx = G[j, j] - 2 * G[i, j] + G[i, i]
        cost = (Pq[j] - Pq[i]) - sx / (Pw[j] - Pw[i])
        tot = F[i] + cost + beta
        k = int(np.argmin(tot))
        F[j], back[j] = tot[k], i[k]
    starts, j = [], T
    while j > 0:
        j = back[j]
        starts.append(j)
    return sorted(starts) if T >= min_len else [0]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def discover(X, w, penalty=20.0, min_len=5, merge=0.05):
    """Returns (labels [T] state id per frame, centroids [S, D] unit)."""
    T = len(X)
    starts = changepoints(X, w, penalty, min_len) + [T]
    segs = list(zip(starts[:-1], starts[1:]))
    wm = np.maximum(w, 1e-3)
    means = unit(np.stack([(wm[a:b, None] * X[a:b]).sum(0) for a, b in segs]))
    if len(segs) > 1 and merge > 0:
        from sklearn.cluster import AgglomerativeClustering
        ids = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=merge).fit_predict(means)
    else:
        ids = np.arange(len(segs))
    labels = np.zeros(T, int)
    for (a, b), k in zip(segs, ids):
        labels[a:b] = k
    _, labels = np.unique(labels, return_inverse=True)  # state ids in order of id
    cents = unit(np.stack([(wm[labels == s, None] * X[labels == s]).sum(0) for s in range(labels.max() + 1)]))
    return labels, cents


def vis_smooth(X, w, sigma=3.0):
    """Two-sided visibility-weighted Gaussian smoothing (frames the detector misses borrow from neighbours)."""
    if sigma <= 0:
        return unit(X)
    r = int(3 * sigma)
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    wm = np.maximum(w, 1e-3)
    num = np.stack([np.convolve(wm * X[:, j], g, "same") for j in range(X.shape[1])], 1)
    return unit(num / np.convolve(wm, g, "same")[:, None])


# --- evaluation of discovered states against GT state sets (frames with >= 1 GT state) ---

def gt_segments(gt_sets):
    """(evaluated frame indices, GT state-set id per evaluated frame)."""
    idx = [t for t, g in enumerate(gt_sets) if g]
    keys = {}
    ids = [keys.setdefault(frozenset(gt_sets[t]), len(keys)) for t in idx]
    return np.array(idx, int), np.array(ids, int)


def boundaries(labels):
    return [t for t in range(1, len(labels)) if labels[t] != labels[t - 1]]


def boundary_counts(pred_b, gt_b, tol):
    """Greedy one-to-one matching within `tol` frames: (hits, n_pred, n_gt)."""
    used, hits = set(), 0
    for b in gt_b:
        c = [p for p in pred_b if abs(p - b) <= tol and p not in used]
        if c:
            used.add(min(c, key=lambda p: abs(p - b)))
            hits += 1
    return hits, len(pred_b), len(gt_b)


def discovery_scores(labels, gt_sets, tol=5):
    """Clustering agreement and change timing of discovered states vs GT state sets, on evaluated frames."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    idx, g = gt_segments(gt_sets)
    if not len(idx):
        return None
    p = labels[idx]
    M = np.zeros((p.max() + 1, g.max() + 1))
    np.add.at(M, (p, g), 1)
    r, c = linear_sum_assignment(-M)
    # Boundaries in evaluated-frame positions, so gaps without GT don't count as changes.
    hits, npb, ngb = boundary_counts(boundaries(p), boundaries(g), tol)
    return {"mof": M[r, c].sum() / len(idx), "nmi": normalized_mutual_info_score(g, p),
            "ari": adjusted_rand_score(g, p), "k_pred": len(np.unique(p)), "k_gt": g.max() + 1,
            "b_hits": hits, "b_pred": npb, "b_gt": ngb}
