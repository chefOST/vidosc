"""E2: state sensitivity of a space on MOSCATO tracks (GT used only to score), with no discovery penalty to tune.

Per (video, object) track (whole frames or OWLv2 crops), mapped into the space:
  change_auc  windowed change score d_t = 1 - cos(mean x[t-w:t], mean x[t:t+w]); AUC of d_t at GT state-set
              changes (within +-2 s) vs frames >= 10 s from any change. Tracks with >= 1 change.
  sep         mean cos to own GT segment centroid minus mean cos to other segments' centroids (same track).
  frame_top1  per-frame ranking of the object's candidate states (pipeline/embedding/embed.py), and
  seg_top1    the same with each frame replaced by its GT segment's centroid (naming potential of a state
              centroid, independent of discovery).
Bootstrap CIs resample videos, paired against a base space.

    python experiments/statespace/eval_tracks.py --spaces identity ss-v1 --feats siglip crops-siglip --split test
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

import corpus
from corpus import ROOT
from pipeline import data
from pipeline.discovery import discover, run
from pipeline.embedding import embed


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def segments(gt):
    """GT segment id per frame (runs of identical state sets; frames without GT keep -1)."""
    ids, cur, last = np.full(len(gt), -1), -1, None
    for t, g in enumerate(gt):
        if not g:
            continue
        key = frozenset(g)
        if key != last:
            cur, last = cur + 1, key
        ids[t] = cur
    return ids


def track_metrics(X, w, gt, bank, states, win=5):
    T = len(X)
    out = {}
    S = discover.unit(discover.vis_smooth(X, w, 1.0))
    seg = segments(gt)
    ch = [t for t in range(1, T) if seg[t] >= 0 and seg[t - 1] >= 0 and seg[t] != seg[t - 1]]
    if ch and T > 4 * win:
        cs = np.cumsum(np.vstack([np.zeros(X.shape[1]), S]), 0)
        d = np.full(T, np.nan)
        for t in range(win, T - win):
            a, b = cs[t] - cs[t - win], cs[t + win] - cs[t]
            d[t] = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        dist = np.min(np.abs(np.arange(T)[:, None] - np.array(ch)[None]), 1)
        ok = ~np.isnan(d)
        out["change_auc"] = auc(d[ok & (dist <= 2)], d[ok & (dist >= 10)])
    ks = [k for k in range(seg.max() + 1) if (seg == k).sum()]
    if len(ks) > 1:
        C = discover.unit(np.stack([X[seg == k].mean(0) for k in ks]))
        own = np.array([X[t] @ C[ks.index(seg[t])] for t in range(T) if seg[t] >= 0])
        oth = np.array([np.mean([X[t] @ C[j] for j in range(len(ks)) if ks[j] != seg[t]]) for t in range(T) if seg[t] >= 0])
        out["sep"] = float((own - oth).mean())
    out["frame"] = embed.score(X, bank, states, gt)
    Cs = np.zeros_like(X)
    for k in ks:
        Cs[seg == k] = discover.unit(X[seg == k].mean(0))
    Cs[seg < 0] = X[seg < 0]
    out["seg"] = embed.score(Cs, bank, states, gt)
    return out


def load(name):
    """run.load_space names, plus 'axes-proj' and 'axes-mix<lam>' (axes.py)."""
    if name.startswith("axes"):
        import axes
        from space import text_mean
        return axes.AxisSpace(text_mean(), mode="proj" if name == "axes-proj" else "mix",
                              lam=float(name.removeprefix("axes-mix") or 1) if name != "axes-proj" else 1)
    return run.load_space(name)


def evaluate(spaces, feats="siglip", dataset="CMU", split="test", base="identity", n_boot=1000, log=print):
    ds = data.Moscato(f"{ROOT}/../chefOST-data/MOSCATO", dataset, cache_dir=f"{ROOT}/data/cache")
    vids = run.split_videos(ds, split)
    a = argparse.Namespace(feats=feats, mix=0.0, vis="score")
    cands = ds.candidate_states("pseudo_train")
    tb = corpus.bank() if run.FEATS[feats][2] == corpus.SPEC else run.TextBank(run.FEATS[feats][2])
    tracks = list(run.tracks(a, ds, vids))
    mu = discover.unit(np.concatenate([X for _, _, X, _, _ in tracks]).mean(0))
    per = {n: defaultdict(lambda: defaultdict(list)) for n in spaces}
    for name, sp in spaces.items():
        for vid, obj, X, w, gt in tracks:
            states = ds.candidates_for(cands, "pseudo_train", vid, obj)
            if not states:
                continue
            if hasattr(sp, "axes"):  # per-object space (axes.py)
                m = track_metrics(sp.image(X, obj, mu), w, gt, sp.text(tb(obj, states, "basic"), obj), states)
            else:
                m = track_metrics(sp.image(X, mu), w, gt, sp.text(tb(obj, states, "basic")), states)
            for k, v in m.items():
                if v is not None:
                    per[name][vid][k].append(v)
    res = {}
    vl = sorted(per[base])
    rng = np.random.default_rng(0)
    boots = [rng.choice(vl, len(vl)) for _ in range(n_boot)]

    def stat(name, k, vs):
        if k in ("frame", "seg"):
            c = [sum((x for v in vs for x in per[name][v][k]), start=type(per[name][vl[0]][k][0])())]
            c = c[0]
            return c["top1"] / c["evaluated_frames"], c["rr"] / c["evaluated_frames"]
        x = [y for v in vs for y in per[name][v][k]]
        return (float(np.mean(x)), None) if x else (np.nan, None)

    for name in spaces:
        r = {}
        for k in ("change_auc", "sep", "frame", "seg"):
            val = stat(name, k, vl)
            d = [stat(name, k, b)[0] - stat(base, k, b)[0] for b in boots] if name != base else [0]
            r[k] = (val[0], val[1], float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975)))
        res[name] = r
        log(f"  {name:22s} " + "  ".join(
            f"{k} {v[0]:.3f}" + (f"/{v[1]:.3f}" if v[1] is not None else "") + f" [{v[2]:+.3f},{v[3]:+.3f}]"
            for k, v in r.items()))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spaces", nargs="+", default=["identity"])
    p.add_argument("--feats", nargs="+", default=["siglip"])
    p.add_argument("--dataset", default="CMU")
    p.add_argument("--split", default="test")
    a = p.parse_args()
    sp = {n: load(n) for n in a.spaces}
    for f in a.feats:
        print(f"{a.dataset} {a.split} {f}  (top1/MRR for frame, seg)")
        evaluate(sp, f, a.dataset, a.split, base=a.spaces[0])
