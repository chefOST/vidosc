"""Paired video-bootstrap CIs on discovery and ranking metrics between two discovery runs (run.py results with
'discovery_pairs').

    python experiments/statespace/compare_disc.py ss-ci-test-siglip-axes ss-ci-test-siglip-base
"""
import json
import sys
from collections import defaultdict

import numpy as np

from corpus import ROOT


def load(name):
    r = json.load(open(f"{ROOT}/runs/{name}/results.json"))
    pv = defaultdict(lambda: defaultdict(float))
    for p in r["discovery_pairs"]:
        if p["method"] != "discover":
            continue
        v = pv[p["video"]]
        v["pairs"] += 1
        v["mof"] += p["mof"]
        v["changing"] += p["changing"]
        v["nmi"] += p["nmi"] * p["changing"]
        v["ari"] += p["ari"] * p["changing"]
        v["k"] += p["k_pred"]
        for k in ("b_hits", "b_pred", "b_gt"):
            v[k] += p[k]
    for vid, ms in r["per_video"].items():
        for meth, c in ms.items():
            pv[vid][f"top1:{meth}"] += c["top1"]
            pv[vid][f"rr:{meth}"] += c["rr"]
            pv[vid][f"frames:{meth}"] += c["evaluated_frames"]
    return pv


def metrics(pv, vids):
    s = lambda k: sum(pv[v][k] for v in vids)
    p, r = s("b_hits") / max(s("b_pred"), 1), s("b_hits") / max(s("b_gt"), 1)
    out = {"mof": s("mof") / s("pairs"), "nmi": s("nmi") / s("changing"), "ari": s("ari") / s("changing"),
           "chgF1": 2 * p * r / max(p + r, 1e-9), "K": s("k") / s("pairs")}
    meths = sorted({k.split(":", 1)[1] for v in vids for k in pv[v] if k.startswith("frames:")})
    for m in meths:
        n = s(f"frames:{m}")
        if n:
            out[f"top1 {m}"], out[f"mrr {m}"] = s(f"top1:{m}") / n, s(f"rr:{m}") / n
    return out


def main(a, b, n=2000):
    A, B = load(a), load(b)
    vids = sorted(set(A) & set(B))
    ma, mb = metrics(A, vids), metrics(B, vids)
    rng = np.random.default_rng(0)
    diffs = defaultdict(list)
    for _ in range(n):
        s = rng.choice(vids, len(vids))
        x, y = metrics(A, s), metrics(B, s)
        for k in x:
            diffs[k].append(x[k] - y[k])
    print(f"{a} vs {b} ({len(vids)} videos)")
    for k in [k for k in ma if k in mb]:
        lo, hi = np.quantile(diffs[k], [.025, .975])
        print(f"  {k:6s} {ma[k]:.3f} vs {mb[k]:.3f}  diff {ma[k] - mb[k]:+.3f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
