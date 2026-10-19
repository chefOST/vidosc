"""Paired bootstrap over videos between (run, method) results of discovery/run.py (frame-pooled ratios).

    python -m pipeline.discovery.compare disc/test-crops-final:fused disc/test-siglip-final:frame --key top1
log_summary() logs the final dev/test tables to W&B.
"""
import argparse
import json

import numpy as np

from .. import ROOT



def load(spec):
    run, m = spec.rsplit(":", 1)
    pv = json.load(open(f"{ROOT}/runs/{run}/results.json"))["per_video"]
    return {v: c[m] for v, c in pv.items()}


def diff_ci(a, b, key="top1", n=5000, seed=0):
    vids = sorted(set(a) & set(b))
    rate = lambda d, s: sum(d[v][key] for v in s) / sum(d[v]["evaluated_frames"] for v in s)
    rng = np.random.default_rng(seed)
    boots = [rate(a, s) - rate(b, s) for s in (rng.choice(vids, len(vids)) for _ in range(n))]
    return rate(a, vids) - rate(b, vids), np.quantile(boots, 0.025), np.quantile(boots, 0.975), len(vids)


SUMMARY_RUNS = ["internvideo-final", "siglip-final", "crops-final", "siglip-vlm-final"]
SUMMARY_DIFFS = [("siglip-vlm-final:fused", "siglip-final:frame"), ("siglip-vlm-final:fused", "siglip-vlm-final:discover"),
                 ("siglip-vlm-final:fused", "siglip-vlm-final:vlm_discover"),
                 ("siglip-vlm-final:vlm_discover", "siglip-vlm-final:vlm_uniform"),
                 ("crops-final:vlm_discover", "crops-final:vlm_uniform"), ("siglip-final:discover", "siglip-final:frame"),
                 ("crops-final:discover", "crops-final:frame"), ("siglip-final:smooth", "siglip-final:frame")]


def log_summary(name="ryan/discover-summary"):
    """W&B milestone: ranking and discovery tables for dev/test, plus paired bootstrap differences."""
    import wandb
    dest = json.load(open(f"{ROOT}/weights_and_biases/settings.json"))
    run = wandb.init(entity=dest["entity"], project=dest["projects"]["experiments"], name=name, job_type="summary",
                     tags=["embedding-output", "state-discovery", "zero-shot", "proposal"],
                     notes="State discovery from embeddings: final dev/test tables (runs/disc/*-final). "
                           "Config chosen on dev: penalty 500, merge 0.05, debias 1. MOSCATO used only to score.")
    rank, disc, diffs = [], [], []
    for sp in ["dev", "test"]:
        for r in SUMMARY_RUNS:
            res = json.load(open(f"{ROOT}/runs/disc/{sp}-{r}/results.json"))
            rank += [[sp, r, m, s["top1"], s["mrr"], s["p@2"], s["r@2"], s["chance"], s["chance_rr"],
                      s["evaluated_frames"]] for m, s in res["ranking"].items()]
            disc += [[sp, r, m, d["mof"], d["nmi"], d["ari"], d["b_f1"], d["b_prec"], d["b_rec"], d["k_pred"], d["k_gt"],
                      d["pairs"]] for m, d in res["discovery"].items()]
        for a, b in SUMMARY_DIFFS:
            for k in ["top1", "rr"]:
                d, lo, hi, n = diff_ci(load(f"disc/{sp}-{a}"), load(f"disc/{sp}-{b}"), k)
                diffs.append([sp, a, b, k, d, lo, hi, n])
                if sp == "test" and a == "siglip-vlm-final:fused" and b == "siglip-final:frame":
                    run.summary[f"test/fused_minus_frame/{k}"] = d
    best = [x for x in rank if x[0] == "test" and x[1] == "siglip-vlm-final" and x[2] == "fused"][0]
    run.summary.update({"test/fused/top1": best[3], "test/fused/mrr": best[4], "test/chance/top1": best[7],
                        "test/chance/mrr": best[8]})
    run.log({"ranking": wandb.Table(columns=["split", "features", "method", "top1", "mrr", "p@2", "r@2", "chance_top1",
                                             "chance_mrr", "frames"], data=rank),
             "discovery": wandb.Table(columns=["split", "features", "method", "mof", "nmi", "ari", "boundary_f1",
                                               "boundary_p", "boundary_r", "k_pred", "k_gt", "tracks"], data=disc),
             "paired_bootstrap": wandb.Table(columns=["split", "a", "b", "metric", "a_minus_b", "ci_lo", "ci_hi", "videos"],
                                             data=diffs)})
    print(f"W&B: {run.url}")
    run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--key", default="top1")
    x = p.parse_args()
    d, lo, hi, n = diff_ci(load(x.a), load(x.b), x.key)
    print(f"{x.a} - {x.b} [{x.key}]: {d:+.3f} (95% CI {lo:+.3f}, {hi:+.3f}; {n} videos)")
