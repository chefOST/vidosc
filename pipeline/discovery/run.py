"""State discovery from embeddings on MOSCATO CMU (proposal, not the README metric).

Per (video, object): an embedding track (frame or object-crop features) -> discovered states (pipeline/discovery/discover.py)
-> each frame outputs its state's centroid. MOSCATO is used only to score:
  ranking: the object's candidate states, embedded with the same text encoder, ranked by cosine (pipeline/embedding/embed.py);
  discovery: discovered state ids vs GT state sets (MoF, NMI, ARI, change-point F1 at +-tol s, number of states).

Splits: dev = the first two GT videos of each recipe (hyperparameters), test = the other 20 (reported).

    python -m pipeline.discovery.run --feats crops-siglip --split test
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from .. import ROOT, data, scoring
from ..embedding import embed
from . import discover, objects

FEATS = {  # name: (feature dir, per-object crops?, text encoder spec)
    "internvideo": ("internvideo-mm-l14", False, "internvideo"),
    "siglip": ("clip-ViT-SO400M-14-SigLIP-384-webli", False, "clip:ViT-SO400M-14-SigLIP-384/webli"),
    "clip": ("clip-ViT-L-14-openai", False, "clip:ViT-L-14/openai"),
    "crops-siglip": ("crops-clip-ViT-SO400M-14-SigLIP-384-webli", True, "clip:ViT-SO400M-14-SigLIP-384/webli"),
    "crops-clip": ("crops-clip-ViT-L-14-openai", True, "clip:ViT-L-14/openai"),
}
METHODS = ["frame", "smooth", "const", "uniform", "discover", "vlm_uniform", "vlm_discover", "fused"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=os.path.join(ROOT, "..", "chefOST-data", "MOSCATO"))
    p.add_argument("--feats", default="siglip", choices=list(FEATS))
    p.add_argument("--mix", type=float, default=0.0, help="crop feats only: weight of the full-frame embedding mixed in")
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--state_source", default="pseudo_train", choices=["pseudo_train", "pseudo_recipe"])
    p.add_argument("--templates", default="basic", choices=list(scoring.TEMPLATES))
    p.add_argument("--vis", default="score", choices=["none", "score"], help="visibility weights from detector scores")
    p.add_argument("--debias", type=float, default=0.0, help="subtract this times the mean image embedding (modality gap)")
    p.add_argument("--sigma", type=float, default=3.0, help="smoothing before discovery (s)")
    p.add_argument("--penalty", type=float, default=20.0)
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--merge", type=float, default=0.05)
    p.add_argument("--tol", type=int, default=5, help="change-point tolerance (s)")
    p.add_argument("--vlm_per_state", type=int, default=2, help="VLM keyframes per discovered state")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--name", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def split_videos(ds, split):
    by = defaultdict(list)
    for v in ds.videos:
        by[data.recipe(v)].append(v)
    dev = {v for vs in by.values() for v in sorted(vs)[:2]}
    return sorted(dev) if split == "dev" else sorted(set(ds.videos) - dev) if split == "test" else ds.videos


class TextBank:
    """Cached unit text embeddings per (encoder, prompt); loads the encoder only on a cache miss."""

    def __init__(self, spec):
        self.spec, self.enc = spec, None
        self.path = f"{ROOT}/data/cache/text/{spec.replace('/', '_').replace(':', '_')}.npz"
        self.cache = dict(np.load(self.path)) if os.path.exists(self.path) else {}

    def __call__(self, obj, states, templates):
        prompts = [t.format(object=obj, state=s) for s in states for t in scoring.TEMPLATES[templates]]
        miss = [p for p in prompts if p not in self.cache]
        if miss:
            if self.enc is None:
                from .. import encoders
                self.enc = encoders.build_encoder(self.spec)
            for p, e in zip(miss, self.enc.encode_texts(miss)):
                self.cache[p] = e
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            np.savez(self.path, **self.cache)
        n = len(scoring.TEMPLATES[templates])
        E = np.stack([self.cache[p] for p in prompts]).reshape(len(states), n, -1).mean(1)
        return E / np.linalg.norm(E, axis=1, keepdims=True)


def vis_weights(s):
    lo, hi = np.quantile(s, 0.2), np.quantile(s, 0.9)
    return np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)


def tracks(a, ds, vids):
    """Yields (vid, obj, X [T, D] unit, w [T], gt sets) for every GT (video, object)."""
    fdir, crops, _ = FEATS[a.feats]
    for vid in vids:
        n = len(range(0, ds.num_frames(vid), 30))
        idx = list(range(0, ds.num_frames(vid), 30))
        F = np.load(f"{ROOT}/data/cache/emb/{fdir}/{vid}_fps1.npy").astype(np.float32)[:n]
        if crops:
            objs, boxes = objects.load_boxes(vid)
            boxes = boxes[:n]
            if a.mix:
                full = np.load(f"{ROOT}/data/cache/emb/{FEATS[a.feats.removeprefix('crops-')][0]}/{vid}_fps1.npy")
                full = discover.unit(full.astype(np.float32)[:n])
        for obj in ds.objects(vid):
            if crops:
                j = objs.index(obj)
                X, s = F[:, j], boxes[:, j, 4]
                if a.mix:
                    X = discover.unit(X) * (1 - a.mix) + a.mix * full
                w = vis_weights(s) if a.vis == "score" else np.ones(n)
            else:
                X, w = F, np.ones(n)
            yield vid, obj, discover.unit(X), w, ds.gt_states(vid, obj, idx)


def keyframes(labels, S, cents, w, m):
    """Per discovered state, up to m keyframes: the state's frames split into m time chunks, in each the frame
    maximizing visibility + cosine to the state centroid."""
    out = []
    for s in range(labels.max() + 1):
        ts = np.flatnonzero(labels == s)
        for chunk in np.array_split(ts, min(m, len(ts))):
            out.append(int(chunk[np.argmax(w[chunk] + S[chunk] @ cents[s])]))
    return sorted(out)


def outputs(a, X, w, vlm=None):
    """{method: [T, D] per-frame outputs}, plus discovered labels. vlm(frames) -> [n, D] unit text embeddings of
    the VLM's state descriptions at those frames."""
    out, T = {}, len(X)
    S = discover.vis_smooth(X, w, a.sigma)
    labels, cents = discover.discover(S, w, a.penalty, a.min_len, a.merge)
    wm = np.maximum(w, 1e-3)
    if "frame" in a.methods:
        out["frame"] = X
    if "smooth" in a.methods:
        out["smooth"] = S
    if "const" in a.methods:
        out["const"] = np.repeat(discover.unit((wm[:, None] * X).sum(0))[None], T, 0)
    if "uniform" in a.methods:  # as many segments as discovery found, equal length: timing without video
        k = len(discover.boundaries(labels)) + 1
        ul = np.minimum(np.arange(T) * k // T, k - 1)
        uc = discover.unit(np.stack([(wm[ul == s, None] * S[ul == s]).sum(0) for s in range(k)]))
        out["uniform"] = uc[ul]
        out["_uniform_labels"] = ul
    if "discover" in a.methods:
        out["discover"] = cents[labels]
    if vlm and ("vlm_discover" in a.methods or "vlm_uniform" in a.methods):
        kf = keyframes(labels, S, cents, w, a.vlm_per_state)
        D = vlm(kf)
        state_txt = discover.unit(np.stack([D[[labels[t] == s for t in kf]].sum(0) for s in range(labels.max() + 1)]))
        out["vlm_discover"] = state_txt[labels]
        # Same number of VLM calls at evenly spaced frames; every frame takes the nearest keyframe's description.
        uk = np.linspace(0, T - 1, 2 * len(kf) + 1)[1::2].round().astype(int)
        UD = vlm(list(uk))
        out["vlm_uniform"] = UD[np.abs(np.arange(T)[:, None] - uk[None]).argmin(1)]
    return out, labels


def bootstrap(per_video, methods, key, base, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vids = list(per_video)
    out = {m: [] for m in methods}
    for _ in range(n):
        s = rng.choice(vids, len(vids))
        tot = lambda m, k: sum(per_video[v][m][k] for v in s)
        for m in methods:
            out[m].append(tot(m, key) / tot(m, "evaluated_frames") - tot(base, key) / tot(base, "evaluated_frames"))
    return {m: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] for m, x in out.items()}


def main():
    a = parse_args()
    ds = data.Moscato(a.data_root, "CMU", cache_dir=os.path.join(ROOT, "data", "cache"))
    vids = split_videos(ds, a.split)
    cands = ds.candidate_states(a.state_source)
    bank_of = TextBank(FEATS[a.feats][2])
    all_tracks = list(tracks(a, ds, vids))
    mu = None
    if a.debias:  # mean image embedding over the split's unlabeled tracks
        mu = discover.unit(np.concatenate([X for _, _, X, _, _ in all_tracks]).mean(0))
    vlm = None
    if {"vlm_uniform", "vlm_discover"} & set(a.methods):
        # Dry pass collects the keyframes discovery picks; Modal describes the missing ones; then the real pass.
        from . import describe, modal_jobs
        keys = set()
        with np.errstate(all="ignore"):
            for vid, obj, X, w, gt in all_tracks:
                if ds.candidates_for(cands, a.state_source, vid, obj):
                    outputs(a, X, w, lambda fr: (keys.update(describe.key(vid, t, obj) for t in fr),
                                                 np.zeros((len(fr), 1)))[1])
        desc = modal_jobs.describe(keys)
        vlm = lambda fr: bank_of(obj, [desc[describe.key(vid, t, obj)] for t in fr], a.templates)
    per_video = defaultdict(lambda: defaultdict(Counter))
    disc = defaultdict(Counter)
    examples = []
    for vid, obj, X, w, gt in all_tracks:
        states = ds.candidates_for(cands, a.state_source, vid, obj)
        if not states:
            continue
        bank = bank_of(obj, states, a.templates)
        outs, labels = outputs(a, X, w, vlm)
        if "vlm_discover" in outs and "discover" in outs:  # visual state centroid + its VLM description
            vis = discover.unit(outs["discover"] - (a.debias * mu if mu is not None else 0))
            outs["fused"] = vis + outs["vlm_discover"]
        for m, o in outs.items():
            if m.startswith("_"):
                continue
            if mu is not None and m not in ("vlm_uniform", "vlm_discover", "fused"):
                o = o - a.debias * mu
            o = discover.unit(o)
            per_video[vid][m].update(embed.score(o, bank, states, gt))
        for m, lab in [("discover", labels), ("uniform", outs.get("_uniform_labels")),
                       ("one state", np.zeros(len(labels), int))]:
            if lab is None:
                continue
            d = discover.discovery_scores(lab, gt, a.tol)
            if d:
                ch = d["k_gt"] > 1  # NMI/ARI are degenerate (1.0) when GT has one state set: changing tracks only
                disc[m].update({"mof": d["mof"], "nmi": d["nmi"] * ch, "ari": d["ari"] * ch, "k_pred": d["k_pred"],
                                "k_gt": d["k_gt"], "b_hits": d["b_hits"], "b_pred": d["b_pred"], "b_gt": d["b_gt"]},
                               pairs=1, changing=ch)
                disc[m]["k_abs_err"] += abs(d["k_pred"] - d["k_gt"])
        if len(examples) < 60:
            o = discover.unit(outs["discover"] - (a.debias * mu if mu is not None else 0))
            names = [states[int(k)] for k in (o @ bank.T).argmax(1)]
            spans = [(t0, t1, names[t0]) for t0, t1 in zip([0] + discover.boundaries(labels),
                                                          discover.boundaries(labels) + [len(labels)])]
            examples.append({"video": vid, "object": obj, "states": len(set(labels)), "spans": spans,
                             "gt": [sorted(g) for g in gt[::10]]})
    methods = [m for m in METHODS if m in a.methods]
    totals = {m: sum((per_video[v][m] for v in per_video), Counter()) for m in methods}
    summary = {m: embed.summarize(c) for m, c in totals.items()}
    base = "frame" if "frame" in methods else methods[0]
    cis = {k: bootstrap(per_video, methods, k, base) for k in ("top1", "rr")}
    dsum = {}
    for m, c in disc.items():
        n = c["pairs"]
        p = c["b_hits"] / max(c["b_pred"], 1)
        r = c["b_hits"] / max(c["b_gt"], 1)
        dsum[m] = {"mof": c["mof"] / n, "nmi": c["nmi"] / c["changing"], "ari": c["ari"] / c["changing"], "k_pred": c["k_pred"] / n,
                   "k_gt": c["k_gt"] / n, "k_abs_err": c["k_abs_err"] / n, "b_prec": p, "b_rec": r,
                   "b_f1": 2 * p * r / max(p + r, 1e-9), "pairs": n,
                   "changing_pairs": c["changing"]}
    name = a.name or f"disc-{a.feats}-{a.split}-{time.strftime('%m%d-%H%M%S')}"
    out_dir = os.path.join(ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)
    res = {"args": vars(a), "videos": vids, "ranking": summary, "ci_vs_" + base: cis, "discovery": dsum,
           "per_video": {v: {m: dict(c) for m, c in ms.items()} for v, ms in per_video.items()}, "examples": examples}
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=float)
    if not a.quiet:
        print(f"{a.feats} split={a.split} ({len(vids)} videos, {sum(1 for _ in per_video)} with scores)")
        for m, s in summary.items():
            lo, hi = cis["top1"][m]
            print(f"  {m:9s} top1 {s['top1']:.3f} [{lo:+.3f},{hi:+.3f}]  mrr {s['mrr']:.3f}  p@2 {s['p@2']:.3f}  "
                  f"r@2 {s['r@2']:.3f}   chance top1 {s['chance']:.3f} mrr {s['chance_rr']:.3f}  ({s['evaluated_frames']} fr)")
        for m, d in dsum.items():
            print(f"  [{m}] MoF {d['mof']:.3f} NMI {d['nmi']:.3f} ARI {d['ari']:.3f} bF1 {d['b_f1']:.3f} "
                  f"(P {d['b_prec']:.2f} R {d['b_rec']:.2f})  K {d['k_pred']:.2f} vs GT {d['k_gt']:.2f}")
    if a.wandb:
        log_wandb(a, name, res)
    return res


def log_wandb(a, name, res):
    import wandb
    dest = json.load(open(os.path.join(ROOT, "weights_and_biases", "settings.json")))
    run = wandb.init(entity=dest["entity"], project=dest["projects"]["experiments"], name=name, job_type="discover",
                     tags=["embedding-output", "state-discovery", "zero-shot", "proposal"],
                     notes="State discovery from embeddings. Zero-shot: MOSCATO used only to score. " +
                           "; ".join(f"{m}: top1 {s['top1']:.3f} MRR {s['mrr']:.3f}" for m, s in res["ranking"].items()),
                     config={**vars(a), "videos": res["videos"], "moscato_use": "GT scoring only"})
    for m, s in res["ranking"].items():
        run.summary.update({f"rank/{m}/{k}": v for k, v in s.items()})
    for m, d in res["discovery"].items():
        run.summary.update({f"disc/{m}/{k}": v for k, v in d.items()})
    rows = [[m, s["top1"], s["chance"], s["mrr"], s["chance_rr"], s["p@2"], s["r@2"], *res["ci_vs_frame"]["top1"].get(m, [0, 0])]
            for m, s in res["ranking"].items()] if "ci_vs_frame" in res else []
    run.log({"ranking": wandb.Table(columns=["method", "top1", "chance_top1", "mrr", "chance_mrr", "p@2", "r@2",
                                             "top1_diff_ci_lo", "top1_diff_ci_hi"], data=rows),
             "discovery": wandb.Table(columns=["method", *next(iter(res["discovery"].values())).keys()],
                                      data=[[m, *d.values()] for m, d in res["discovery"].items()]),
             "examples": wandb.Table(columns=["video", "object", "states", "spans"],
                                     data=[[e["video"], e["object"], e["states"], json.dumps(e["spans"])]
                                           for e in res["examples"]])})
    print(f"W&B: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
