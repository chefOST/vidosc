"""Image-only state metric from EPIC action structure (no text, no LLM, no MOSCATO).

For a noun in a video, between the end of action i and the start of the next action on that noun, nothing acts on
it, so its state is constant even though the view changes. Across a state-changing action (STATE_VERBS) its
state changes while the scene stays. MATCHA-style triplets in image space:
    anchor = frame in the untouched gap after action i,  positive = another frame in that gap (other view, same
    state),  negative = a frame from before action i (same object and scene, other state).
Loss: cosine margin, plus ||delta||^2 to stay near SigLIP. Feats: whole frames ('frame') or noun crops ('crop',
from modal_crops.py; only the frames around actions are cropped, so the gap frames are its start/end).
Selection on held-out EPIC participants.

    python experiments/statespace/train_gap.py --name gap-frame
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import corpus
from corpus import ROOT
from pipeline.discovery import finetune
from space import Space, unit
from train import OUT, Head, val_participants

CROPS = f"{ROOT}/data/cache/statespace/epic_crops"


def sec(ts):
    h, m, s = ts.split(":")
    return int(int(h) * 3600 + int(m) * 60 + float(s))


def triplet_sources(min_gap=2, max_gap=120, all_verbs=False):
    """[(participant, vid, noun, before frames, gap frames)] per state-changing action followed by an untouched gap."""
    cls = {r["id"]: r["key"] for r in csv.DictReader(open(f"{corpus.epic.DIR}/EPIC_100_verb_classes.csv"))}
    have = {f[:-9] for f in os.listdir(f"{ROOT}/data/cache/emb/{corpus.FDIR}") if f.startswith("P") and "_" in f[:4]}
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{corpus.epic.DIR}/EPIC_100_train.csv")):
        if r["video_id"] in have:
            by[(r["video_id"], corpus.epic.noun(r["noun"]))].append(
                (sec(r["start_timestamp"]), sec(r["stop_timestamp"]), cls[r["verb_class"]]))
    out = []
    for (v, n), acts in sorted(by.items()):
        acts.sort()
        for k, (s0, e0, vc) in enumerate(acts):
            if not (all_verbs or vc in finetune.STATE_VERBS) or e0 <= s0:
                continue
            prev_end = acts[k - 1][1] if k else -10 ** 9
            nxt = acts[k + 1][0] if k + 1 < len(acts) else None
            if nxt is None or not (min_gap <= nxt - e0 <= max_gap):
                continue
            before = [t for t in (s0 - 1, s0) if t > prev_end]
            gap = list(range(e0, nxt + 1))
            if before and len(gap) >= 2:
                out.append((v.split("_")[0], v, n, before, gap))
    return out


def build(a):
    src = triplet_sources(all_verbs=a.all_verbs)
    emb = {}
    for v in sorted({s[1] for s in src}):
        emb[v] = unit(corpus.frame_emb(v))
    if a.feats == "crop":
        crops = {}
        for v in emb:
            if os.path.exists(f"{CROPS}/{v}.npz"):
                z = np.load(f"{CROPS}/{v}.npz")
                crops[v] = {(int(t), str(n)): unit(e.astype(np.float32)) for t, n, e in zip(z["t"], z["noun"], z["emb"])}

        def get(v, n, t):
            return crops.get(v, {}).get((t, n))
        # crops exist only at t0-1, t0, t1, t1+1 of actions: gap frames are its end and the next start
        src = [(p, v, n, b, [t for t in g if get(v, n, t) is not None]) for p, v, n, b, g in src]
        src = [(p, v, n, [t for t in b if get(v, n, t) is not None], g) for p, v, n, b, g in src]
        src = [s for s in src if s[3] and len(s[4]) >= 2]
    else:
        def get(v, n, t):
            return emb[v][t] if t < len(emb[v]) else None
        src = [(p, v, n, [t for t in b if t < len(emb[v])], [t for t in g if t < len(emb[v])]) for p, v, n, b, g in src]
        src = [s for s in src if s[3] and len(s[4]) >= 2]
    vp = val_participants({s[0] for s in src})
    mu = unit(np.mean([get(s[1], s[2], t) for s in src for t in s[3] + s[4]], 0))
    print(f"{len(src)} state-change sources ({sum(s[0] in vp for s in src)} val), {len(emb)} videos, feats {a.feats}")
    return src, get, vp, mu


def sample(src, get, mu, rng, n):
    A, P, N = [], [], []
    for i in rng.integers(0, len(src), n):
        _, v, nn, b, g = src[i]
        ta, tp = rng.choice(g, 2, replace=False)
        A.append(get(v, nn, int(ta)) - mu); P.append(get(v, nn, int(tp)) - mu); N.append(get(v, nn, int(rng.choice(b))) - mu)
    t = lambda x: torch.tensor(np.stack(x), dtype=torch.float32)
    return t(A), t(P), t(N)


def cos(x, y):
    return (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)


def fixed_val(src, get, mu, vp, seed=1, n=4000):
    val = [s for s in src if s[0] in vp]
    return sample(val, get, mu, np.random.default_rng(seed), n)


def train(a):
    src, get, vp, mu = build(a)
    tr = [s for s in src if s[0] not in vp]
    VA, VP, VN = fixed_val(src, get, mu, vp)
    torch.manual_seed(a.seed)
    m = Head(h=a.hidden, p=a.dropout)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)

    @torch.no_grad()
    def val():
        m.eval()
        x, p, n = m(VA), m(VP), m(VN)
        r = {"val_acc": (cos(x, p) > cos(x, n)).float().mean().item(), "val_margin": (cos(x, p) - cos(x, n)).mean().item()}
        m.train()
        return r

    print("init (SigLIP):", val())
    hist = []
    for step in range(a.steps):
        zs = sample(tr, get, mu, rng, a.bs)
        x, p, n = (m(z) for z in zs)
        loss = F.relu(a.margin + cos(x, n) - cos(x, p)).mean()
        if a.w_reg:
            loss = loss + a.w_reg * m.delta(zs[0]).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            v = val()
            hist.append({"step": step, "loss": float(loss.detach()), **v})
            print(f"step {step} loss {float(loss.detach()):.4f}  " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save({"state": m.state_dict(), "hidden": a.hidden, "mu_txt": mu, "mu_img": mu, "args": vars(a)}, f"{OUT}/{a.name}.pt")
    json.dump({"args": vars(a), "hist": hist}, open(f"{OUT}/{a.name}.json", "w"), indent=1)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gap-frame")
    p.add_argument("--feats", default="frame", choices=["frame", "crop"])
    p.add_argument("--all_verbs", action="store_true")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--w_reg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=250)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse())
