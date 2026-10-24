"""MATCHA-style trained heads: a shared residual head on centred SigLIP, trained so that state differences
dominate. No MOSCATO data. Losses (cosine margins, MATCHA eq. 7: max(0, m + sim_neg - sim_pos)):

  text   state -> its paraphrase  vs  an LLM counterfactual contradiction (gen_triplets.py), plus the other
         phrase of the same EPIC action (before vs after) as a negative.
  cross  EPIC frame at an action's end -> its 'after' phrase vs its 'before' phrase (and start -> before vs after).
  img    EPIC start/end frames of an action: sim(end, end +- 1 s) > sim(end, start) (same scene, other state).
  tcn    (optional) unlabeled video incl. non-GT CMU: sim(t, t +- 1..2 s) > sim(t, t +- 20..90 s).
  reg    keeps the head near identity: mean ||delta||^2.

Validation (model selection) on held-out EPIC participants (cross, img) and held-out nouns (text) only.

    python experiments/statespace/train.py --name ss-v1     # -> data/cache/statespace/<name>.pt + val json
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import corpus
from corpus import ROOT, bank, prompt
from space import Space, text_mean, unit

OUT = f"{ROOT}/data/cache/statespace"
TRIPLETS = f"{corpus.epic.DIR}/state_triplets.json"


class Head(torch.nn.Module):
    """x -> x + MLP(x) (last layer zero-init, so training starts at SigLIP); optional object conditioning for
    images is not used (frames are shared by all objects in the pipeline)."""

    def __init__(self, d=1152, h=1024, p=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(h, d))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def delta(self, x):
        return self.net(x)

    def forward(self, x):
        return x + self.net(x)


def held_out(s, frac=0.2):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < frac * 100


def text_data(no_loc=False):
    """Text triplets [(anchor, pos, neg)] as prompt strings, with a held-out-noun flag. no_loc: drop
    counterfactuals that only change location."""
    trip = json.load(open(TRIPLETS)) if os.path.exists(TRIPLETS) else {}
    rows = []
    for k, r in trip.items():
        n, s = k.split("|", 1)
        same = [p for p in r.get("same", []) if isinstance(p, str)]
        contra = [c[1] if isinstance(c, list) else c for c in r.get("contra", [])
                  if not (no_loc and isinstance(c, list) and c[0] == "location")]
        for p in same:
            for c in contra:
                if isinstance(c, str):
                    rows.append((prompt(n, s), prompt(n, p), prompt(n, c), held_out(n)))
    # EPIC before/after as an extra contradiction, positive = paraphrase when we have one
    para = {k: [p for p in r.get("same", []) if isinstance(p, str)] for k, r in trip.items()}
    for _, n, b, a in corpus.text():
        if b and a and b.lower() != a.lower():
            for s, o in ((b, a), (a, b)):
                for p in para.get(f"{n}|{s}", [])[:1]:
                    rows.append((prompt(n, s), prompt(n, p), prompt(n, o), held_out(n)))
    return rows


def frame_data(state_only=False):
    """[(participant, vid, t_start, t_end, before prompt, after prompt, noun)] with before != after."""
    from pipeline.discovery import finetune
    import csv
    cls = {r["id"]: r["key"] for r in csv.DictReader(open(f"{corpus.epic.DIR}/EPIC_100_verb_classes.csv"))}
    verb_cls = {}
    for r in csv.DictReader(open(f"{corpus.epic.DIR}/EPIC_100_train.csv")):
        verb_cls[r["verb"]] = cls[r["verb_class"]]
    out = []
    for p, v, t0, t1, verb, n, b, a in corpus.frames():
        if not b or not a or b.lower() == a.lower() or t1 <= t0:
            continue
        if state_only and verb_cls.get(verb) not in finetune.STATE_VERBS:
            continue
        out.append((p, v, t0, t1, prompt(n, b), prompt(n, a), n))
    return out


def val_participants(parts, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return set(rng.choice(sorted(parts), max(1, int(len(parts) * frac)), replace=False))


def build(a):
    tb = bank()
    T = text_data(a.no_loc)
    Fr = frame_data(a.state_only)
    vids = sorted({r[1] for r in Fr})
    emb = {v: unit(corpus.frame_emb(v)) for v in vids}
    Fr = [r for r in Fr if r[3] + 1 < len(emb[r[1]])]
    mu_img = unit(np.concatenate(list(emb.values())).mean(0))
    mu_txt = text_mean()
    nouns = sorted({r[6] for r in Fr})
    strings = sorted({s for r in T for s in r[:3]} | {s for r in Fr for s in r[4:6]} | {f"a photo of {n}." for n in nouns})
    print(f"{len(T)} text triplets, {len(Fr)} EPIC actions with frames ({len(vids)} videos), {len(strings)} prompts")
    E = dict(zip(strings, tb.prompts(strings)))
    tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    TA, TP, TN = (tt([E[r[i]] - mu_txt for r in T]) for i in range(3))
    Tval = np.array([r[3] for r in T], bool)
    vp = val_participants({r[0] for r in Fr})
    Fval = np.array([r[0] in vp for r in Fr], bool)
    FS = tt([emb[r[1]][r[2]] - mu_img for r in Fr])
    FE = tt([emb[r[1]][r[3]] - mu_img for r in Fr])
    FEn = tt([emb[r[1]][min(r[3] + 1, len(emb[r[1]]) - 1)] - mu_img for r in Fr])  # end + 1 s
    FSn = tt([emb[r[1]][max(r[2] - 1, 0)] - mu_img for r in Fr])  # start - 1 s
    PB = tt([E[r[4]] - mu_txt for r in Fr])
    PA = tt([E[r[5]] - mu_txt for r in Fr])
    tcn = None
    if a.tcn:
        vids_u = [f[:-9] for f in os.listdir(f"{ROOT}/data/cache/emb/{corpus.FDIR}")
                  if (f.startswith("P") and "_" in f[:4] and f[:3] not in {p for p in vp}) or f[:-9] in set(open(f"{ROOT}/data/cmu_nongt.txt").read().split())]
        tcn = [tt(unit(corpus.frame_emb(v)) - mu_img) for v in vids_u]
        print(f"tcn on {len(tcn)} unlabeled videos")
    pool = tt([E[s] - mu_txt for s in strings])  # general vocabulary for the keep term
    NN = tt([E[f"a photo of {n}."] - mu_txt for n in nouns])
    noun_id = torch.tensor([nouns.index(r[6]) for r in Fr])
    return dict(TA=TA, TP=TP, TN=TN, Tval=Tval, FS=FS, FE=FE, FEn=FEn, FSn=FSn, PB=PB, PA=PA, Fval=Fval, tcn=tcn,
                mu_img=mu_img, mu_txt=mu_txt, pool=pool, NN=NN, noun_id=noun_id)


def cos(x, y):
    return (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)


@torch.no_grad()
def validate(m, D):
    m.eval()
    out = {}
    v = D["Tval"]
    if v.any():
        a, p, n = m(D["TA"][v]), m(D["TP"][v]), m(D["TN"][v])
        out["text_acc"] = (cos(a, p) > cos(a, n)).float().mean().item()
    v = D["Fval"]
    s, e, b, af = m(D["FS"][v]), m(D["FE"][v]), m(D["PB"][v]), m(D["PA"][v])
    out["cross_acc"] = 0.5 * ((cos(e, af) > cos(e, b)).float().mean().item() + (cos(s, b) > cos(s, af)).float().mean().item())
    en = m(D["FEn"][v])
    out["img_acc"] = (cos(e, en) > cos(e, s)).float().mean().item()
    ids = D["noun_id"][v]
    vn = torch.unique(ids)
    S = F.normalize(e, dim=-1) @ F.normalize(m(D["NN"][vn]), dim=-1).T
    out["keep_noun"] = (vn[S.argmax(1)] == ids).float().mean().item()
    out["score"] = np.mean([out.get("text_acc", 0.5), out["cross_acc"]])
    m.train()
    return out


def train(a):
    D = build(a)
    torch.manual_seed(a.seed)
    m = Head(h=a.hidden, p=a.dropout)
    print("val at init (SigLIP):", validate(m, D))
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    rng = np.random.default_rng(a.seed)
    ti, fi = np.flatnonzero(~D["Tval"]), np.flatnonzero(~D["Fval"])
    hinge = lambda pos, neg: F.relu(a.margin + neg - pos).mean()
    hist = []
    for step in range(a.steps):
        loss = 0.0
        if len(ti) and a.w_text:
            b = rng.choice(ti, a.bs)
            x, p, n = m(D["TA"][b]), m(D["TP"][b]), m(D["TN"][b])
            loss = loss + a.w_text * hinge(cos(x, p), cos(x, n))
        b = rng.choice(fi, min(a.bs, len(fi)))
        s, e, pb, pa = m(D["FS"][b]), m(D["FE"][b]), m(D["PB"][b]), m(D["PA"][b])
        if a.w_cross:
            loss = loss + a.w_cross * 0.5 * (hinge(cos(e, pa), cos(e, pb)) + hinge(cos(s, pb), cos(s, pa)))
        if a.w_nce:  # in-batch cross-modal InfoNCE; the action's other phrase is in the batch as a hard negative
            txt = F.normalize(torch.cat([pa, pb]), dim=-1)
            tgt = torch.arange(len(b))
            le = F.cross_entropy(F.normalize(e, dim=-1) @ txt.T / a.tau, tgt)
            ls = F.cross_entropy(F.normalize(s, dim=-1) @ txt.T / a.tau, tgt + len(b))
            loss = loss + a.w_nce * 0.5 * (le + ls)
        if a.w_keep:  # preserve SigLIP's image->text similarity distribution over a general vocabulary
            P = D["pool"][rng.choice(len(D["pool"]), a.keep_k)]
            img = torch.cat([D["FE"][b], D["FS"][b]])
            s0 = F.normalize(img, dim=-1) @ F.normalize(P, dim=-1).T
            s1 = F.normalize(m(img), dim=-1) @ F.normalize(m(P), dim=-1).T
            loss = loss + a.w_keep * F.kl_div(F.log_softmax(s1 / a.tau, -1), F.log_softmax(s0 / a.tau, -1),
                                              log_target=True, reduction="batchmean")
        if a.w_img:
            en, sn = m(D["FEn"][b]), m(D["FSn"][b])
            loss = loss + a.w_img * 0.5 * (hinge(cos(e, en), cos(e, s)) + hinge(cos(s, sn), cos(s, e)))
        if D["tcn"] is not None and a.w_tcn:
            xs, ps, ns = [], [], []
            for _ in range(a.bs):
                X = D["tcn"][rng.integers(len(D["tcn"]))]
                t = rng.integers(0, len(X))
                dp = rng.integers(1, 3) * rng.choice([-1, 1])
                dn = rng.integers(20, 91) * rng.choice([-1, 1])
                xs.append(X[t]); ps.append(X[np.clip(t + dp, 0, len(X) - 1)]); ns.append(X[np.clip(t + dn, 0, len(X) - 1)])
            x, p, n = m(torch.stack(xs)), m(torch.stack(ps)), m(torch.stack(ns))
            loss = loss + a.w_tcn * hinge(cos(x, p), cos(x, n))
        if a.w_reg:
            z = torch.cat([D["TA"][rng.choice(len(D["TA"]), a.bs)] if len(D["TA"]) else D["PB"][b], D["FE"][b]])
            loss = loss + a.w_reg * m.delta(z).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            v = validate(m, D)
            hist.append({"step": step, "loss": float(loss.detach()), **v})
            print(f"step {step} loss {float(loss):.4f}  " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save({"state": m.state_dict(), "hidden": a.hidden, "mu_txt": D["mu_txt"], "mu_img": D["mu_img"],
                "args": vars(a)}, f"{OUT}/{a.name}.pt")
    json.dump({"args": vars(a), "hist": hist}, open(f"{OUT}/{a.name}.json", "w"), indent=1)
    return m, hist


def load(name):
    ck = torch.load(f"{OUT}/{name}.pt", weights_only=False)
    m = Head(h=ck["hidden"])
    m.load_state_dict(ck["state"])
    m.eval()
    return Space(ck["mu_txt"], ck["mu_img"], net=m, name=name)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ss-v1")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--w_text", type=float, default=1.0)
    p.add_argument("--w_cross", type=float, default=1.0)
    p.add_argument("--w_img", type=float, default=0.5)
    p.add_argument("--w_tcn", type=float, default=0.0)
    p.add_argument("--tcn", action="store_true")
    p.add_argument("--w_reg", type=float, default=0.1)
    p.add_argument("--w_nce", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--w_keep", type=float, default=0.0)
    p.add_argument("--keep_k", type=int, default=512)
    p.add_argument("--no_loc", action="store_true")
    p.add_argument("--state_only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=250)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse())
