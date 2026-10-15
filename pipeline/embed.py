"""Embedding-output pipeline (proposal; README precision is unchanged).

Methods output one vector per frame and object in the encoder's text space instead of a state name.
Scoring embeds the object's candidate MOSCATO states with the same text encoder and ranks them by
cosine to the output. Ranking, not raw cosine: the object words shared by every prompt cancel out,
so only the difference between states counts.

  top1: the nearest candidate is a GT state of the frame (README precision, for the nearest state).
  mrr:  1 / rank of the best-ranked GT state (0 if no GT state is a candidate).
"""
import os
from collections import Counter

import numpy as np
import torch

from .vidosc import hard_pseudo_labels, load_vidosc, trunk


def smooth(e, window=1):
    """Moving average of [T, D] embeddings over `window` frames, renormalized (window 1 = unchanged)."""
    w = int(window)
    if w > 1:
        k = np.ones(w) / w
        e = np.stack([np.convolve(e[:, j], k, mode="same") for j in range(e.shape[1])], 1)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def score(out, bank, states, gt_sets):
    """out: [T, D] unit outputs; bank: [K, D] unit state embeddings; gt_sets: GT state set per frame.
    Counts over evaluated frames (>=1 GT state)."""
    sims = out @ bank.T
    order = np.argsort(-sims, 1)
    c = {"evaluated_frames": 0, "top1": 0, "rr": 0.0, "oracle_frames": 0, "chance": 0.0}
    for t, g in enumerate(gt_sets):
        if not g:
            continue
        hit = [r for r, k in enumerate(order[t]) if states[k] in g]
        c["evaluated_frames"] += 1
        c["top1"] += bool(hit) and hit[0] == 0
        c["rr"] += 1 / (hit[0] + 1) if hit else 0.0
        c["oracle_frames"] += bool(hit)
        c["chance"] += len(hit) / len(states)
    return c


def errors(out, bank, states, gt_sets):
    """Wrong frames as Counter((predicted, GT states joined), text cosine of predicted to its closest GT state)."""
    pred = (out @ bank.T).argmax(1)
    S = bank @ bank.T
    c = Counter()
    for t, g in enumerate(gt_sets):
        gi = [k for k, s in enumerate(states) if s in g]
        if g and pred[t] not in gi:
            near = round(float(max(S[pred[t], k] for k in gi)), 3) if gi else None
            c[(states[pred[t]], " | ".join(sorted(g)), near)] += 1
    return c


def summarize(c):
    n = c["evaluated_frames"]
    f = lambda k: c[k] / n if n else float("nan")
    return {"top1": f("top1"), "mrr": f("rr"), "oracle": f("oracle_frames"), "chance": f("chance"), "evaluated_frames": n}


def bank_spread(bank):
    """Mean pairwise cosine between an object's state embeddings (how bunched the bank is)."""
    k = len(bank)
    return float((bank @ bank.T)[~np.eye(k, dtype=bool)].mean()) if k > 1 else float("nan")


def adapt(videos, target="frame", bank=None, ckpt="multi_task", steps=300, lr=1e-4, wd=1e-4, keep=0.5,
          scale=100.0, seed=0, log=print):
    """VidOSC trunk + one linear projection into the text space, self-trained per video set with no
    MOSCATO labels or states. Outputs are per frame (the same for every object).
    videos: {vid: (x [T, 1536], frame embedding [T, D])}.
    target 'frame': regress the frozen zero-shot frame embedding (cosine), no vocabulary.
    target 'bank':  InfoNCE on confident zero-shot pseudo-labels over `bank` [M, D], an external state
                    vocabulary (e.g. ChangeIt's).
    Returns {vid: unit outputs [T, D]}."""
    torch.manual_seed(seed)
    model = load_vidosc(ckpt)
    dim = next(iter(videos.values()))[1].shape[1]
    proj = torch.nn.Linear(model.proj.out_features, dim)
    xs = {v: x for v, (x, _) in videos.items()}
    teach = {v: torch.as_tensor(e, dtype=torch.float32) for v, (_, e) in videos.items()}
    if target == "bank":
        b = torch.as_tensor(bank, dtype=torch.float32)
        teach = {v: hard_pseudo_labels(torch.softmax(scale * e @ b.T, -1).numpy(), keep) for v, e in teach.items()}
        ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss_fn = lambda e, y: ce(scale * e @ b.T, y)
    else:
        loss_fn = lambda e, y: (1 - (e * y).sum(-1)).mean()
    emb = lambda h: torch.nn.functional.normalize(proj(h), dim=-1)
    params = list(model.proj.parameters()) + list(model.ln.parameters()) + \
        list(model.transformer_encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    model.train()
    for step in range(steps):
        loss = sum(loss_fn(emb(trunk(model, x.unsqueeze(0))[0]), teach[v]) for v, x in xs.items())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"[adapt] step {step} loss {loss.item() / len(xs):.3f}")
    model.eval()
    with torch.no_grad():
        return {v: emb(trunk(model, x.unsqueeze(0))[0]).numpy() for v, x in xs.items()}


def changeit_prompts():
    """'{object}, {state}' for every initial/end state of ChangeIt (VidOSC's training vocabulary)."""
    import pandas as pd
    from .vidosc import VIDOSC
    df = pd.read_csv(os.path.join(VIDOSC, "data_changeit", "state_description_vocab.csv"))
    return sorted({(o, s.strip()) for o, a, b in zip(df["Object"], df["Initial_State"], df["End_State"]) for s in (a, b)})
