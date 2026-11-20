'Embedding-output inference and self-training helpers. Predictions are scored in chefOST-data.'
import os


import numpy as np
import torch

from ..vidosc import hard_pseudo_labels, load_vidosc, trunk


def smooth(e, window=1):
    """Moving average of [T, D] embeddings over `window` frames, renormalized (window 1 = unchanged)."""
    w = int(window)
    if w > 1:
        k = np.ones(w) / w
        e = np.stack([np.convolve(e[:, j], k, mode="same") for j in range(e.shape[1])], 1)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


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
    from ..vidosc import VIDOSC
    df = pd.read_csv(os.path.join(VIDOSC, "data_changeit", "state_description_vocab.csv"))
    return sorted({(o, s.strip()) for o, a, b in zip(df["Object"], df["Initial_State"], df["End_State"]) for s in (a, b)})
