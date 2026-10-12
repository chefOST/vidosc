"""VidOSC's ChangeIt temporal transformer (VidOSC/model.py) on MOSCATO features.

Input per second: [global InternVideo feature, object-centric InternVideo feature] (1536-d), as in
the multi-task checkpoint (--det 1). VidOSC fills the object half with zeros when its hand-object
detector finds nothing; we do the same until we add object crops.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDOSC = os.path.join(ROOT, "VidOSC")
CHECKPOINTS = {"multi_task": "checkpoints_changeit/multi_task/state_best.ckpt",
               "open_world": "checkpoints_changeit/open_world/state_best.ckpt"}


def changeit_vocab(dataset_root=os.path.join(VIDOSC, "data_changeit"), classes=None):
    """State and action vocabularies in the order VidOSC built them (dataset_changeit.build_vocab)."""
    df = pd.read_csv(os.path.join(dataset_root, "state_description_vocab.csv"))
    if classes is not None:
        df = df[df["Dir_Name"].isin(classes)]
    states = list(dict.fromkeys(["background"] + df["Initial_State"].tolist() + df["End_State"].tolist()))
    actions = list(dict.fromkeys(["background"] + df["Action"].tolist()))
    return states, actions


def load_vidosc(ckpt="multi_task", device="cpu"):
    sys.path.insert(0, VIDOSC)
    from model import FeatTimeTransformerDualHead
    path = os.path.join(VIDOSC, CHECKPOINTS.get(ckpt, ckpt))
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    args = argparse.Namespace(input_dim=sd["proj.weight"].shape[1], transformer_dim=sd["proj.weight"].shape[0],
                              transformer_heads=4, transformer_layers=3, transformer_dropout=0.1,
                              state_classes=sd["state_head.weight"].shape[0],
                              action_classes=sd["action_head.weight"].shape[0])
    model = FeatTimeTransformerDualHead(args)
    model.load_state_dict(sd)
    return model.to(device).eval()


def vidosc_input(global_feat, obj_feat=None):
    """[T, 1536] float tensor; zeros for the object half when no object features are given."""
    g = torch.as_tensor(np.asarray(global_feat), dtype=torch.float32)
    o = torch.zeros_like(g) if obj_feat is None else torch.as_tensor(np.asarray(obj_feat), dtype=torch.float32)
    return torch.cat([g, o], -1)


@torch.no_grad()
def changeit_probs(model, x):
    """Off-the-shelf checkpoint: per-second softmax over ChangeIt states [T, S] and actions [T, A]."""
    out = model(x.unsqueeze(0))
    return torch.softmax(out["state"], -1).numpy(), torch.softmax(out["action"], -1).numpy()


# ---- Method "vidosc_changeit": the checkpoint off the shelf ---------------------------------
# MOSCATO states that have a ChangeIt counterpart. Exact matches plus these synonyms; the rest of
# an object's candidates cannot be predicted by this method (its `oracle` shows the coverage).
SYNONYMS = {"chopped": "cut", "sliced": "cut", "grinded": "ground", "beaten": "whisked", "cooked": "fried"}


def changeit_map(states, vocab):
    """MOSCATO state -> ChangeIt state index, for the states that have one."""
    idx = {s.strip(): i for i, s in enumerate(vocab)}
    return {s: idx[SYNONYMS.get(s, s)] for s in states if SYNONYMS.get(s, s) in idx and s != "background"}


# ---- Method "vidosc_adapt": checkpoint trunk + a new head per object -------------------------
def trunk(model, x):
    """VidOSC's projection + temporal transformer, without the ChangeIt heads: [B, T, 512]."""
    h = model.pos_encoder(model.ln(model.proj(x)))
    return model.transformer_encoder(h)


def hard_pseudo_labels(p, keep=0.5):
    """VidOSC-style confident pseudo-labels: argmax state on the `keep` fraction of most confident
    frames, -1 (ignored) elsewhere."""
    conf = p.max(1)
    lab = p.argmax(1)
    lab[conf < np.quantile(conf, 1 - keep)] = -1
    return torch.as_tensor(lab, dtype=torch.long)


def adapt(pairs, ckpt="multi_task", steps=300, lr=1e-4, wd=1e-4, keep=0.5, seed=0, log=print):
    """Fine-tune the checkpoint's trunk with one new state head per object, on the pseudo-labels of
    the given (video, object) pairs only. pairs: dicts with 'vid', 'obj', 'x' [T, 1536], 'zs' [T, K].
    Returns {(vid, obj): probs [T, K]}."""
    torch.manual_seed(seed)
    model = load_vidosc(ckpt)
    # One head per object and candidate set (the same object can have different candidates per recipe).
    key = {(pr["vid"], pr["obj"]): (pr["obj"], tuple(pr.get("head_states", [pr["zs"].shape[1]]))) for pr in pairs}
    slot = {kk: i for i, kk in enumerate(dict.fromkeys(key.values()))}
    sizes = {slot[key[(pr["vid"], pr["obj"])]]: pr["zs"].shape[1] for pr in pairs}
    heads = torch.nn.ModuleList([torch.nn.Linear(model.proj.out_features, sizes[i]) for i in range(len(slot))])
    head = lambda pr: heads[slot[key[(pr["vid"], pr["obj"])]]]
    labels = {(pr["vid"], pr["obj"]): hard_pseudo_labels(pr["zs"], keep) for pr in pairs}
    xs = {pr["vid"]: pr["x"] for pr in pairs}
    params = list(model.proj.parameters()) + list(model.ln.parameters()) + \
        list(model.transformer_encoder.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    for step in range(steps):
        loss = 0.0
        for vid, x in xs.items():
            h = trunk(model, x.unsqueeze(0))[0]
            for pr in pairs:
                if pr["vid"] == vid:
                    loss = loss + ce(head(pr)(h), labels[(vid, pr["obj"])])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"[adapt] step {step} loss {loss.item() / len(pairs):.3f}")
    model.eval()
    out = {}
    with torch.no_grad():
        for vid, x in xs.items():
            h = trunk(model, x.unsqueeze(0))[0]
            for pr in pairs:
                if pr["vid"] == vid:
                    out[(vid, pr["obj"])] = torch.softmax(head(pr)(h), -1).numpy()
    return out
