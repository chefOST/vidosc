'Discover object states and save per-frame embedding predictions. Scoring lives in chefOST-data.'
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
    """Yields (vid, obj, X [T, D] unit, w [T], source frame indices)."""
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
            yield vid, obj, discover.unit(X), w, idx


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
            for vid, obj, X, w, idx in all_tracks:
                if ds.candidates_for(cands, a.state_source, vid, obj):
                    outputs(a, X, w, lambda fr: (keys.update(describe.key(vid, t, obj) for t in fr),
                                                 np.zeros((len(fr), 1)))[1])
        desc = modal_jobs.describe(keys)
        vlm = lambda fr: bank_of(obj, [desc[describe.key(vid, t, obj)] for t in fr], a.templates)
    name = a.name or f"disc-{a.feats}-{a.split}-{time.strftime('%m%d-%H%M%S')}"
    out_dir = os.path.join(ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)
    rows, files = [], []
    for k, (vid, obj, X, w, idx) in enumerate(all_tracks):
        states = ds.candidates_for(cands, a.state_source, vid, obj)
        if not states:
            continue
        bank = bank_of(obj, states, a.templates)
        outs, labels = outputs(a, X, w, vlm)
        if "vlm_discover" in outs and "discover" in outs:
            vis = discover.unit(outs["discover"] - (a.debias * mu if mu is not None else 0))
            outs["fused"] = vis + outs["vlm_discover"]
        row = {"video": vid, "object": obj, "source_frame_indices": idx, "methods": {}}
        for m, values in outs.items():
            if m.startswith("_"):
                continue
            if mu is not None and m not in ("vlm_uniform", "vlm_discover", "fused"):
                values = values - a.debias * mu
            values = discover.unit(values)
            filename = f"{k:04d}-{m}.npz"
            path = os.path.join(out_dir, filename)
            np.savez_compressed(path, embeddings=values, source_frame_indices=idx, state_ids=labels)
            files.append(path)
            row["methods"][m] = {"embedding_file": filename,
                                  "predicted_states": [states[int(i)] for i in (values @ bank.T).argmax(1)]}
        rows.append(row)
    index = os.path.join(out_dir, "predictions.json")
    json.dump({"args": vars(a), "predictions": rows}, open(index, "w"), indent=1)
    if a.wandb:
        import wandb
        dest = json.load(open(f"{ROOT}/weights_and_biases/settings.json"))
        run = wandb.init(entity=dest["entity"], project=dest["projects"]["experiments"],
                         name=name, job_type="predictions", config=vars(a))
        artifact = wandb.Artifact(name.replace('/', '-'), type='predictions')
        for path in files + [index]: artifact.add_file(path, name=os.path.basename(path))
        run.log_artifact(artifact)
        run.finish()
    print(f"[saved] {out_dir}")

if __name__ == "__main__":
    main()
