'Generate per-frame attribute predictions; all ground-truth scoring lives in chefOST-data.'
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from .. import ROOT, data, decoders, encoders, scoring, vidosc
from ..internvideo import video_size
from . import attributes

METHODS = ["prior", "state_zs", "zeroshot", "adapt"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=os.path.join(ROOT, "..", "chefOST-data", "MOSCATO"))
    p.add_argument("--dataset", default="CMU")
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--encoder", default="internvideo", help="video-level encoder (adapt needs InternVideo)")
    p.add_argument("--templates", default="basic", choices=list(scoring.TEMPLATES))
    p.add_argument("--decoder", default="viterbi:2", help="applied to each attribute independently")
    p.add_argument("--values", default="seen", choices=["seen", "all"],
                   help="candidate attributes/values per object: seen in training pseudo-labels, or the whole schema")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--ckpt", default="multi_task")
    p.add_argument("--adapt_steps", type=int, default=300)
    p.add_argument("--adapt_lr", type=float, default=1e-4)
    p.add_argument("--adapt_keep", type=float, default=0.5)
    p.add_argument("--adapt_teacher", default="zeroshot", choices=["zeroshot", "state_zs"],
                   help="pseudo-labels adapt is trained on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    return p.parse_args()


def describe(preds, t):
    """Human-readable state at frame t: 'form=mixed, location=transferred'."""
    return ", ".join(f"{a}={p[t]}" for a, p in preds.items() if p[t] is not None)


def main():
    a = parse_args()
    ds = data.Moscato(a.data_root, a.dataset, cache_dir=os.path.join(ROOT, "data", "cache"))
    schema = attributes.Schema(a.dataset)
    state_cands = ds.candidate_states("pseudo_train")
    enc = encoders.build_encoder(a.encoder)
    if not getattr(enc, "video_level", False):
        raise SystemExit("this pipeline uses video-level features: --encoder internvideo")
    temperature = getattr(enc, "logit_scale", 100.0)
    name = a.name or f"attr-{a.values}-{a.templates}-{time.strftime('%m%d-%H%M%S')}"
    out_dir = os.path.join(ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Features per video; per object: candidates and zero-shot probabilities.
    pairs = []
    for vid in a.videos:
        video = data.cmu_video(vid, os.path.join(ROOT, "data", "cmu_videos"))
        step = round(ds.src_fps / a.fps)
        idx = list(range(0, min(ds.num_frames(vid), video_size(video)[2]), step))
        cache = os.path.join(ROOT, "data", "cache", "emb", enc.name, f"{vid}_fps{a.fps:g}.npy")
        raw, _ = encoders.cached_video_embeddings(enc, video, idx, ds.src_fps, cache)
        emb = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        for obj in a.objects or ds.objects(vid):
            cands = attributes.candidates(schema, ds, vid, obj, a.values)
            pr = {"vid": vid, "obj": obj, "idx": idx, "raw": raw, "cands": cands,
                  "obj_emb": enc.encode_texts([f"a photo of {obj}."])[0],
                  "zs": {at: scoring.state_probs(emb, scoring.text_embeddings(
                      enc, obj, [schema.phrases[at][v] for v in vs], a.templates), temperature)
                         for at, vs in cands.items()}}
            states = state_cands.get(obj)
            if states:
                sp = scoring.state_probs(emb, scoring.text_embeddings(enc, obj, states, a.templates), temperature)
                pr["state_zs"] = {at: attributes.from_states(schema, obj, sp, states, at, vs) for at, vs in cands.items()}
            pairs.append(pr)

    # 2. Per method: {attribute: [T, len(cands)] probabilities, or None}.
    probs = defaultdict(dict)
    for pr in pairs:
        key = (pr["vid"], pr["obj"])
        if "prior" in a.methods:
            counts = attributes.value_counts(schema, ds, pr["vid"], pr["obj"])
            probs["prior"][key] = {at: np.eye(len(vs))[[max(range(len(vs)), key=lambda k: counts[at][vs[k]])] * len(pr["idx"])]
                                   for at, vs in pr["cands"].items()}
        if "state_zs" in a.methods:
            probs["state_zs"][key] = pr.get("state_zs", {})
        if "zeroshot" in a.methods:
            probs["zeroshot"][key] = pr["zs"]
    if "adapt" in a.methods:
        for pr in pairs:
            pr["x"] = vidosc.vidosc_input(pr["raw"])
            teacher = pr["zs"] if a.adapt_teacher == "zeroshot" else pr.get("state_zs", {})
            pr["teacher"] = {at: p for at, p in teacher.items() if p is not None}
        probs["adapt"] = vidosc.adapt_attributes(pairs, schema.attrs, a.ckpt, steps=a.adapt_steps, lr=a.adapt_lr,
                                                 keep=a.adapt_keep, seed=a.seed)

    # 3. Save decoded attribute states and unthresholded probabilities.
    results = {"args": vars(a), "methods": {}}
    for m in a.methods:
        per_pair = {}
        for pr in pairs:
            preds, raw = {}, {}
            for at in schema.attrs:
                vs = pr["cands"].get(at)
                p = probs[m][(pr["vid"], pr["obj"])].get(at) if vs else None
                preds[at] = [None] * len(pr["idx"]) if p is None else [vs[k] for k in decoders.decode(a.decoder, p)]
                raw[at] = {"values": vs, "probabilities": None if p is None else p.tolist()}
            per_pair[f"{pr['vid']}/{pr['obj']}"] = {
                "source_frame_indices": pr["idx"], "attributes": raw,
                "timeline": [{"t": i / ds.src_fps, "pred": describe(preds, k)} for k, i in enumerate(pr["idx"])],
            }
        results["methods"][m] = {"per_pair": per_pair}
    json.dump(results, open(os.path.join(out_dir, "predictions.json"), "w"), indent=1)
    print(f"[saved] {out_dir}")

if __name__ == "__main__":
    main()
