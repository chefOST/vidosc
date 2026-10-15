"""Attribute pipeline: track each object as a set of attribute values per frame (form, heat, fill, ...)
instead of one MOSCATO state, and score per attribute. Proposal, not the README metric; see
pipeline/attributes.py and state_attributes/<dataset>.json.

Methods (all run on the same features, --methods):
  prior     no video: each attribute's most frequent value for the object in training pseudo-labels
  state_zs  the state pipeline's zero-shot state probabilities, summed into attribute values
  zeroshot  encoder-text similarity per attribute, prompts '{object}, {value phrase}'
  adapt     VidOSC trunk + one head per attribute shared across objects (conditioned on the object's
            text embedding), fine-tuned on the zeroshot attribute pseudo-labels of the selected videos

Example:
    python -m pipeline.run_attributes --videos S08_Eggs_7150991-727
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from . import attributes, data, decoders, encoders, scoring, vidosc
from .internvideo import video_size

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def summarize(counts, nontrivial, schema):
    """counts: {attribute: Counter}. Micro accuracy over scored (frame, attribute) pairs, macro over
    attributes, and per-attribute accuracy / oracle / chance / ordinal distance. The nontrivial_*
    numbers leave out (object, attribute) pairs with a single candidate value (always correct when
    the GT value is that candidate)."""
    per = {}
    for a, c in counts.items():
        if c["frames"]:
            per[a] = {"frames": c["frames"], "accuracy": c["correct"] / c["frames"], "oracle": c["oracle"] / c["frames"],
                      "chance": c["chance"] / c["frames"]}
            if schema.ordered[a]:
                per[a]["dist"] = c["dist"] / c["frames"]
    n = sum(c["frames"] for c in counts.values())
    n_nt = sum(c["frames"] for c in nontrivial.values())
    return {"micro_accuracy": sum(c["correct"] for c in counts.values()) / n if n else float("nan"),
            "macro_accuracy": float(np.mean([v["accuracy"] for v in per.values()])) if per else float("nan"),
            "nontrivial_accuracy": sum(c["correct"] for c in nontrivial.values()) / n_nt if n_nt else float("nan"),
            "nontrivial_chance": sum(c["chance"] for c in nontrivial.values()) / n_nt if n_nt else float("nan"),
            "scored": n, "nontrivial_scored": n_nt, "per_attribute": per}


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

    # 1. Features per video; per object: GT attribute values, candidates, zero-shot probabilities.
    pairs, coverage = [], Counter()
    for vid in a.videos:
        video = data.cmu_video(vid, os.path.join(ROOT, "data", "cmu_videos"))
        step = round(ds.src_fps / a.fps)
        idx = list(range(0, min(ds.num_frames(vid), video_size(video)[2]), step))
        cache = os.path.join(ROOT, "data", "cache", "emb", enc.name, f"{vid}_fps{a.fps:g}.npy")
        raw, _ = encoders.cached_video_embeddings(enc, video, idx, ds.src_fps, cache)
        emb = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        for obj in a.objects or ds.objects(vid):
            gt_sets = ds.gt_states(vid, obj, idx)
            for g in gt_sets:
                for s in g:
                    coverage["mapped" if schema.values_of(obj, s) else "unmapped"] += 1
            cands = attributes.candidates(schema, ds, vid, obj, a.values)
            pr = {"vid": vid, "obj": obj, "idx": idx, "raw": raw, "cands": cands,
                  "gt": attributes.gt_values(schema, obj, gt_sets), "gt_states": gt_sets,
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

    # 3. Decode each attribute over time, score, describe.
    results = {"args": vars(a), "coverage": coverage["mapped"] / max(1, sum(coverage.values())), "methods": {}}
    for m in a.methods:
        totals, nontrivial, per_pair = defaultdict(Counter), defaultdict(Counter), {}
        for pr in pairs:
            preds, rows = {}, {}
            for at in schema.attrs:
                vs = pr["cands"].get(at)
                p = probs[m][(pr["vid"], pr["obj"])].get(at) if vs else None
                preds[at] = [None] * len(pr["idx"]) if p is None else [vs[k] for k in decoders.decode(a.decoder, p)]
                c = attributes.score(preds[at], pr["gt"][at], vs, schema.attrs[at] if schema.ordered[at] else None)
                totals[at].update(c)
                if vs and len(vs) > 1:
                    nontrivial[at].update(c)
                if c["frames"]:
                    rows[at] = {"accuracy": c["correct"] / c["frames"], "frames": c["frames"],
                                "gt_values": sorted(set().union(*pr["gt"][at])), "pred_values": sorted(set(preds[at]) - {None})}
            per_pair[f"{pr['vid']}/{pr['obj']}"] = {
                "per_attribute": rows,
                "timeline": [{"t": i / ds.src_fps, "pred": describe(preds, k), "gt_states": sorted(g)}
                             for k, (i, g) in enumerate(zip(pr["idx"], pr["gt_states"]))]}
        s = summarize(totals, nontrivial, schema)
        results["methods"][m] = {**s, "per_pair": per_pair}
        print(f"\n== {m}: micro {s['micro_accuracy']:.3f}  macro {s['macro_accuracy']:.3f}  ({s['scored']} scored frame-attributes)"
              f"  nontrivial {s['nontrivial_accuracy']:.3f} (chance {s['nontrivial_chance']:.3f}, n={s['nontrivial_scored']})")
        for at, v in s["per_attribute"].items():
            d = f" dist {v['dist']:.3f}" if "dist" in v else ""
            print(f"   {at:9s} n={v['frames']:4d} acc {v['accuracy']:.3f} oracle {v['oracle']:.3f} chance {v['chance']:.3f}{d}")
    print(f"\nGT state coverage by the schema: {results['coverage']:.3f}")
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=str)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
