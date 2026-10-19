"""Embedding-output pipeline: methods output a text-space vector per frame and object, scored by
ranking the object's embedded candidate MOSCATO states. Proposal, not the README metric; see
pipeline/embedding/embed.py.

Methods (--methods):
  zeroshot  the frame's video embedding itself (InternVideo's video and text towers share a space)
  adapt     VidOSC trunk + a projection into the text space, self-trained on the selected videos with no
            MOSCATO labels or states (--adapt_target). Per frame, the same for every object.

MOSCATO is used only for scoring: --state_source picks the states the scorer ranks.

--smooth averages output embeddings over time (the embedding analogue of a decoder).

Example:
    python -m pipeline.embedding.run --videos S08_Eggs_7150991-727 --smooth 5
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from .. import ROOT, data, encoders, scoring, vidosc
from ..internvideo import video_size
from . import embed

METHODS = ["zeroshot", "adapt"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=os.path.join(ROOT, "..", "chefOST-data", "MOSCATO"))
    p.add_argument("--dataset", default="CMU")
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--encoder", default="internvideo", help="video-level encoder (adapt needs InternVideo)")
    p.add_argument("--templates", default="basic", choices=list(scoring.TEMPLATES))
    p.add_argument("--state_source", default="pseudo_train", choices=["pseudo_train", "pseudo_recipe", "all"],
                   help="candidate states ranked at scoring time (and the adapt pseudo-label vocabulary)")
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--ckpt", default="multi_task")
    p.add_argument("--adapt_steps", type=int, default=300)
    p.add_argument("--adapt_lr", type=float, default=1e-4)
    p.add_argument("--adapt_keep", type=float, default=0.5)
    p.add_argument("--adapt_target", default="frame", choices=["frame", "changeit"],
                   help="self-training target: the zero-shot frame embedding, or pseudo-labels over ChangeIt states")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--wandb", action="store_true", help="log to W&B (destination: weights_and_biases/settings.json)")
    p.add_argument("--wandb_project", default=None, help="override the experiments project")
    return p.parse_args()


def log_wandb(a, enc, name, results, pairs, errs):
    """One W&B run: config, headline summary per method, and readable tables."""
    import wandb
    dest = json.load(open(os.path.join(ROOT, "weights_and_biases", "settings.json")))
    head = {m: r for m, r in results["methods"].items()}
    notes = "Embedding-output pipeline (proposal, not the README metric). Zero-shot: MOSCATO used only to score. " + \
        "; ".join(f"{m}: top1 {r['top1']:.3f}, MRR {r['mrr']:.3f} (chance {r['chance']:.3f})" for m, r in head.items())
    run = wandb.init(entity=dest["entity"], project=a.wandb_project or dest["projects"]["experiments"], name=name,
                     job_type="embed", notes=notes, tags=["embedding-output", "zero-shot", "proposal"],
                     config={**vars(a), "encoder_name": enc.name, "moscato_use": "GT scoring only (ranked candidates)",
                             "scored_frames": "sampled frames where the object has >=1 GT state",
                             "top1": "nearest candidate state is in the frame's GT set",
                             "mrr": "1 / rank of the best-ranked GT state"})
    for m, r in head.items():
        run.summary.update({f"{m}/top1": r["top1"], f"{m}/mrr": r["mrr"], f"{m}/oracle": r["oracle"],
                            f"{m}/chance": r["chance"], f"{m}/top1_over_chance": r["top1"] / r["chance"]})
    run.summary.update({"scored_frames": next(iter(head.values()))["evaluated_frames"], "videos": len(a.videos),
                        "video_object_pairs": len(pairs), "bank_mean_pairwise_cosine": results["bank_mean_pairwise_cosine"]})
    cols = ["method", "video", "object", "candidates", "scored_frames", "top1", "mrr", "chance"]
    rows = [[m, *k.split("/", 1), v["candidates"], v["evaluated_frames"], v["top1"], v["mrr"], v["chance"]]
            for m, r in head.items() for k, v in r["per_pair"].items()]
    err_cols = ["method", "object", "predicted", "ground_truth", "frames", "text_cos_pred_to_gt"]
    err_rows = [[m, o, p, g, n, cs] for m, c in errs.items() for (o, p, g, cs), n in c.most_common(25)]
    run.log({"per_object": wandb.Table(columns=cols, data=rows),
             "top_errors": wandb.Table(columns=err_cols, data=err_rows)})
    print(f"W&B: {run.url}")
    run.finish()


def main():
    a = parse_args()
    ds = data.Moscato(a.data_root, a.dataset, cache_dir=os.path.join(ROOT, "data", "cache"))
    cands = ds.candidate_states(a.state_source)
    enc = encoders.build_encoder(a.encoder)
    if not getattr(enc, "video_level", False):
        raise SystemExit("this pipeline uses video-level features: --encoder internvideo")
    temperature = getattr(enc, "logit_scale", 100.0)
    name = a.name or f"embed-{a.templates}-s{a.smooth}-{time.strftime('%m%d-%H%M%S')}"
    out_dir = os.path.join(ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Features per video; per object: candidate bank, GT, zero-shot outputs.
    pairs = []
    for vid in a.videos:
        video = data.cmu_video(vid, os.path.join(ROOT, "data", "cmu_videos"))
        idx = list(range(0, min(ds.num_frames(vid), video_size(video)[2]), round(ds.src_fps / a.fps)))
        cache = os.path.join(ROOT, "data", "cache", "emb", enc.name, f"{vid}_fps{a.fps:g}.npy")
        raw, _ = encoders.cached_video_embeddings(enc, video, idx, ds.src_fps, cache)
        emb = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        for obj in a.objects or ds.objects(vid):
            states = ds.candidates_for(cands, a.state_source, vid, obj)
            if not states:
                print(f"[skip] {vid}/{obj}: no candidate states")
                continue
            bank = scoring.text_embeddings(enc, obj, states, a.templates)
            pairs.append({"vid": vid, "obj": obj, "idx": idx, "raw": raw, "states": states, "bank": bank,
                          "gt": ds.gt_states(vid, obj, idx), "emb": emb,
                          "zs": scoring.state_probs(emb, bank, temperature)})

    # 2. Per method: {(vid, obj): [T, D] outputs}.
    outputs = {}
    if "zeroshot" in a.methods:
        outputs["zeroshot"] = {(p["vid"], p["obj"]): p["emb"] for p in pairs}
    if "adapt" in a.methods:
        videos = {p["vid"]: (vidosc.vidosc_input(p["raw"]), p["emb"]) for p in pairs}
        bank = None
        if a.adapt_target == "changeit":
            bank = np.concatenate([scoring.text_embeddings(enc, o, [s], a.templates) for o, s in embed.changeit_prompts()])
        out = embed.adapt(videos, "bank" if bank is not None else "frame", bank, a.ckpt, steps=a.adapt_steps,
                          lr=a.adapt_lr, keep=a.adapt_keep, scale=temperature, seed=a.seed)
        outputs["adapt"] = {(p["vid"], p["obj"]): out[p["vid"]] for p in pairs}

    # 3. Smooth over time, rank the candidate bank, score.
    spread = float(np.nanmean([embed.bank_spread(p["bank"]) for p in pairs]))
    results = {"args": vars(a), "bank_mean_pairwise_cosine": spread, "methods": {}}
    errs = {}
    for m, outs in outputs.items():
        totals, per_pair, errs[m] = Counter(), {}, Counter()
        for p in pairs:
            o = embed.smooth(outs[(p["vid"], p["obj"])], a.smooth)
            c = embed.score(o, p["bank"], p["states"], p["gt"])
            totals.update(c)
            errs[m].update({(p["obj"], *k): n for k, n in embed.errors(o, p["bank"], p["states"], p["gt"]).items()})
            per_pair[f"{p['vid']}/{p['obj']}"] = {"candidates": len(p["states"]), **embed.summarize(c)}
        s = embed.summarize(totals)
        results["methods"][m] = {**s, "per_pair": per_pair}
        print(f"== {m}: top1 {s['top1']:.3f}  mrr {s['mrr']:.3f}  oracle {s['oracle']:.3f}  "
              f"chance {s['chance']:.3f}  ({s['evaluated_frames']} frames, {len(pairs)} pairs)")
    print(f"mean pairwise cosine within an object's candidate bank: {spread:.3f}")
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    print(f"[saved] {out_dir}")
    if a.wandb:
        log_wandb(a, enc, name, results, pairs, errs)


if __name__ == "__main__":
    main()
