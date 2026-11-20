'Generate and save per-frame text-space embedding outputs; scoring lives in chefOST-data.'
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
                   help="candidate vocabulary for saved inference outputs")
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

    # 1. Features per video; per object: candidate bank and zero-shot outputs.
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
                          "emb": emb,
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

    # 3. Save each method's embeddings with their source-frame alignment.
    results = {"args": vars(a), "methods": {}}
    files = []
    for m, outs in outputs.items():
        per_pair = {}
        for k, p in enumerate(pairs):
            values = embed.smooth(outs[(p["vid"], p["obj"])], a.smooth)
            filename = f"{m}-{k:04d}.npz"
            path = os.path.join(out_dir, filename)
            np.savez_compressed(path, embeddings=values, source_frame_indices=p["idx"],
                                timestamps_seconds=np.asarray(p["idx"]) / ds.src_fps)
            files.append(path)
            per_pair[f"{p['vid']}/{p['obj']}"] = {"embedding_file": filename, "candidate_states": p["states"]}
        results["methods"][m] = {"per_pair": per_pair}
    index = os.path.join(out_dir, "predictions.json")
    json.dump(results, open(index, "w"), indent=1)
    if a.wandb:
        import wandb
        dest = json.load(open(os.path.join(ROOT, "weights_and_biases", "settings.json")))
        run = wandb.init(entity=dest["entity"], project=a.wandb_project or dest["projects"]["experiments"],
                         name=name, job_type="predictions", config=vars(a))
        artifact = wandb.Artifact(name.replace('/', '-'), type='predictions')
        for path in files + [index]: artifact.add_file(path, name=os.path.basename(path))
        run.log_artifact(artifact)
        run.finish()
    print(f"[saved] {out_dir}")

if __name__ == "__main__":
    main()
