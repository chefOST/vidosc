"""Track the states of given objects in a few MOSCATO videos and score against ground truth.

Methods (--method):
  prior            no video: the object's most frequent state in training pseudo-labels of the recipe (baseline)
  zeroshot         encoder-text similarity over the object's candidate states (VidOSC's pseudo-labeling stage)
  vidosc_changeit  VidOSC multi-task ChangeIt checkpoint off the shelf; candidates mapped to ChangeIt states
  vidosc_adapt     the checkpoint's temporal transformer + a new head per object, fine-tuned on the zero-shot
                   pseudo-labels of the selected videos (needs --encoder internvideo)

Example:
    python -m pipeline.states.run --videos S12_Sandwich_7150991-2470 S48_Salad_7150991-222 \
        --encoder internvideo --method vidosc_adapt --decoder viterbi:2 --wandb
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from .. import ROOT, data, decoders, encoders, metrics, scoring, vidosc
from ..internvideo import video_size
from . import clusters



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=os.path.join(ROOT, "..", "chefOST-data", "MOSCATO"))
    p.add_argument("--dataset", default="CMU")
    p.add_argument("--videos", nargs="+", required=True, help="MOSCATO video ids (ground-truth split)")
    p.add_argument("--objects", nargs="*", default=None, help="objects to track (default: all annotated)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--encoder", default="clip:ViT-L-14/openai")
    p.add_argument("--templates", default="ensemble", choices=list(scoring.TEMPLATES))
    p.add_argument("--temperature", type=float, default=None, help="default: the encoder's logit scale, else 100")
    p.add_argument("--decoder", default="viterbi:2")
    p.add_argument("--method", default="zeroshot", choices=["prior", "zeroshot", "vidosc_changeit", "vidosc_adapt"])
    p.add_argument("--ckpt", default="multi_task", help="VidOSC checkpoint: multi_task, open_world or a path")
    p.add_argument("--adapt_steps", type=int, default=300)
    p.add_argument("--adapt_lr", type=float, default=1e-4)
    p.add_argument("--adapt_keep", type=float, default=0.5, help="fraction of most confident frames pseudo-labeled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state_source", default="pseudo_train", choices=["pseudo_train", "pseudo_recipe", "all"])
    p.add_argument("--clusters", default=None, help="near-synonym cluster file in pipeline/states/state_clusters, e.g. CMU")
    p.add_argument("--within", default="prior", choices=["prior", "model"], help="member choice inside a cluster")
    p.add_argument("--name", default=None)
    p.add_argument("--wandb", action="store_true", help="log to W&B (destination: weights_and_biases/settings.json)")
    p.add_argument("--wandb_project", default=None, help="override the experiments project")
    return p.parse_args()


def internvideo_frames(video):
    return video_size(video)[2]


def timeline_figure(t, pred, gt_sets, states, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(set(states) | set().union(*gt_sets))
    y = {s: i for i, s in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(rows) + 1.2))
    for s in rows:
        ts = [ti for ti, g in zip(t, gt_sets) if s in g]
        ax.scatter(ts, [y[s]] * len(ts), marker="s", s=30, color="#bbbbbb", label="ground truth" if s == rows[0] else None)
    ok = [p in g for p, g in zip(pred, gt_sets)]
    ax.scatter(t, [y[p] for p in pred], s=8, c=["#1a9850" if o else "#d73027" for o in ok], label="prediction")
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def main():
    a = parse_args()
    ds = data.Moscato(a.data_root, a.dataset, cache_dir=os.path.join(ROOT, "data", "cache"))
    cands = ds.candidate_states(a.state_source)
    enc = encoders.build_encoder(a.encoder)
    name = a.name or f"{a.method}-{a.encoder.split(':')[0]}-{a.templates}-{a.decoder.replace(':', '')}-{time.strftime('%m%d-%H%M%S')}"
    out_dir = os.path.join(ROOT, "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    run = None
    if a.wandb:
        import wandb
        dest = json.load(open(os.path.join(ROOT, "weights_and_biases", "settings.json")))
        run = wandb.init(entity=dest["entity"], project=a.wandb_project or dest["projects"]["experiments"],
                         name=name, job_type=a.method, config={**vars(a), "encoder_name": enc.name,
                                                                  "eval_frames": "object has >=1 GT state"})

    temperature = a.temperature or getattr(enc, "logit_scale", 100.0)
    if a.method != "zeroshot" and not getattr(enc, "video_level", False):
        raise SystemExit(f"--method {a.method} needs InternVideo features: --encoder internvideo")

    if a.clusters and a.method in ("vidosc_changeit", "prior"):
        raise SystemExit("--clusters applies to zeroshot and vidosc_adapt only")
    s2c = clusters.load(a.clusters) if a.clusters else None

    # 1. Per video: features at the sampled frames. Per object: candidates and zero-shot probabilities.
    pairs, timing = [], Counter()
    for vid in a.videos:
        t0 = time.time()
        video = data.cmu_video(vid, os.path.join(ROOT, "data", "cmu_videos"))
        cache = os.path.join(ROOT, "data", "cache", "emb", enc.name, f"{vid}_fps{a.fps:g}.npy")
        if getattr(enc, "video_level", False):
            step = round(ds.src_fps / a.fps)
            n = min(ds.num_frames(vid), internvideo_frames(video))
            idx = list(range(0, n, step))
            timing["frame_s"] += time.time() - t0
            t0 = time.time()
            raw, hit = encoders.cached_video_embeddings(enc, video, idx, ds.src_fps, cache)
            emb = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        else:
            idx, paths = data.sample_frames(video, ds.src_fps, a.fps, os.path.join(ROOT, "data", "frames", f"{vid}_fps{a.fps:g}"))
            idx_ok = [k for k, i in enumerate(idx) if i < ds.num_frames(vid)]
            idx, paths = [idx[k] for k in idx_ok], [paths[k] for k in idx_ok]
            timing["frame_s"] += time.time() - t0
            t0 = time.time()
            emb, hit = encoders.cached_image_embeddings(enc, paths, cache)
            raw = emb
        timing["encode_s"] += time.time() - t0
        timing["frames"] += len(idx)
        timing["cached_videos"] += hit
        for obj in a.objects or ds.objects(vid):
            if obj not in ds.gt[vid]["state"]:
                print(f"[skip] {vid}: {obj} not annotated")
                continue
            states = ds.candidates_for(cands, a.state_source, vid, obj)
            if not states:
                print(f"[skip] {vid}/{obj}: no candidate states")
                continue
            zs = scoring.state_probs(emb, scoring.text_embeddings(enc, obj, states, a.templates), temperature)
            pr = {"vid": vid, "obj": obj, "idx": idx, "states": states, "zs": zs, "raw": raw,
                  "gt": ds.gt_states(vid, obj, idx), "head_states": states}
            if s2c:  # decode over clusters; members are chosen after decoding
                pr["groups"] = clusters.group(states, s2c)
                pr["member_zs"], pr["zs"] = zs, clusters.aggregate(zs, pr["groups"])
            pairs.append(pr)

    # 2. Method: per-frame probabilities over the states it can predict.
    t0 = time.time()
    if a.method == "prior":
        for pr in pairs:
            prior = ds.state_prior(pr["vid"], pr["obj"])
            top = max(range(len(pr["states"])), key=lambda k: prior.get(pr["states"][k], 0))
            pr["pred_states"], pr["probs"] = pr["states"], np.eye(len(pr["states"]))[[top] * len(pr["idx"])]
            pr.pop("groups", None)
    elif a.method == "zeroshot":
        for pr in pairs:
            pr["pred_states"], pr["probs"] = pr["states"], pr["zs"]
    elif a.method == "vidosc_changeit":
        model = vidosc.load_vidosc(a.ckpt)
        vocab, _ = vidosc.changeit_vocab()
        for pr in pairs:
            m = vidosc.changeit_map(pr["states"], vocab)
            sp, _ = vidosc.changeit_probs(model, vidosc.vidosc_input(pr["raw"]))
            pr["pred_states"] = list(m)
            pr["probs"] = sp[:, list(m.values())] if m else None
    else:
        for pr in pairs:
            pr["x"] = vidosc.vidosc_input(pr["raw"])
        probs = vidosc.adapt(pairs, a.ckpt, steps=a.adapt_steps, lr=a.adapt_lr, keep=a.adapt_keep, seed=a.seed)
        for pr in pairs:
            pr["pred_states"], pr["probs"] = pr["states"], probs[(pr["vid"], pr["obj"])]
    timing["method_s"] += time.time() - t0

    # 3. Decode over time and score.
    totals, rows, results, figs = Counter(), [], {}, {}
    for pr in pairs:
        vid, obj, states, gt = pr["vid"], pr["obj"], pr["pred_states"], pr["gt"]
        t_sec = [i / ds.src_fps for i in pr["idx"]]
        t0 = time.time()
        if pr["probs"] is None:  # nothing this method can predict for this object
            pred = [None] * len(gt)
        else:
            p = pr["probs"] / pr["probs"].sum(1, keepdims=True)
            seq = decoders.decode(a.decoder, p)
            if "groups" in pr and a.method != "vidosc_changeit":
                pred = clusters.expand(seq, pr["groups"], states, pr["member_zs"], ds.state_prior(vid, obj), a.within)
            else:
                pred = [states[k] for k in seq]
        timing["decode_s"] += time.time() - t0
        c = metrics.score(pred, gt, states)
        totals.update(c)
        s = metrics.summarize(c)
        rows.append([vid, obj, len(states), len(pr.get("groups", states)), s["evaluated_frames"], s["precision"], s["oracle"], s["chance"]])
        results[f"{vid}/{obj}"] = {"candidates": states, "t": t_sec, "pred": pred, "gt": [sorted(g) for g in gt], **s}
        if s["evaluated_frames"] and pr["probs"] is not None:
            figs[f"{vid}/{obj}"] = timeline_figure(t_sec, pred, gt, states,
                                                   f"{vid} / {obj}: precision {s['precision']:.2f} (oracle {s['oracle']:.2f})")
        print(f"{vid:28s} {obj:22s} K={len(states):3d} C={len(pr.get('groups', states)):3d} eval={s['evaluated_frames']:4d} "
              f"precision={s['precision']:.3f} oracle={s['oracle']:.3f} chance={s['chance']:.3f}")

    overall = metrics.summarize(totals)
    # Throughput (sampled frames per second of encode + decode) only when no embeddings came from cache.
    tput = timing["frames"] / (timing["encode_s"] + timing["method_s"] + timing["decode_s"]) \
        if not timing["cached_videos"] else float("nan")
    overall.update({"videos": len(a.videos), "video_object_pairs": len(rows), "throughput_fps": tput, **timing})
    print("\n" + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in overall.items()))
    json.dump({"args": vars(a), "overall": overall, "per_pair": results}, open(os.path.join(out_dir, "results.json"), "w"))
    for key, fig in figs.items():
        fig.savefig(os.path.join(out_dir, key.replace("/", "__") + ".png"), dpi=110)
    print(f"[saved] {out_dir}")

    if run:
        import wandb
        cols = ["video", "object", "num_candidates", "num_clusters", "evaluated_frames", "precision", "oracle", "chance"]
        run.log({"per_object": wandb.Table(columns=cols, data=rows),
                 **{f"timeline/{k}": wandb.Image(f) for k, f in figs.items()}})
        run.summary.update(overall)
        print(f"W&B: {run.url}")
        run.finish()


if __name__ == "__main__":
    main()
