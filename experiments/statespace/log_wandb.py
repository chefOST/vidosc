"""Logs the state-space investigation (2026-10-21/22) to W&B as one summary run. Numbers are copied from the
runs listed in README.md; discovery CIs are recomputed from runs/ss-ci-*."""
import json
import os
import sys

import wandb

import compare_disc
from corpus import ROOT

dest = json.load(open(os.path.join(ROOT, "weights_and_biases", "settings.json")))
run = wandb.init(entity=dest["entity"], project=dest["projects"]["experiments"], name="ryan/state-space-matcha-adjacent",
                 job_type="analysis", tags=["state-space", "matcha", "zero-shot", "proposal"],
                 notes="MATCHA-adjacent state-sensitive spaces on SigLIP SO400M. No MOSCATO data in training; MOSCATO GT "
                       "used only to score. See experiments/statespace/README.md.",
                 config={"moscato_use": "GT scoring only", "encoder": "SigLIP SO400M/14-384"})

diag = [  # where SigLIP is state-blind
    ["EPIC held-out: frame -> before vs after phrase (2AFC), all verbs", 0.538],
    ["EPIC held-out: same, state-changing verbs only", 0.487],
    ["EPIC held-out: text paraphrase vs LLM counterfactual", 0.936],
    ["CMU dev: change AUC, whole frames", 0.523],
    ["CMU dev: change AUC, crops", 0.468],
]
heads = [  # name, EPIC val cross 2AFC, text acc, noun retention (init .24 on state-only val), CMU dev frame top1
    ["SigLIP (centred)", 0.538, 0.936, None, 0.296],
    ["LDA state subspace (k=256)", None, None, None, None],
    ["MLP head: text triplets + frame margins (smoke)", 0.574, 0.969, None, 0.179],
    ["MLP head: cross-modal InfoNCE", 0.530, 0.908, 0.064, None],
    ["MLP head: InfoNCE + keep", 0.542, 0.903, 0.127, None],
    ["MLP head: InfoNCE + keep, state verbs, no location", 0.475, 0.914, 0.203, None],
    ["image-only gap triplets (EPIC action structure), val acc init .699", 0.685, None, None, None],
]
run.log({"diagnostics": wandb.Table(columns=["check", "value"], data=diag),
         "trained_heads": wandb.Table(columns=["head", "epic_cross_2afc", "text_acc", "keep_noun", "cmu_dev_frame_top1"],
                                      data=heads),
         "desc_match_test_top1": wandb.Table(columns=["space", "top1"], data=[
             ["SigLIP raw", 0.392], ["SigLIP centred", 0.404], ["LDA k=64..512", 0.302], ["LDA best (k64 eps1)", 0.367],
             ["MATCHA (earlier)", 0.233]]),
         "axes_tracks_cmu_dev_frames": wandb.Table(columns=["space", "change_auc", "sep", "frame_top1", "seg_top1"], data=[
             ["SigLIP centred", 0.523, 0.283, 0.296, 0.317], ["contradiction axes", 0.576, 0.393, 0.217, 0.237]])})
rows = []
for split in ("dev", "test"):
    for f in ("siglip", "crops-siglip"):
        A, B = compare_disc.load(f"ss-ci-{split}-{f}-axes"), compare_disc.load(f"ss-ci-{split}-{f}-base")
        vids = sorted(set(A) & set(B))
        ma, mb = compare_disc.metrics(A, vids), compare_disc.metrics(B, vids)
        rows += [[split, f, k, ma[k], mb[k]] for k in ma if k in mb]
run.log({"discovery_axes_vs_siglip": wandb.Table(columns=["split", "feats", "metric", "axes_seg_pen800", "siglip_pen500"],
                                                 data=rows)})
print(run.url)
run.finish()
