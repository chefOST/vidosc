Pursue the attribute/slider idea to completion. I want a full pipeline that
  takes in an unannotated video (from MOSCATO), an object to track, and surfaces
  an arbitrary number of states. Use MOSCATO GT for evaluation (you may have to
  do some preprocessing to map GT labels to attributes). The pipeline will
  start with using an LLM to generate possible attributes for the given object.
  You may use VidOSC, adapted or as is, if you find it useful. You may use any
  VLMs you may find useful, or train small transformers, etc - just let me know
  beforehand, and let me know if compute might be expensive and if we need
  Modal. Once you have presentable and genuine research paper worthy results,
  create a short .tex writeup with the following sections: Method, Experiments
  (w results), Conclusion. Log any milestones with simple but sufficiently
  detailed telemetry on WB.

# Handoff: attributes → sliders (ChefOST experiment, 2026-10-17)

## Ground rules (from Ryan)
- Zero-shot. MOSCATO (GT, pseudo-labels, state vocabulary) is for evaluation only: no training, priors or candidate states from it.
- Pipeline input is (video, object name). Using the object name, and the recipe, in prompts or LLM calls is allowed.
- External checkpoints (InternVideo, VidOSC/ChangeIt, CLIP, SigLIP) are fine.

## Why move away from labels
On the official README metric (30 GT videos), the best single constant state per (video, object) gets 0.899. Tracking change adds at most ~0.07.
Much of the remaining error is synonyms (e.g. "mixed" vs "beaten", "covered" vs "egg-covered").
Zero-shot InternVideo frame–text similarity was near noise for picking MOSCATO words.

## Step 1: discrete attributes (earlier, superseded)
- Files: `pipeline/attributes/{attributes,run}.py`, `pipeline/attributes/state_attributes/CMU.json`, `AttributeHeads`/`adapt_attributes` in `pipeline/vidosc.py`, plus a "proposal" section in `pipeline/README.md`.
- Groups the 80 MOSCATO words into 8 attributes (form, texture, heat, …). The map is built from MOSCATO vocabulary, so it can only be used for evaluation.

## Step 2: sliders (current), in `experiments/sliders/`
Each object gets a few continuous sliders (0..1): fullness, dirt, mixed, done, cut, heat, brown, open.
The three rules from Ryan's note: some sliders only go one way, actions move specific sliders, and labels act as occasional checkpoints.

**Best approach: LLM script + video timing** (`script_align.py`)
1. LLM (so far, me) picks sliders per object (`object_sliders.json`) and writes each slider's ordered levels from recipe + object (`scripts.json`).
   Examples: Eggs/eggs mixed [0,1]; Eggs/bowl fullness [0,1,0].
2. Evidence per frame = InternVideo similarity to the high prompts minus the low prompts. Prompts are generic plus recipe-aware ones (`recipe_anchors.json`).
   The evidence updates only on frames where the object is visible (`bench.hold2`).
3. Dynamic programming fits the script's segments to the evidence, so the video only decides *when* each step happens.

**Evaluation** (my proposal, unconfirmed)
- `sl.py: EVAL` maps MOSCATO words to slider levels, for scoring only.
- Score = per-frame range hit, (pred ≥ .5) == (GT ≥ .5), with balanced accuracy pooled per slider. The 95% CIs bootstrap over videos.

**Results** (30 GT videos; W&B [ryan/sliders-script-align](https://wandb.ai/chefOST/vidosc-experiments/runs/i1g9har7))

| slider | script + video | script, even split (no video) | gain 95% CI |
|---|---|---|---|
| mixed | 0.917 | 0.821 | +0.05 .. +0.14 |
| cut | 0.930 | 0.874 | +0.03 .. +0.12 |
| heat (frame acc) | 0.903 | 0.802 | +0.00 .. +0.19 |
| done | 0.973 | 0.957 | +0.00 .. +0.05 |
| fullness | 0.603 | 0.587 | n.s. |
| dirt | 0.650 | 0.754 | worse |

- Control: aligning the script to evidence from another video loses the gain on every slider except dirt.
- No script: the raw evidence thresholded is near 0.5–0.7, because the readouts are uncalibrated. The script does the calibration.

**What else was learned**
- Rising within the video is a trivial target for one-way sliders: a no-video time ramp scores ~1.
- Action cues ("someone whisking the eggs") place the egg-mixing change within ~4 s in Eggs videos but fire early in Brownie videos.
  On top of the script they add little.
- Visibility gating (update only when the object is seen) is the most useful single rule.
- Encoders inside script alignment: InternVideo ≥ SigLIP SO400M > CLIP L/14.
  Without a script, SigLIP's absolute readouts are better (e.g. done 0.91 vs 0.72).
- Fullness and dirt readouts stay weak with every encoder tried.

## Caveats
- The LLM outputs (slider choices, scripts, anchors) are **not blind**: I wrote them after seeing MOSCATO labels. The "done" score comes mostly from the script.
- Sample sizes are small: 11–30 videos per slider. `brown` and `open` have no GT mapping, so they are predicted but not scored.

## Open questions for Ryan
1. Is the word → slider-level map (`sl.EVAL`) acceptable for scoring?
2. Main metric: balanced range-hit accuracy, or change-timing error in seconds?

## Next steps
1. Blind run: a fresh LLM call, with recipe + object only, writes the sliders, scripts and anchors. Rerun `run_wandb.py`.
2. Better fullness and dirt evidence, e.g. object crops via a detector, or SigLIP readouts for those sliders.
3. Let the LLM output multi-step scripts with action cues (e.g. "fill when pouring"), and use the cues as boundary priors.
4. If adopted, move from `experiments/` into `pipeline/` with a README entry marked as a proposal.

## How to run (from `vidosc/`)
```
.venv/bin/python experiments/sliders/script_align.py - both     # main table (enc, anchors[, fuse-enc])
.venv/bin/python experiments/sliders/boot.py                    # bootstrap CIs
.venv/bin/python experiments/sliders/run_wandb.py --name ryan/<name>
.venv/bin/python experiments/sliders/encode_frames.py ViT-SO400M-14-SigLIP-384/webli   # frame features
```

**Data**
- Features: `data/cache/emb/{internvideo-mm-l14, clip-ViT-L-14-openai, clip-ViT-SO400M-14-SigLIP-384-webli}/{vid}_fps1.npy`
- Frames: `data/frames/{vid}_fps1/`
- Local results: `runs/ryan__sliders-script-align/`

**Environment change:** I added `transformers` to `.venv`, which the SigLIP tokenizer needs.
