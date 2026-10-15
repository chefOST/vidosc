# Sliders (ChefOST experiment, zero-shot)

Objects described by continuous sliders (fullness, dirt, mixed, done, cut, heat, brown, open) instead of state labels.
No training. MOSCATO is used only in `sl.py: EVAL`, an eval-side map from GT words to slider levels (unconfirmed choice).

## Best approach: LLM script + video timing (`script_align.py`)
1. LLM, given recipe + object, picks sliders (`object_sliders.json`) and writes each slider's ordered levels over the
   video (`scripts.json`), e.g. Eggs/bowl fullness [0,1,0], Eggs/eggs mixed [0,1].
2. Video gives timing: per-frame evidence = InternVideo similarity to high vs low anchors (generic + recipe-aware,
   `recipe_anchors.json`), updated only on frames where the object is visible (`bench.hold2`), z-scored per track.
3. DP splits the track into the script's segments to best fit the evidence.

Range hit, balanced acc, 30 GT videos (bootstrap over videos, `boot.py`):

| slider | script + video | script, even split (no video) | 95% CI of gain |
|---|---|---|---|
| mixed | 0.917 | 0.821 | +0.05 .. +0.14 |
| cut | 0.930 | 0.874 | +0.03 .. +0.12 |
| heat (frame acc; no low frames) | 0.903 | 0.802 | +0.00 .. +0.19 |
| done | 0.973 | 0.957 | +0.00 .. +0.05 |
| fullness | 0.603 | 0.587 | -0.06 .. +0.09 |
| dirt | 0.650 | 0.754 | -0.22 .. -0.03 |

Control: evidence from a different video (same slider) loses the gain on all but dirt.
Caveat: the LLM (Claude) outputs are not blind; it had seen MOSCATO labels in the same session.

## Other scripts
- `ev.py`, `cp.py`, `hmm.py`: first probes (within-video concordance is too easy: a time ramp wins).
- `bench.py [enc] [generic|recipe|both]`: level concordance per object and step timing.
- `fuse.py`: InternVideo + CLIP fusion. `encode_frames.py`: CLIP/SigLIP frame features (1 fps).

Run from `vidosc/`: `.venv/bin/python experiments/sliders/script_align.py - both`

W&B: `run_wandb.py --name ryan/...` logs per-slider metrics, bootstrap CIs, a per-track table, timelines and the LLM inputs. First run: [ryan/sliders-script-align](https://wandb.ai/chefOST/vidosc-experiments/runs/i1g9har7).
