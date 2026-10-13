# VIDOSC-pipeline

Video object state-change (OSC) tooling built on top of
[facebookresearch/VidOSC](https://github.com/facebookresearch/VidOSC): feature
extraction, standalone inference, VLM-based state description, and macOS /
Apple Silicon support.

**This repo contains only our own code.** Upstream VidOSC is not vendored or
submoduled here — it is a separate clone that these scripts locate by path.

> New here? [README.md](README.md) explains what object state changes are, what
> VidOSC does, and what this pipeline adds — with worked examples. This file
> covers installation and usage.

---

## Contents

1. [What each file does](#what-each-file-does)
2. [How the pieces fit together](#how-the-pieces-fit-together)
3. [Setup](#setup) — clone, environment, checkpoints, data
4. [Running the scripts](#running-the-scripts)
5. [Things you will need to change](#things-you-will-need-to-change)
6. [Troubleshooting](#troubleshooting)

---

## What each file does

| Path | Purpose | Needs upstream clone? |
|---|---|---|
| [extract_features.py](extract_features.py) | Decodes a video at 1 fps and encodes each frame with CLIP ViT-L/14 → `(n_seconds, 768)` tensor | No |
| [run_inference.py](run_inference.py) | Loads a VidOSC checkpoint, runs a forward pass on a feature file, prints a per-second state timeline, optionally logs to W&B | **Yes** (`model.FeatTimeTransformer`) |
| [describe_states.py](describe_states.py) | Runs inference, groups the timeline into segments, captions one frame per segment with a VLM | Yes (indirectly, via `run_inference`) |
| [vidosc_path.py](vidosc_path.py) | Locates the upstream checkout and puts it on `sys.path` | — |
| [moscato/](moscato/) | Benchmarks the pipeline on MOSCATO and scores it with the TubeletGraph evaluator — see [MOSCATO.md](MOSCATO.md) | Yes (via `run_inference`) |
| [configs/moscato/](configs/moscato/) | One JSON per benchmarked video: clip bounds, target object, fps, paths | — |
| [scripts/setup_moscato_data.sh](scripts/setup_moscato_data.sh) | Places the MOSCATO annotations/vocabulary and tells you where to get the video | — |
| [tests/](tests/) | `pytest tests/test_moscato.py` — alignment, evaluator parity, ground-truth firewall | — |
| [environment_mac.yml](environment_mac.yml) | macOS / Apple Silicon conda env (replaces upstream's Linux-only `environment.yml`) | — |
| [patches/](patches/) | Local modifications to upstream VidOSC, kept as a diff so no upstream code lands in this repo | — |

Only `run_inference.py` imports upstream code, and only one symbol
(`model.FeatTimeTransformer`). `extract_features.py` is fully standalone —
you can run it without the clone at all.

### How `vidosc_path.py` finds the clone

It resolves the checkout in this order:

1. **`$VIDOSC_ROOT`**, if set — use this if your clone lives elsewhere:
   ```bash
   export VIDOSC_ROOT=/path/to/VidOSC
   ```
2. **`../VidOSC`** — a sibling of this repo (the layout [Setup](#setup) creates).

It then verifies the directory actually contains `model.py`, `task.py`, and
`dataset.py` before prepending it to `sys.path`. Any script needing an upstream
module does:

```python
from vidosc_path import ensure_on_path
ensure_on_path()
from model import FeatTimeTransformer
```

Check the resolved path at any time — this is the fastest way to confirm your
setup is correct:

```bash
python vidosc_path.py     # prints the checkout it would use, or explains what's missing
```

---

## How the pieces fit together

```
  your_video.mp4
        │
        ├─────────────────────────────┐
        │                             │
        ▼  extract_features.py        │  (raw frames, re-decoded at 1 fps)
  data/feats/<id>.pth.tar             │
  (n_seconds, 768) float tensor       │
        │                             │
        ▼  run_inference.py           │
  per-second state predictions ───────┤
  (n_seconds,) ints in 0..3           │
        │                             ▼
        └──────────────────►  describe_states.py
                                      │
                                      ▼
                             one caption per state segment
```

**States:** `0 = background`, `1 = initial_state`, `2 = transitioning`,
`3 = end_state`.

The model is a small transformer over the per-second feature sequence. It sees
the whole clip at once, so predictions at second *t* depend on the entire video,
not just that frame.

---

## Setup

### 1. Clone upstream VidOSC as a sibling directory

Pinned commit: **`59575773c97878ea30a0228f28ab00a3ee2f1ea2`**
(`code for ChangeIt and ChangeIt (open-world)`, 2024-09-09)

From the parent directory of this repo:

```bash
git clone https://github.com/facebookresearch/VidOSC.git VidOSC
git -C VidOSC checkout 59575773c97878ea30a0228f28ab00a3ee2f1ea2
git -C VidOSC apply "$PWD/VIDOSC-pipeline/patches/"*.patch
```

Resulting layout:

```
WAT.AI/                 # Parent Folder Name
├── VidOSC/             # upstream clone, untracked by this repo
└── VIDOSC-pipeline/    # this repo
```

The patches are what make upstream run on macOS at all (the causal-ordering
constraint is a Linux-only compiled CUDA extension) and what generalize it to
N transitioning states. See [patches/README.md](patches/README.md) for the full
rationale, plus how to check, reverse, and regenerate them.

> Don't clone VidOSC *inside* this repo — `vidosc_path.py` looks for a sibling.
> (`.gitignore` does ignore `/VidOSC/` as a safety net, but you'd still need to
> set `$VIDOSC_ROOT` for it to resolve.)

### 2. Create the conda environment

```bash
conda env create -f environment_mac.yml
conda activate vidosc_mac
brew install ffmpeg
```

`ffmpeg` is a real system binary, not just the `ffmpeg-python` wrapper —
`extract_features.py` shells out to it for decoding.

**Optional extras**, deliberately kept out of the env file so the base install
stays small. Each is imported lazily and raises a clear message if missing:

| Install | Needed for |
|---|---|
| `pip install wandb` | `--wandb` logging in either script |
| `pip install matplotlib` | the timeline image in W&B runs (silently skipped if absent) |
| `pip install anthropic` | `describe_states.py --vlm_backend claude` |
| `pip install openai` | `describe_states.py --vlm_backend openai` |
| `pip install transformers accelerate` | `describe_states.py --vlm_backend llava` |

### 3. Download a checkpoint

Checkpoints (~127 MB each) are not in git. Download from the
[VidOSC Google Drive](https://drive.google.com/drive/folders/1tChqwGmfmBWUq0KGFaru2wPB_4Q2hiYP)
and place under `checkpoints/` (gitignored):

```
checkpoints/chopping.ckpt
```

`run_inference.py` reads `input_dim` and `vocab_size` straight off the
checkpoint weights, so you never pass architecture flags — but see the
[object-centric features](#object-centric-features-important) note below,
because it determines what features the checkpoint actually expects.

### 4. Get features

Either extract your own (see [extract_features.py](#extract_featurespy)) or
download the pre-extracted InternVideo features from the same Google Drive.
The pre-extracted set is organized per OSC category:

```
data/feats_handobj/<osc>/<video_id>_st<start>_dur<duration>_obj.pth.tar
data/feats_handobj/<osc>/<video_id>_st<start>_dur<duration>_obj.npy
data/eval_clips/<osc>/<video_id>_st<start>_dur<duration>.mp4
```

where `<osc>` is `<verb>_<object>`, e.g. `browning_apple`. That naming matters:
`describe_states.py` derives the object name for its prompt from the parent
directory, and `run_inference.py` finds the `.npy` index by swapping the
`.pth.tar` suffix.

### Object-centric features (important)

All three released checkpoints (`browning`, `chopping`, `rolling`) were trained
with `det=1` — they expect **1536-dim** input: 768 dims of whole-frame features
concatenated with 768 dims of object-centric (hand–object detector) features.

`extract_features.py` produces only the 768-dim whole-frame half. When you feed
that to a 1536-dim checkpoint, `run_inference.py` zero-pads the missing half and
prints:

```
Warning: checkpoint expects object-centric features (det=1) but none were
provided; padding with zeros (results may be degraded).
```

This runs and produces plausible output, but it is **not** a faithful
reproduction of the paper's numbers. For that, use the pre-extracted
`feats_handobj` features and pass both halves:

```bash
python run_inference.py \
    --feat  data/feats/pof9jFBhHVA.pth.tar \
    --obj_feat data/feats_handobj/browning_apple/pof9jFBhHVA_st13.0_dur40.0_obj.pth.tar \
    --ckpt  checkpoints/browning.ckpt
```

The accompanying `_obj.npy` file (same path, `.pth.tar` → `.npy`) holds the
second-indices the object features correspond to — frames where the detector
found nothing stay zero. It's picked up automatically if present.

---

## Running the scripts

### `extract_features.py`

Decodes at 1 fps, resizes to 224×224, encodes each frame with CLIP ViT-L/14,
L2-normalizes, and stacks.

```bash
python extract_features.py \
    --video /path/to/your_video.mp4 \
    --video_id your_video_id
```

| Flag | Default | Meaning |
|---|---|---|
| `--video` | *required* | Input `.mp4` |
| `--video_id` | *required* | Used as the output filename stem |
| `--feat_dir` | `./data/feats` | Output directory (created if needed) |

Writes `data/feats/your_video_id.pth.tar`. Device is auto-selected: MPS → CUDA →
CPU.

> **Backbone caveat:** VidOSC was trained on InternVideo-MM-L14 features. This
> script uses CLIP ViT-L/14 (same architecture family, same 768-dim output) as a
> stand-in, so results are approximate. This is a *separate* approximation from
> the object-centric one above — using this script alone, you're off on both axes.

### `run_inference.py`

```bash
python run_inference.py \
    --feat data/feats/your_video_id.pth.tar \
    --ckpt checkpoints/chopping.ckpt
```

Prints a per-second timeline, then a summary:

```
t=  0s  [████████████████████]  initial_state  (0.97)
t= 13s  [▒▒▒▒▒▒▒▒▒▒▒▒▒▒      ]  transitioning  (0.71)
t= 46s  [░░░░░░░░░░░░░░░░░░░░]  end_state      (0.94)
```

| Flag | Default | Meaning |
|---|---|---|
| `--feat` | *required* | `.pth.tar` feature file |
| `--ckpt` | *required* | VidOSC `.ckpt` |
| `--obj_feat` | `None` | Object-centric features for `det=1` checkpoints |
| `--wandb` | **on if `WANDB_API_KEY` is set** | Log to Weights & Biases |
| `--no-wandb` | off | Force W&B off, overriding the above |
| `--wandb-entity` | `chefOST` | Team W&B entity |
| `--wandb-project` | `vidosc-experiments` | Development project; use `vidosc - full evals` for intended full evaluations |
| `--wandb-run-name` | auto | Defaults to `vidosc-<ckpt>-<video_id>` |

It's also importable, which is how `describe_states.py` uses it:

```python
from run_inference import run_inference
predicted, probs = run_inference(feat_path, ckpt_path, use_wandb=False)
```

With `--wandb` it logs a per-second predictions table, a state-duration bar
chart, a color-coded timeline image, and `seconds/<state>` + `pct/<state>`
summary metrics.

### `describe_states.py`

Runs inference, groups consecutive seconds sharing a state into segments, and
sends the middle frame of each segment to a VLM asking what the object looks
like at that moment.

```bash
python describe_states.py \
    --video data/eval_clips/browning_apple/pof9jFBhHVA_st13.0_dur40.0.mp4 \
    --feat data/feats/your_video_id.pth.tar \
    --ckpt checkpoints/browning.ckpt \
    --vlm_backend claude
```

`--video` must be the same clip `--feat` was extracted from. If the frame count
and prediction count disagree, both are truncated to the shorter one and you get
a warning.

| Flag | Default | Meaning |
|---|---|---|
| `--video` | *required* | Source clip, re-decoded at 1 fps for the frames |
| `--feat`, `--ckpt`, `--obj_feat` | | Same as `run_inference.py` |
| `--object` | auto | Object name for the prompt, e.g. `apple` |
| `--osc` | auto | OSC category, e.g. `browning_apple` |
| `--sample_rate` | `per_segment` | Or `per_second` — one call per second, much more expensive |
| `--skip_states` | `background` | Comma-separated states not to caption; pass `''` to caption everything |
| `--vlm_backend` | `claude` | `claude` \| `openai` \| `llava` |
| `--vlm_model` | per-backend | Overrides the default model |
| `--api_key` | env var | Else `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` |
| `--max_tokens` | `80` | Per description |
| `--dry_run` | off | Print segments without calling any VLM — **no extra deps needed** |

Default models: Claude `claude-haiku-4-5-20251001`, OpenAI `gpt-4o-mini`,
LLaVA `llava-hf/llava-v1.6-mistral-7b-hf` (local, downloads ~15 GB on first run).

**Object name resolution**, in priority order: `--object` → `--osc` →
the parent directory of `--video` (VidOSC's `<clip_dir>/<verb>_<object>/<file>.mp4`
convention) → the filename stem, with a warning. If your clips aren't laid out
that way, pass `--object` explicitly or you'll get a nonsense prompt.

**Start with `--dry_run`.** It shows exactly which segments and frames would be
sent, costs nothing, and needs no API key or extra packages.

---

## Things you will need to change

If you are not the original author, these are hardcoded and will not work for you
as-is:

- **W&B destination** — [run_inference.py:54](run_inference.py#L54) defaults to
  entity `chefOST`, project `vidosc-experiments` for development.
  For intended full evaluations pass `--wandb-project "vidosc - full evals"`.
  `describe_states.py` imports the same defaults. Keep existing historical runs
  unchanged and use a new run ID when changing projects.
- **W&B is on by default** whenever `WANDB_API_KEY` is in your environment, in
  *both* scripts. If you have that variable set for unrelated reasons, every run
  silently creates a run under the entity above. Pass `--no-wandb`, or unset it.
- **State names assume 4 classes.** `STATE_NAMES` covers indices 0–3. The
  multi-transition-state patch allows checkpoints with a larger vocab, but this
  repo's printing and captioning code would need matching entries — a checkpoint
  with `vocab_size > 4` raises `KeyError`. All three released checkpoints are
  `vocab_size=4`, so this only bites if you train your own.

### What's gitignored

`checkpoints/`, `*.ckpt`, `data/`, `feats/`, `*.pth.tar`, `wandb/`, `.env`, and
`/VidOSC/`. So a fresh clone has code only — you supply the model and the data.

---

## Troubleshooting

**`FileNotFoundError: VidOSC checkout not found at ...`**
The sibling clone is missing or in the wrong place. Run `python vidosc_path.py`
— the error message includes the exact clone commands for your paths. Set
`$VIDOSC_ROOT` if your clone lives elsewhere.

**`... does not look like a VidOSC checkout -- missing model.py, task.py`**
The directory exists but isn't upstream VidOSC (or the clone failed partway).

**`ModuleNotFoundError: No module named 'lookforthechange'`**
The patches aren't applied. That import is Linux-only; the patch wraps it in a
`try/except`. Re-run the `git apply` from step 1, or check with
`git -C ../VidOSC apply --check patches/*.patch`.

**`Feature dim 768 does not match checkpoint input_dim 1536`**
Raised only when the dims are incompatible in a way padding can't fix. The
ordinary 768→1536 case zero-pads with a warning instead — see
[object-centric features](#object-centric-features-important).

**`RuntimeError: Error(s) in loading state_dict`**
The checkpoint's transformer config differs from the hardcoded one in
`load_model` (4 heads, 3 layers, dim 512). Those match the released checkpoints;
a checkpoint you trained with different hyperparameters needs them edited.

**`ffmpeg` errors from `extract_features.py`**
`brew install ffmpeg` — the pip package `ffmpeg-python` is only a wrapper around
the system binary.

**Predictions look like noise / all one state**
Expected to some degree if you extracted features with `extract_features.py`:
you're stacking a CLIP-for-InternVideo substitution on top of zero-padded
object-centric features. Try a pre-extracted `feats_handobj` file with its
`--obj_feat` pair to see what the checkpoint does when fed what it was trained on.
