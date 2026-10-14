# VidOSC + Claude on MOSCATO

Runs this pipeline against the [MOSCATO](https://github.com/eelhami/iccv25_moscato)
benchmark (ICCV 2025) and scores it
with the **same evaluator** the TubeletGraph × MOSCATO experiment uses, so the
three methods stay comparable:

```
TubeletGraph   vs.   VidOSC   vs.   VidOSC + Claude
```

MOSCATO annotates the state of every object in a procedural cooking video, at
every raw frame. This harness takes one video, one object, and one clip; runs it
through VidOSC and Claude; projects the result onto MOSCATO's state vocabulary;
and scores it on per-frame accuracy, micro-precision, micro-F1, and F1-max.

ChefOST uses [frame-averaged state precision and recall (detail)](https://github.com/chefOST/vidosc/blob/main/README.md#metrics). This branch's evaluator still reports legacy frame-hit accuracy and pooled (micro) scores; it does not yet implement those frame-averaged metrics. The metric descriptions and results below refer to the existing evaluator.

---

## Contents

1. [Status](#status) — what is verified and what is not
2. [Setup](#setup)
3. [Running it](#running-it)
4. [Configured experiments](#configured-experiments)
5. [How the temporal alignment works](#how-the-temporal-alignment-works)
6. [What is reused from TubeletGraph](#what-is-reused-from-tubeletgraph)
7. [Outputs](#outputs)
8. [Experimental integrity](#experimental-integrity)
9. [Findings so far](#findings-so-far)
10. [Known caveats](#known-caveats)
11. [Choosing other videos](#choosing-other-videos)

---

## Status

| Stage | State |
|---|---|
| Frame alignment | **Verified against the real S29 video** — see [Findings](#findings-so-far) |
| Full chain, dry run | End to end on the real video, no API calls |
| VidOSC inference | Runs. `--decode ordered` fixes the fragmentation; see [Decoding](#decoding-argmax-vs-ordered) |
| Claude descriptions | Ran once under the old prompt and produced hallucinated captions. Prompt reworked, **not yet re-run** |
| Vocabulary mapping | Crashed on an SDK incompatibility; fixed, **not yet run live** |
| Evaluation | Runs, but no meaningful numbers yet — no complete live run has happened |

**No end-to-end live run has completed.** The plumbing is verified; the science
is not. Don't quote a number out of this harness until a full `predict` →
`evaluate` cycle has run and its timeline has been eyeballed.

---

## Setup

### 1. Data

```bash
scripts/setup_moscato_data.sh          # MOSCATO_SRC defaults to ~/Downloads
```

Creates and populates:

```
data/moscato/
├── videos/       <video_id>.avi                       <- manual download
├── annotations/  gt_annotations_cmu.json              <- MOSCATO release
├── vocabulary/   state_dict.json, files_dict.json     <- MOSCATO release
├── metadata/
└── work/         decoded frames, clip MP4, features     (generated)
```

All of it is gitignored, as are `runs/` and `*.avi`.

**MOSCATO ships annotations, not video.** Its `files_dict.json` only records
where the authors kept each file. Video comes from the upstream datasets:

| Dataset | Source | Notes |
|---|---|---|
| CMU, EGTEA | [EgoProceL](https://sid2697.github.io/egoprocel/) → [video Drive folder](https://drive.google.com/drive/folders/17u7ReqPOJ29lbVZT8PDEd5dN17VBqS5Q) | Easiest. That Drive contains both an annotation tree and a video tree that reuse the same folder names — make sure you're in the video one |
| EgoPER | [EgoPER page](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm) | Access path not resolvable from the docs; treat as unresolved |

CMU videos arrive as a per-subject zip holding **six camera angles**. Extract
only the one MOSCATO annotates — the camera ID is part of the video ID
(`S29_Salad_`**`7150991`**`-1`). Any other angle silently produces garbage.

MOSCATO's pre-extracted `.npy` features are deliberately not fetched; VidOSC
extracts its own. Note that `files_dict[...]["feature_len"]` is MOSCATO's
feature stride, **not** a frame count.

### 2. Dependencies

On top of the base `vidosc_mac` environment:

```bash
pip install anthropic matplotlib scikit-learn pytest
```

### 3. Checkpoint

`--ckpt` is required and has no default — see
[caveats](#known-caveats) on checkpoint/domain fit.

---

## Running it

```bash
export ANTHROPIC_API_KEY=...
CFG=configs/moscato/s29_salad_7150991-1.json

python -m moscato.run_experiment inspect  --config $CFG   # alignment gate
python -m moscato.run_experiment prepare  --config $CFG   # decode + cut the clip
python -m moscato.run_experiment predict  --config $CFG \
    --ckpt ../VidOSC/checkpoints/chopping.ckpt --run-name baseline
python -m moscato.run_experiment evaluate --config $CFG --run-name baseline
```

`all` runs all four under one shared run name. Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--dry-run` | off | Exercise the whole chain with zero API calls |
| `--decode {argmax,ordered}` | `argmax` | Temporal decoding — see [Decoding](#decoding-argmax-vs-ordered) |
| `--prompt-mode {neutral,stated_state}` | `neutral` | Whether Claude is told the VidOSC label — see [Findings](#findings-so-far) |
| `--sample-rate {per_segment,per_second}` | `per_segment` | One Claude call per segment, or one per second |
| `--force` | off | Overwrite an existing run directory |
| `--no-wandb` | off | Skip W&B logging (**on by default**) — see [Weights & Biases](#weights--biases) |
| `--wandb-project` / `--wandb-entity` | `vidosc-experiments` / `chefOST` | W&B destination; use `--wandb-project "vidosc - full evals"` for full evaluations |

`prepare` is idempotent — it reuses decoded frames, the clip MP4, and extracted
features. Only `predict` costs money: 2 Claude calls per segment.

For a repeatable local invocation, keep a `*.local.sh` wrapper — that pattern is
gitignored, so it can safely hold your key.

---

## Weights & Biases

Logging is **on by default**, to `chefOST` / `vidosc-experiments`. Full
evaluations go to `vidosc - full evals` (exact name, spaces included) via
`--wandb-project "vidosc - full evals"`. Pass `--no-wandb` to skip logging.
Credentials come from `wandb login` (`~/.netrc`) or `WANDB_API_KEY` — either
works. Runs logged before this migration remain in `chefOST` / `VidOSC + MOSCATO`.

Each stage logs its own run, so a run that fails before scoring still leaves a
trace:

| Stage | Run name | `job_type` | Contents |
|---|---|---|---|
| `predict` | `<run-name>-predict` | `prediction` | Decode mode and detection indices, per-second table (decoded label **and** raw argmax side by side), segment table with the representative frame and Claude's description, timeline image, and the prediction artifacts |
| `evaluate` | `vidosc-claude-<run-name>` | `evaluation` | `word_accuracy/*` metrics, F1 threshold curve, per-frame table, scored artifacts |

Both carry the same config axes and the same `decode:` / `prompt:` tags, so an
argmax-vs-ordered comparison can be grouped on one axis across both job types:

```
decode_mode  prompt_mode  num_segments  checkpoint  video_id  clip bounds  fps ...
```

The `word_accuracy/*` namespace is unchanged from the TubeletGraph port, so runs
in this project are directly comparable with the `tubeletgraph` project.

Ground truth is not available during `predict` and none is logged there; metrics
appear only in the evaluation run.


## Configured experiments

| Config | Video | Object | Transition | Clip | Notes |
|---|---|---|---|---|---|
| `s29_salad_7150991-1.json` | `S29_Salad_7150991-1` | `vegetables` | `separated` → `chopped` | `[1469, 4679)` · 107 s | **Start here.** Cleanest transition in the benchmark; 0.8 % unannotated |
| `s12_sandwich_7150991-2470.json` | `S12_Sandwich_7150991-2470` | `bread slices` | `on dish` → `peanut butter topped` → … | `[2824, 6714)` · 130 s | The original TubeletGraph clip. Sandwich assembly — no checkpoint matches it |

Both are CMU at 30 fps and share `gt_annotations_cmu.json` and
`state_dict.json`. Adding another CMU video is a new JSON file, not new code —
see [Choosing other videos](#choosing-other-videos).

---

## How the temporal alignment works

MOSCATO annotates every raw frame; VidOSC predicts once per second. Reconciling
them is the part most likely to be silently wrong, so it is concentrated in
`moscato/frame_mapping.py` and asserted rather than assumed.

```
vidosc_timestep t ─┬─ timestamp    = t seconds (clip-relative)
                   ├─ local_frame  ∈ [t·fps, (t+1)·fps)
                   └─ global_frame = clip_start_global + local_frame
```

Two rules, and nothing else re-times a frame anywhere in the package:

```
global_frame = clip_start_global + local_frame
timestep(local_frame) = min(local_frame // fps, num_timesteps - 1)
```

The clamp handles the one awkward fact: a clip is rarely a whole number of
seconds, so the **final timestep absorbs the leftover tail**. That's a forward
fill — no interpolation and no smoothing anywhere. Every run records the rule in
its manifest under `temporal_alignment`.

Three things make the mapping trustworthy rather than merely plausible:

- **The clip MP4 is built from the extracted local JPEGs**, not by seeking into
  the source video, so clip frame *i* is local frame *i* by construction. The
  encoder's frame count is verified before the file is accepted.
- **`build_mapping` proves it is a partition** — spans must be gapless, must end
  exactly at `num_local_frames`, and every frame is round-tripped through the
  forward rule.
- **`segments_to_events` cross-checks two independent paths** — the event
  densification (what the evaluator uses) must agree frame-by-frame with
  `timestep_for_local_frame` (what the CSV uses), or it raises.

`frame_mapping.json` in every run persists the full timestep table.

---

## What is reused from TubeletGraph

From chefOST-ml, branch `feat/tubeletgraph-moscato-cmu-s12`. Each module names
its source in its docstring.

| This repo | Ported from | Change |
|---|---|---|
| `moscato/evaluate.py` | `scripts/evaluate.py` | **Verbatim**; VidOSC additions appended below the ported section, touching no metric |
| `moscato/inspect_video.py` | `scripts/inspect_video.py` | Repo-relative paths, `-count_frames` fallback for AVI, richer report |
| `moscato/prepare_clip.py` | `scripts/prepare_frames.py` | Same decode + hardlink + assertions, plus the clip MP4 |
| `moscato/map_states.py` | `scripts/map_event_states.py` | OpenAI → Anthropic; strategy, alias folding, retry and provenance preserved |
| `moscato/log_wandb.py` | `scripts/log_wandb.py` | Tags only |

**The trick that keeps evaluation identical.** Rather than writing a second
evaluator for VidOSC's per-second segments, `moscato/segments_to_events.py`
expresses them in the *same event schema* TubeletGraph emitted:

```
segment 0           -> initial_state
every later segment -> transition at local frame start_second × fps
```

The ported `densify` / `evaluate_events` then runs on VidOSC output unmodified.
Same evaluation unit (one raw MOSCATO frame), same multi-label rule (correct if
present in the GT list), same empty-`[]` policy (skip), same vocabulary handling
(`s2i` + `all2one`), same F1-max sweep over thresholds 0.1–0.9.

### Files in this repo that were modified

None of the original pipeline. `run_inference.run_inference`,
`extract_features.{extract_clip_features, extract_frames_1fps}` and
`describe_states.{group_segments, ClaudeBackend}` are imported as-is;
`NeutralClaudeBackend` subclasses `ClaudeBackend` and overrides only its prompt.
Only `.gitignore` and two doc links changed.

---

## Outputs

```
runs/<video_id>/<run-name>/
├── manifest.json                       config, checkpoint, feature + prompt provenance, integrity flags
├── frame_mapping.json                  the full timestep <-> frame table
├── vidosc_predictions.json             per-second states with per-class probabilities
├── vidosc_segments.json                contiguous segments + representative frames
├── claude_raw.json                     prompts and raw responses, unedited
├── moscato_predictions.json            events in TubeletGraph's schema
├── per_frame_predictions.csv           one row per raw frame, NO ground truth
├── per_frame_predictions_scored.csv    the same rows + GT, written by evaluate
├── evaluation.json                     metrics, policy, artifact digests
└── visualizations/
    ├── timeline.png                    VIDOSC / CLAUDE / MOSCATO-mapped lanes
    ├── timeline_with_gt.png            + the ground-truth lane
    └── representative_frames.jpg       the frames Claude actually saw
```

`per_frame_predictions_scored.csv` answers the traceability question on one line
per frame:

```
local_frame, global_moscato_frame, timestamp, vidosc_timestep, vidosc_state,
claude_description, mapped_moscato_state, ground_truth_state, evaluated, correct
```

---

## Experimental integrity

- **Ground truth is loaded in exactly two places**: `inspect_video` (which
  produces no predictions, only the alignment gate) and `evaluate`. Every other
  stage receives `config.inference_config(cfg)`, from which `annotations_path`
  has been removed, and calls `assert_no_ground_truth`, which raises both on a
  GT path *and* on a config that never went through that filter.
- **Claude never sees a label.** The description stage gets the frame and the
  object name. The mapping stage gets the frame, the object name, Claude's own
  description, the VidOSC temporal label, and the allowed vocabulary — never the
  GT state, never future states, never the annotation file. `map_states.py` does
  not import an annotation loader at all.
- **Raw predictions are written before evaluation**, and their sha256 is
  recorded in `evaluation.json` under `evaluated_artifacts`. The unscored CSV is
  never rewritten.
- **The evaluator refuses** predictions not marked
  `ground_truth_used_for_inference: false`.
- **Reproducibility**: the manifest records source-video sha256, checkpoint
  sha256, both repo commits, feature provenance, model IDs, prompt mode, and a
  sha256 of the allowed state vocabulary.

Run `pytest tests/test_moscato.py` — 25 tests covering the alignment partition,
evaluator parity, the ground-truth firewall, and prompt leakage.

---

## Findings so far

All measured on `S29_Salad_7150991-1`.

### Alignment is correct — confirmed three ways

The evaluator reported `skipped_empty_gt_frames: 1100`. Counting empty-`[]`
frames in `[1469, 4679)` straight from the raw annotations also gives 1100. Two
independent paths agreeing on that number means the frame indexing is right.
Spot checks land exactly on annotated transitions, and the frame at local 1439 —
where GT flips to `chopped` — shows a carrot under a knife.

EgoProceL's own key-step CSV corroborates from a third source: it marks *"cut
carrot"* at 95.999 s where MOSCATO flips to `chopped` at 96.9 s, and *"peel the
cucumber"* at 121.807 s where MOSCATO flips to `peeled` at 121.6 s.

The `inspect` gate also caught a real error: S29 is **1024×768**, not the
800×600 of S12. That is what the gate is for.

### VidOSC only predicts 2 of its 4 states here

| state | argmax wins | mean p | max p | ever top-2 |
|---|---:|---:|---:|---:|
| background | 0 | 0.028 | 0.050 | 0 |
| initial_state | 34 | 0.389 | 0.732 | 107 |
| transitioning | 0 | 0.053 | 0.127 | 0 |
| end_state | 73 | 0.529 | 0.816 | 107 |

`transitioning` and `background` never reach top-2 at any timestep, so argmax
can never select them. The transition is the shortest and hardest state to
localize even upstream; the feature approximations below make it worse.

### The ordering constraint is now wired in — see [Decoding](#decoding-argmax-vs-ordered)

Previously `run_inference.py`'s plain `argmax` was the only decode path, and the
ordering-constrained DP in `patches/` was never called. It is now selectable via
`--decode ordered`. The comparison is below.

### The description prompt was leaking the label

The upstream captioning prompt names the VidOSC state:

> "This frame is from a cooking video showing a vegetables during the **'end
> state' stage of a state change**…"

On a raw salad that produced *"Frozen, solid, icy crystalline coating"*, *"Diced
red peppers, partially caramelized and softened"*, and *"Charred, softened, and
collapsed"* — nothing in S29 is cooked. This is the failure mode `README.md`
already documents: the VLM will not push back on a stated label.

`--prompt-mode neutral` (now the default) withholds it and asks only what is
visible, offering *"not visible"* as an allowed answer. `stated_state`
reproduces upstream's prompt byte-for-byte for comparison. The mode and the
verbatim prompt text are recorded per response in `claude_raw.json`.
**The neutral prompt has not yet been run live.**

---

## Decoding: argmax vs. ordered

### Why independent argmax fragments the timeline

`run_inference.py` decodes each second independently: `probs.argmax(-1)`.
Nothing ties timestep *t* to timestep *t−1*, so whenever two classes are close
in probability, ordinary noise flips the label back and forth. On S29 the model
is genuinely uncertain in places — `initial_state` and `end_state` sit at 0.45–0.55
across several stretches — and the result is a timeline that changes state 12
times, including `end_state → initial_state` transitions that are physically
impossible: chopped vegetables do not become unchopped.

That fragmentation is not merely cosmetic. Every segment boundary spawns a Claude
call on a mid-flicker representative frame, and every segment becomes a
transition in the MOSCATO event schema, so a decode artifact propagates all the
way into the scored predictions.

### What ordered decoding does

VidOSC specifies a temporal-ordering constraint at decode time. It is implemented
in `patches/0001-macos-and-multi-transition-states.patch` as
`task.ordered_state_indices`, an O(T·S) dynamic program, and
`moscato/decode.py` **calls that function directly** — with the same argument
slice upstream uses in `FrameCls.infer_state_idx`:

```python
pred_idx = ordered_state_indices(prob[:, 1:])
```

Upstream leaves it off by default (`self.infer_ordering = False`), which is why
the argmax path was faithful rather than buggy.

The DP returns **three frame indices, one per state** — VidOSC's own protocol is
to detect a representative frame per state, not to label a timeline. MOSCATO
scores every frame, so `moscato/decode.py` adds one documented densification
step:

> state *k* holds from detection index *k* until detection index *k+1*; the
> first state extends back to t=0 and the last runs to the end.

The alternative — treating `idx[0]` as an onset and calling everything before it
`background` — was rejected on evidence: background peaks at p=0.050 across the
whole clip while `initial_state` wins argmax outright over the same span.

**Ordered mode never emits `background`.** That follows from the reused
algorithm, not from a choice made here: upstream slices the background column off
before the DP sees it. Use `argmax` where background genuinely needs predicting.

### Measured on S29_Salad_7150991-1 (`chopping.ckpt`, `vegetables`)

| | argmax | ordered |
|---|---:|---:|
| segments | **12** | **3** |
| `transitioning` ever selected | no | yes |
| monotonic (no backward flips) | no | yes |
| first `end_state` onset | t=20 s | t=51 s |
| main `end_state` onset | t=28 s | t=51 s |
| error vs. GT boundary (48.0 s) | **−20 s** | **+3 s** |

```
argmax   0-19 initial | 20 end | 21-27 initial | 28-79 end | 80-81 initial |
         82-86 end | 87 initial | 88 end | 89 initial | 90-96 end |
         97-99 initial | 100-106 end
ordered  0-11 initial | 12-50 transitioning | 51-106 end
```

### First live results — and a baseline that reframes them

S29_Salad, `chopping.ckpt`, `vegetables`, neutral prompt, Claude Haiku 4.5:

| | argmax | ordered |
|---|---:|---:|
| segments | 12 | 3 |
| accuracy | **0.6224** | 0.5278 |
| micro-F1 | **0.5357** | 0.4543 |
| correct frames | 1981 / 3183 | 1680 / 3183 |

**Ordered decoding scored worse than argmax, despite the better boundary.** The
temporal fix did not translate into a benchmark gain.

Now the number that matters most:

| trivial baseline | accuracy |
|---|---:|
| predict `chopped` for every frame | **0.5564** |

Ground truth is `chopped` for 56 % of evaluated frames, so a constant predictor
beats ordered decoding and comes within 7 points of argmax. **Neither decode
mode has demonstrated it is doing something a majority-class guess would not.**
Argmax's apparent lead comes largely from labelling more of the timeline
`chopped` — it mapped 8 of its 12 segments to that state — rather than from
localising the transformation better.

Read the decode comparison as: ordered fixes a real defect in the *inference
path*, and the pipeline is not yet good enough downstream for that to show up in
the score.

### Model sweep: better models scored worse

| decode | describer | mapper | accuracy | micro-F1 |
|---|---|---|---:|---:|
| argmax | Haiku 4.5 | Haiku 4.5 | **0.6224** | 0.5357 |
| argmax | Sonnet 5 | Haiku 4.5 | 0.4910 | 0.4227 |
| argmax | Sonnet 5 | Sonnet 5 | 0.4533 | 0.3902 |
| ordered | Haiku 4.5 | Haiku 4.5 | 0.5278 | 0.4543 |
| ordered | Sonnet 5 | Haiku 4.5 | 0.5278 | 0.4543 |
| ordered | Sonnet 5 | Sonnet 5 | 0.5278 | 0.4543 |
| *constant `chopped`* | — | — | *0.5564* | — |

Sonnet 5's descriptions are unambiguously better. Haiku called a sliced cucumber
"two black olives"; Sonnet produced "a carrot is being sliced into small round
orange discs on a wooden cutting board, with a whole cucumber lying uncut
beside it" — accurate to the frame. **Accuracy still fell by 13 points.**

The per-state breakdown explains it. Score is almost entirely a function of how
many frames get mapped to `chopped`:

| run | frames mapped `chopped` | of those, correct |
|---|---:|---:|
| argmax / Haiku | 2343 | 1771 (76 %) |
| argmax / Sonnet | 1833 | 1261 (69 %) |
| ordered / either | 1680 | **1680 (100 %)** |

Every other predicted state — `cooked`, `clean`, `peeled`, `in use`, `covered`,
`inside`, `placed` — scores near zero. Haiku's vaguer captions pushed the mapper
toward `chopped` more often, and `chopped` is the majority class. Sonnet's
precise captions led the mapper to specific states that are *defensible
descriptions of the frame* but not MOSCATO's label for it.

Note the third row: ordered decoding's `chopped` span is **100 % precise**. The
boundary is right. It simply covers less of the timeline than argmax's
scattergun, and its other two segments never map correctly.

### The real ceiling is task definition, not model quality

MOSCATO labels `vegetables` as `separated` for the first 47 seconds. Sonnet
described that period accurately — "a whole carrot, uncut and unpeeled, lies on
a cutting board while a plastic-wrapped, sealed package of leafy greens is held
unopened". The mapper chose `clean`, then `covered`, then `inside`. None is
wrong about the *image*; none is MOSCATO's label.

`separated` is an annotation-scheme concept about where the vegetables are in
the recipe, not a visual property recoverable from one frame. No amount of VLM
capability closes that gap, and a stronger model widens it by being more
specific. Raising model quality is not the lever here; the candidates are richer
temporal context for the mapper, few-shot examples of the vocabulary's intent,
or accepting that some MOSCATO states are out of reach for a single-frame method.

### What this does and does not establish

It establishes that the **inference path is now correct**: the decoder produces a
temporally valid sequence, the boundary lands 3 seconds from ground truth rather
than 20, and segmentation, captioning, and event conversion all consume the
decoded labels.

It does **not** establish a benchmark improvement — the live runs above show the
opposite. A better boundary is a necessary condition for a better score, not a
sufficient one, and the bottleneck is downstream: Claude's descriptions.

Three description defects found by running it, in order of size:

1. **Captions were built from 224×224 thumbnails.** `extract_frames_1fps` scales
   to CLIP's input size, and reusing it for captioning fed Claude a squashed
   thumbnail of a 1024×768 frame while the *mapper* read the full-resolution
   file. Fixing it moved argmax from 0.5187 to 0.6224 accuracy. Both stages now
   read the same file; `tests/test_moscato.py` locks that in.
2. **One frame per segment is fragile, and worse with fewer segments.** The
   midpoint of the ordered run's 39-second middle segment landed on a
   motion-blurred frame; Claude correctly answered "not visible", which has no
   canonical MOSCATO state, and the mapper crashed. The describer now falls back
   to the 25 % and 75 % points before giving up.
3. **Claude still hallucinates on clear frames.** A cucumber being sliced was
   described as "two black olives and a red pepper". Haiku 4.5 at this frame
   size is the current ceiling; a stronger model is the obvious next lever.


## Known caveats

**No checkpoint matches sandwich assembly.** The S12 clip's ground truth runs
*on dish → peanut butter topped → jam-topped → assembled*. None of the released
checkpoints covers that. S29 (`chopping` → `chopped`) is a genuine match, which
is why it's the recommended starting point.

**Two stacked feature approximations.** `extract_features.py` produces CLIP
ViT-L/14 features, but the checkpoints were trained on InternVideo-MM-L14; and
the checkpoints expect 1536 dims (whole-frame ⊕ hand–object detector), of which
this repo supplies only the first 768 — the rest is zero-filled. Both are
recorded in `manifest.json → features` and the warning is printed, not
suppressed. Pass `--obj_feat` if you have real object-centric features.

**Four temporal states, dozens of MOSCATO states.** VidOSC can express at most
four phases; the mapper must project each onto one of 77 canonical CMU labels.
An upper bound on achievable accuracy exists before Claude is called.

**One frame represents a whole segment.** At `--sample-rate per_segment` the
segment's middle second is what Claude and the mapper see. For a 50-second
segment that's a strong assumption. `per_second` trades cost for resolution.

**The mapper still sees the VidOSC label** in its `RAW_VIDOSC_CONTEXT`. That is
deliberate — it's a genuine prediction, and the mapper sees the frame too — but
a wrong label can still influence the mapped state.

**Segments are never skipped.** Upstream `describe_states.py` defaults to
skipping `background`; MOSCATO scores every frame, so `skip_states` must stay
empty and `predict` raises if it isn't.

**The mapper is no longer strictly deterministic.** The TubeletGraph original
passed `temperature=0`; the Anthropic SDK removed that parameter in 1.x, and
current models reject sampling parameters outright. Steering now comes from the
system prompt. Every mapping decision is recorded in `mapping.raw_responses`, so
non-reproducibility is visible rather than silent.

---

## Choosing other videos

Adding a video is a new config JSON, not new code — provided it's CMU. A
separate analysis ranked all 88 MOSCATO ground-truth videos against the
available VidOSC checkpoints; the short version:

- **8 of 21 checkpoints have a defensible match**: chopping, crushing, slicing,
  whipping, peeling, frying, grating, rolling.
- **7 have none**: sauteing, grilling, mashing, squeezing, mincing, shredding,
  zesting. `juiced` exists in the CMU vocabulary but appears in zero test videos.
- The filter that matters is **food vs. tool** — `sauteing` looks like it has 52
  candidates, every one a bowl or pan turning `oily`.

Strong CMU candidates beyond S29, all reusing the same annotations and
vocabulary:

| Checkpoint | Video | Object | Transition |
|---|---|---|---|
| `whipping` | `S28_Eggs_7150991-1` | eggs | `cracked, in bowl` → `beaten` |
| `peeling` | `S41_Salad_7150991-349` | vegetables | `chopped, separated` → `peeled` |
| `chopping` | `S51_Salad_7150991-225` | vegetables | `relocated` → `chopped` |

Non-CMU videos need more than a config. Three assumptions break:
`inspect_video` asserts `video_path.stem == video_id`, which fails for two EGTEA
test videos whose files carry an `O` prefix; `prepare_clip.decode_all` decodes
every raw frame before cutting, ruinous for EGTEA's 30–42-minute videos; and
EgoPER has five separate vocabularies rather than one. Frame rates also differ —
30 fps for CMU, ~24 for EGTEA, ~15 for EgoPER — inferred from annotation-length ÷
`feature_len` ratios, never measured. Run `inspect` first; it will tell you.
