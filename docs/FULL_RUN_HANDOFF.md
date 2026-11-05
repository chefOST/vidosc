# Active-object full-video workflow

The maintained workflow extends Pritika's October 24 active-object runs to the
pinned R2 release. It produces one unnamed hand-associated stream per video;
that stream may switch objects over time. Inference receives no ground-truth
labels and computes no evaluation metrics.

## Setup

Run commands from this repository's root. The local environment uses Python 3.13
(setup checked with 3.13.7); the Modal image uses Python 3.11.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_checkpoints.py
```

The checkpoint script downloads and verifies the expected SHA256 for both files:

- `checkpoints/internvideo/InternVideo-MM-L-14.ckpt`
- `checkpoints/changeit/state_best.ckpt` (ChangeIt multitask state checkpoint)

InternVideo's current official download requires access to
[OpenGVLab/InternVideo1.0](https://huggingface.co/OpenGVLab/InternVideo1.0).
Accept its access conditions, run `.venv/bin/hf auth login` locally with that
account's read token, then rerun the checkpoint script. Never put tokens in Git or chat.

Weights are uploaded from the laptop to Modal. The Modal image supplies FFmpeg,
the hand/object detector, InternVideo at `3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3`,
and VidOSC at `59575773c97878ea30a0228f28ab00a3ee2f1ea2`. No local InternVideo
source checkout or feature extraction is needed. Image dependencies and detector
checkpoint verification are in [`scripts/modal_detector.py`](../scripts/modal_detector.py)
and [`scripts/modal_clip.py`](../scripts/modal_clip.py).
Plans select Modal image builder `2025.06`, supported by the pinned client, without
changing the workspace's default image builder.

CMU's existing mapping requires the matching vocabulary, separate from the model
inputs. Copy it from the sibling MOSCATO checkout:

```sh
mkdir -p data/moscato/vocabulary
cp ../MOSCATO/vocabulary/CMU/state_dict.json data/moscato/vocabulary/state_dict.json
```

Its expected SHA256 is
`0fbc692af4c5ff399a467804abb3e4d8af27fedb47256dc9e6072fb4f6b8aa05`.
No annotation files are required by this workflow. Configure
[your Modal, R2 and W&B accounts](account-setup.md) before inventory or execution.

## R2 inventory

[`configs/r2_release.json`](../configs/r2_release.json) is canonical for the bucket,
endpoint, release prefix, manifest and expected release hash. Read the release
and check its object listing using the existing local AWS profile:

```sh
AWS_PROFILE=chefost-r2 .venv/bin/python scripts/inventory_r2.py
```

Inspect `runs/batch-preparation/inventory.json` before planning. The pinned release
has 41 available videos: 29 CMU and 12 EGTEA. The report also records 46 unavailable
EgoPER videos, which are excluded from execution. Availability is not a successful
inference result or a benchmark result.

The worker mounts R2 read-only through the Modal Secret `chefost-r2`, then uses
temporary files inside its container for video decoding. Full videos are not
downloaded to the laptop. Do not substitute the historical CMU30 AVI inventory:
the R2 release has its own source hashes and sampling metadata.

## Pipeline

1. `scripts/prepare_full_video_plan.py` creates one Modal model stage per video
   and, for CMU only, a local mapping stage.
2. `scripts/modal_clip.py` samples eight frames per one-second interval, shortening
   the final interval. It detects hand-associated objects on the fifth sampled
   frame and applies their union crop to all eight frames. InternVideo encodes
   global frames and valid crops into separate L2-normalized
   768-dimensional features. Missing crops use zeros; concatenation yields 1,536
   dimensions.
3. For N ≤ 900 observed intervals, model inputs are zero-padded to 900. Longer
   videos use all N intervals without chunking or truncation, up to the released
   checkpoint's 5,000-position capacity. Saved logits and softmax probabilities
   retain only observed intervals. No temporal ordering decoder is applied.
4. CMU probabilities pass through the existing fixed alias mapping. EGTEA has no
   supported MOSCATO alias mapping and retains raw source-index outputs only.
   `scripts/log_inference_wandb.py` validates and uploads each completed bundle.

The crop policy and feature extraction are ChefOST adaptations; exact parity with
the training feature generator remains unverified. The CMU mapping reconstructs
source index order from released vocabulary metadata and covers only a subset of
CMU labels. Processing longer videos does not establish accuracy on them.

## Preview and execution

Create `runs/batch-preparation/modal_rates.json` with numeric `cpu_hour_cost` and
`mem_gib_hour_cost` values applicable to your Modal account. No current prices are
bundled. GPU runs also require `gpu_hour_cost: {"L4": <hourly rate>}` in that file.
Read current rates with `.venv/bin/python -m modal billing rates --json`.
Preview the available release without starting models or uploading outputs:

```sh
.venv/bin/python scripts/run_full_batch.py
```

The default experiment is `full-r2-v1`. To select a two-video test, take one CMU
and one EGTEA ID from the inventory and repeat `--video-id`. Preview that selection
first, then add execution only after approving its budget:

```sh
.venv/bin/python scripts/run_full_batch.py \
  --video-id "$CMU_VIDEO_ID" --video-id "$EGTEA_VIDEO_ID"

# Only after approval; set BUDGET_USD to the approved amount.
.venv/bin/python scripts/run_full_batch.py \
  --video-id "$CMU_VIDEO_ID" --video-id "$EGTEA_VIDEO_ID" \
  --execute --budget-usd "$BUDGET_USD"
```

`--execute` requires an explicit positive, finite `--budget-usd`; there is no
25 USD ceiling. The sequential worker requests 4 CPUs and 16 GiB; `--accelerator L4`
adds one L4 GPU and its hourly cost to the reservation.
`--timeout` defaults to 21,600 (six hours), with a supported range of 60–86,400.
The ledger reserves worst-case compute per attempt using the requested timeout
and configured rates. It is an estimate, not a provider billing hard cap.
All pending videos must fit the approved budget before the first model call.
Before paid stages, the coordinator checks static inputs, dependencies, weights
and the CMU vocabulary for the selected batch. Failures are recorded and the loop
continues to later videos; paid jobs are not automatically retried. Inspect
failures before manually attempting them again.
An attempted model stage cannot launch again in the same experiment directory;
a new paid attempt requires a new experiment and explicit authorization.

### GPU and batching experiment

`--accelerator L4 --batch-size 8` runs the detector and InternVideo encoder on one
L4 GPU. The detector still processes one frame at a time. The encoder processes
up to eight clips together, keeps missing-object rows zero, and shares a decoding
pass for global and object features. It retains FP32, disables TF32, and keeps
the same frame sampling, weights and crop rules. The first batches check numerical
agreement with single-clip execution and one CPU reference. The existing feature
manifest records the settings and checks; no extra feature artifacts are exported.
GPU crops and features may have small numerical differences from CPU results.
Feature timing includes validation overhead, also recorded in
`numerical_validation.seconds`; `feature_extraction_wall` includes the shared decode.

The three-video comparison uses the same Eggs and Salad videos as the CPU pilot,
plus a longer EGTEA video that exercises more than 900 intervals. It produces raw
EGTEA outputs, without a CMU mapping or evaluation metrics. Preview first:

```sh
WANDB_ENTITY=chefOST WANDB_PROJECT=vidosc-experiments \
  .venv/bin/python scripts/run_full_batch.py \
  --experiment r2-l4-batch8-three-video-v1 --accelerator L4 --batch-size 8 \
  --timeout 3600 --budget-usd 20 \
  --video-id S08_Eggs_7150991-727 --video-id S29_Salad_7150991-1 \
  --video-id P17-R04-ContinentalBreakfast
```

Add `--execute` only for the authorized experiment. It runs sequentially and does
not change another checkout's running CPU job. Use a separate checkout and output
directory; cache identities include the compute settings. Do not edit its code
while it runs. The CPU/GPU comparison should use recorded stage times and actual
Modal charges rather than the conservative reservation ledger.

For the full available batch, omit `--video-id`. For an unattended Mac run, load
credentials in the current shell and set the approved budget before starting:

```sh
mkdir -p runs
nohup caffeinate -is .venv/bin/python scripts/run_full_batch.py \
  --execute --budget-usd "$BUDGET_USD" >runs/full-r2.log 2>&1 &
```

Leave the laptop open, connected to AC power and the internet, and running until
results have been retrieved and uploaded. The display can turn off; the computer
cannot be powered off. `nohup` permits closing the terminal, while `caffeinate`
keeps the coordinator awake. Do not run two coordinators in the same experiment
directory. Follow `runs/full-r2.log` and the experiment's `batch_status.json`.

## Outputs and reuse

Outputs live under `runs/batches/<experiment>/<video_id>/model/`. Every video
retains the existing five raw files:

- `raw_logits.npz`: `state_logits` (N×52) and `action_logits` (N×28), before softmax.
- `raw_probabilities.npz`: probabilities over the source classes.
- `combined_features.pth.tar`: N×1536, global 768 followed by object 768.
- `object_features.pth.tar`: N×768, including missing-crop zeros.
- `feature_manifest.json`: dataset, sampling, crops, input/checkpoint hashes,
  extraction device, per-stage elapsed seconds and reused interval counts.

CMU adds three files: `predictions/probabilities.npz`,
`predictions/predictions_by_second.csv` and `predictions/manifest.json`. Its W&B
artifact therefore retains eight files. EGTEA uploads the five raw files with
`mapping_applied: false`; any mapped prediction directory is rejected. No separate
global-feature file is exported. These are VidOSC outputs, not reusable feature
inputs promised for TubeletGraph or MOSCATO.

`wandb_inference_receipt.json` records the actual destination and run URL. The
experiment directory contains `batch_status.json` and per-video logs. W&B defaults
to `chefOST/vidosc-experiments`, with group equal to the experiment name. Confidence
and crop coverage are not accuracy; one active-object stream does not provide a
separate prediction for every annotated object.

Completed stages are reused only when receipt commands, environments, inputs and
output hashes match. Inspect partial outputs without receipts before retrying.
For changed code or inputs, choose a fresh directory with `--experiment`; use the
same value for preview and execution. An existing completed bundle can be uploaded
to a changed W&B destination or explicit group without repeating model inference.

The `vidosc-active-object-progress` Modal volume saves detections and global/object
feature progress every ten intervals. Resume requires matching videos, sampling,
weights, crop policy and extraction code. Per-stage timings describe one attempt,
including the R2 input copy but excluding image startup, laptop uploads and earlier attempts. Local files, upload
receipts and the remote volume are separate and are not included in a fresh clone.
