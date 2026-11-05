# Active-object full-video workflow

This is the maintained workflow on `harrison/vidosc-experiments`, based on
Pritika's October 30 handoff for the October 24 runs. The frozen IDs in
[`configs/moscato_cmu30_v1.json`](../configs/moscato_cmu30_v1.json) describe a
historical 30-video CMU subset. Each video produces one unnamed hand-associated
active-object stream that may switch objects over time. Ground-truth object labels
are not inference inputs, and this workflow computes no evaluation metrics.

## Setup

From this repository's root, use Python 3.11 and a local `.venv`:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Install the FFmpeg system tools (`ffmpeg` and `ffprobe`; `brew install ffmpeg` on
macOS). Configure [your Modal and W&B accounts](account-setup.md).

Download these pretrained weights from the linked official releases and place them
at the paths expected by the planner:

- `~/Downloads/InternVideo-MM-L-14.ckpt`: the InternVideo-MM-L-14 model from
  [InternVideo's multimodal release](https://github.com/OpenGVLab/InternVideo/tree/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo1/Pretrain/Multi-Modalities-Pretraining).
  SHA256: `a508e42229072f84a3fcd5fc3acdb7ebc969b27519ee1f56dac9996999cd40f6`.
- `checkpoints/changeit/state_best.ckpt`: the ChangeIt **multitask** state checkpoint
  from the [VidOSC checkpoint release](https://drive.google.com/drive/folders/1UyRQ_dlT0AjeCLM3LlxnjFtRRUm7Fowj).
  SHA256: `e6d1fa0d4708c8dffecba2a37f25149cc80c85fed33e6d4537bcaf98b0d94d36`.

The Modal image clones InternVideo at `3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3`
and VidOSC at `59575773c97878ea30a0228f28ab00a3ee2f1ea2`, and builds the hand/object
detector environment. Neither source checkout is needed on the local machine.
The detector uses a pinned checkpoint mirror with
Git LFS size/hash checks. Source revisions and dependencies are defined in
[`scripts/modal_detector.py`](../scripts/modal_detector.py) and
[`scripts/modal_clip.py`](../scripts/modal_clip.py).

## Data and inventory

Populate these ignored local paths using the matching CMU release snapshot:

- `data/moscato/annotations/gt_annotations_cmu.json`
- `data/moscato/vocabulary/files_dict.json` and `state_dict.json`
- `data/moscato/videos/<video_id>.avi`

The frozen annotation SHA256 is
`41ddb05eb3b8dd3fd1eb159d10c29e8d2eba95bbdd3fd2cf697a202cf943f912`.
The mapping expects CMU `state_dict.json` SHA256
`0fbc692af4c5ff399a467804abb3e4d8af27fedb47256dc9e6072fb4f6b8aa05`.
Use the corresponding historical files from [chefOST-data](https://github.com/chefOST/chefOST-data).
The current shared release excludes `S13_Brownie_7150991-631` for a video/annotation
length mismatch; the historical CMU30 configuration still includes it. Do not
silently replace this snapshot with the current curated subset.

The current runner uploads local AVIs and pretrained weights to Modal. It does not
mount the shared R2 MP4s. R2 access is documented in [account setup](account-setup.md);
using those MP4s requires explicit integration, fresh manifests and frame-clock checks.

To download the frozen AVI subset from the original archives, then inspect it:

```sh
mkdir -p runs/batch-preparation
cp configs/cmu30_download_inventory.json runs/batch-preparation/download_inventory.json
.venv/bin/python scripts/download_cmu30.py
.venv/bin/python scripts/inventory_moscato.py
```

The download inventory records historical availability and sizes, not guaranteed
current availability. The downloader extracts the named camera video from each
archive. Inspect `runs/batch-preparation/inventory.json` and `inventory.md` before
planning. Metadata counts alone do not verify annotation alignment. Inventory may
read annotations, but the remote inference worker receives no ground truth.

## Pipeline and execution

1. `scripts/prepare_full_video_plan.py` creates the Modal model stage and local
   mapping stage. The local machine does no feature extraction or model inference.
2. `scripts/modal_clip.py` samples eight frames per one-second interval, with a
   shortened final interval. It detects hand-associated objects on the fifth
   sampled frame and applies their union crop to the interval's eight frames.
   InternVideo encodes global frames and valid crops on Modal CPU into separate
   L2-normalized 768-dimensional features. Missing crops use zeros; global and
   object features concatenate to 1,536 dimensions.
3. For N ≤ 900 observed intervals, the transformer still receives zero padding
   to 900. Longer videos use all N intervals without chunking or truncation, up
   to the released checkpoint's 5,000-position capacity. Saved logits and softmax
   probabilities retain only observed intervals. No temporal ordering decoder
   or internal-activation export is applied.
4. `moscato.import_modal_predictions` projects saved state probabilities through
   the fixed CMU alias mapping. `scripts/log_inference_wandb.py` validates and uploads
   each completed output bundle.

The crop policy and extraction compatibility are adaptations; their exact parity
with the original training feature generator is unverified. The mapping reconstructs
source index order from released vocabulary metadata and supports only a subset
of CMU labels. Preserve the raw outputs alongside mapped scores.

Create `runs/batch-preparation/modal_rates.json` with numeric `cpu_hour_cost` and
`mem_gib_hour_cost` values from the rates applicable to your account. No current
prices are bundled. Set `BUDGET_USD` to the amount approved for your run; the
coordinator currently accepts positive amounts up to 25 USD. That software limit
does not carry forward approval from Pritika's earlier experiment.

Preview the configured batch without running models or uploading results:

```sh
.venv/bin/python scripts/run_cmu30_active.py --budget-usd "$BUDGET_USD"
```

After reviewing the inventory, account destination and budget, execution uses:

```sh
.venv/bin/python scripts/run_cmu30_active.py --execute --budget-usd "$BUDGET_USD"
```

The coordinator runs sequentially, using 4 CPUs, 16 GiB, a 7,200-second worker
timeout and no automatic inference retries. Its ledger estimates reservations;
it is not a provider billing cap. Keep the local coordinator awake and connected
for result retrieval and upload. Do not run two coordinators against the same
experiment directory. Inputs beyond the checkpoint's 5,000-position capacity are
rejected; support for other dataset vocabularies requires a separate mapping.
Processing a longer sequence does not establish the model's accuracy on long videos.
Long videos may exceed the worker's two-hour execution timeout; a later attempt
can resume its saved extraction progress. Per-stage timings cover that attempt,
not image startup, input transfers or earlier attempts.

## Outputs and reuse

Outputs live under `runs/batches/<experiment>/<video_id>/model/`, with experiment
default `cmu30-active-object-v1`:

- `raw_logits.npz`: `state_logits` (N×52) and `action_logits` (N×28), before softmax.
- `raw_probabilities.npz`: state/action probabilities over the source classes.
- `combined_features.pth.tar`: N×1536, global 768 followed by object 768.
- `object_features.pth.tar`: N×768, including explicit missing-crop zeros.
- `feature_manifest.json`: sampling windows, crops, checkpoint/input hashes,
  extraction device, per-stage elapsed seconds, reused interval counts and output-stage metadata.
- `predictions/probabilities.npz`: source probabilities and CMU-mapped state scores.
- `predictions/predictions_by_second.csv` and `predictions/manifest.json`: named
  interval predictions and mapping provenance.
- `wandb_inference_receipt.json`: the actual upload destination and run URL.

N is the number of observed intervals. The experiment directory also holds
`batch_status.json` and per-video logs. [W&B configuration](../weights_and_biases/README.md)
defaults to `chefOST/vidosc-experiments`, group `cmu30-active-object-v1`.
Confidence and crop coverage are not accuracy, and a single active-object stream
does not establish separate predictions for every annotated object.

`scripts/batch_vidosc.py` reuses completed stages only when receipt commands,
environments, inputs and output hashes match. Inspect partial outputs without
receipts before retrying.
Receipt hashes from older code can fail after cleanup; use a fresh experiment
directory instead of weakening the checks. Both the planner and coordinator accept
`--experiment`. For example, preview a fresh directory with:

```sh
.venv/bin/python scripts/run_cmu30_active.py --experiment cmu30-active-object-v2 --budget-usd "$BUDGET_USD"
```

Use the same experiment value when adding `--execute`. The W&B group remains
`cmu30-active-object-v1` unless overridden at upload. If a complete bundle was
uploaded to a different W&B destination, the coordinator can upload those saved
outputs to the configured destination during execution without repeating inference.

Modal saves detections and global/object-feature progress every 10 intervals in the
`vidosc-active-object-progress` volume in your workspace. Resume requires matching
video, sampling, weights, crop policy and extraction code. An account's local files,
upload receipts and remote volume are separate; none is included in a fresh clone.
