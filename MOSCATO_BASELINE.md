# VidOSC-only evaluation on MOSCATO — work in progress

This adds a fixed-label-mapping transfer evaluator without Claude or training.
The existing `MOSCATO.md` describes a separate VidOSC + Claude experiment.
Neither document's results should be attributed to the other method.

## Current status

- Official ChangeIt multitask `state_best.ckpt` downloaded and loaded locally.
  It has 1,536 input dimensions, 52 state outputs and 28 action outputs.
- CMU ground truth and `S29_Salad_7150991-1.avi` downloaded locally.
  The video decodes to 8,080 frames at 30 FPS, matching 8,080 annotation entries.
  Sample frames are consistent with the annotation timeline. Formal timing
  metadata still requires review; do not assume the generic 10-FPS README
  description applies to this file.
- 18 regression tests passed locally, including a forward pass with the real
  checkpoint on **synthetic inputs**. This verifies loading, not model accuracy.
- No real-video feature extraction, prediction run, or benchmark result yet.
- No training and no Claude/API calls are involved in this baseline.

The checkpoint, video, annotations, virtual environment and generated outputs
are ignored by Git. They are not included in a fresh clone.

## Why features are needed

The released VidOSC transformer consumes precomputed feature tensors, not AVI
or MP4 bytes. Its input pipeline is:

    video -> matching feature extractor -> feature tensor -> VidOSC -> scores

The downloaded multitask checkpoint expects global and object-centric
InternVideo features concatenated together. The repository's older
`extract_features.py` uses CLIP instead; it is an approximation and does not
produce the full required input. This adapter rejects mismatched dimensions
and does not silently zero-pad them. Matching dimensions alone do not establish
feature compatibility: backbone, preprocessing, temporal sampling and layout
must also be checked. Feature-model setup was paused at the user's request.

## Files

- `moscato/checkpoint_adapter.py`: loads official single-head HowToChange or
  dual-head ChangeIt models directly from upstream `model.py`, without importing
  training code or the CUDA-only ordering extension. Uses CPU inference and
  restricted `torch.load(weights_only=True)`. No automatic unsafe-pickle fallback.
- `moscato/baseline.py`: checkpoint inspection, prediction, explicit probability
  mapping, annotation-clock alignment and scoring commands.
- `moscato/baseline_metrics.py`: multi-label frame F1-Max/mAP and segment IoU F1.
- `tests/test_baseline.py`: timing, mapping, metrics, file integrity and adapter tests.

## Setup and verification

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
git clone https://github.com/facebookresearch/VidOSC.git ../vidosc-upstream
git -C ../vidosc-upstream checkout 59575773c97878ea30a0228f28ab00a3ee2f1ea2
python -m pytest -q tests/test_baseline.py
python -m moscato.baseline inspect-checkpoint \
  --checkpoint checkpoints/changeit/state_best.ckpt
```

The real-checkpoint test skips when the checkpoint or upstream checkout is
absent. Model attention heads default to the upstream setting of 4; this cannot
be recovered from parameter shapes and must match the checkpoint's training.

## Inputs to prepare before a real run

1. A compatible `[timesteps, input_dim]` PyTorch feature tensor.
2. A feature manifest with `video_id`, `object`, `video_sha256`,
   `clip_start_seconds`, `clip_end_seconds`, `interval_starts_seconds`,
   `feature_backbone`, and `feature_layout`. Intervals are clip-relative,
   start at zero, and are half-open; the last ends at the clip end.
3. A mapping JSON with `checkpoint_sha256`, a documented `rationale`,
   `ground_truth_used_to_define_mapping: false`, and `source_index_to_states`.
   Every source class index must appear as a string key, mapping to a list of
   canonical MOSCATO states or `[]`. Recover class meanings from the original
   training vocabulary; do not guess indices or derive mappings from test labels.
4. The MOSCATO `state_dict.json` and string-label ground-truth JSON.
5. A timing JSON with `video_id`, `video_sha256`, `annotation_fps`,
   `annotation_count`, `verified: true`, and `evidence` documenting verification.

Use these commands **after** those inputs exist:

```bash
python -m moscato.baseline predict \
  --checkpoint checkpoints/changeit/state_best.ckpt \
  --features data/moscato/features.pth.tar \
  --feature-manifest data/moscato/feature_manifest.json \
  --mapping data/moscato/mapping.json \
  --state-dict data/moscato/vocabulary/state_dict.json \
  --upstream ../vidosc-upstream --output runs/baseline/predictions

python -m moscato.baseline evaluate \
  --run runs/baseline/predictions \
  --annotations data/moscato/annotations/gt_annotations_cmu.json \
  --state-dict data/moscato/vocabulary/state_dict.json \
  --timing data/moscato/timing.json \
  --output runs/baseline/scoring
```

Prediction does not load ground-truth annotations. Scoring checks prediction
and vocabulary hashes and writes a separate CSV and evaluation JSON. Mapping
provenance is a declaration, not proof: reviewers must verify it. Use fresh
output directories; the commands refuse existing ones.

## Scoring scope and limitations

This is a **single video/object clip transfer baseline**, not a reproduction of
the entire MOSCATO benchmark. It has no confirmed official-evaluator parity.
There is no learned MOSCATO classification head, ordering decoder, multi-video
aggregation, or automatic object localization in this addition.

Scores are projected by summing mutually exclusive source-class probabilities
for each target state. Unmapped GT states remain in evaluation as misses.
Empty GT entries are skipped and split temporal segments. Frame F1-Max sweeps
0.1 through 0.9; mAP averages classes with positive GT within the evaluated clip.
Segment F1 uses one-to-one same-label matching at IoU 0.1, 0.25 and 0.5, with a
fixed decision threshold of 0.5 by default. These choices must be reconciled with
the official protocol before comparing published benchmark numbers.

## Data and model sources

- [VidOSC code and checkpoint descriptions](https://github.com/facebookresearch/VidOSC)
- [User-provided ChangeIt checkpoints](https://drive.google.com/drive/folders/1UyRQ_dlT0AjeCLM3LlxnjFtRRUm7Fowj)
- Multitask state checkpoint SHA256:
  `e6d1fa0d4708c8dffecba2a37f25149cc80c85fed33e6d4537bcaf98b0d94d36`
- [MOSCATO annotations in chefOST-data](https://github.com/chefOST/chefOST-data)
- [MOSCATO authors' dataset documentation](https://github.com/eelhami/iccv25_moscato)
- [CMU source video archive](http://kitchen.cs.cmu.edu/Main/S29_Salad_Video.zip)

Retain dataset/model licenses and citations when using or redistributing assets.
