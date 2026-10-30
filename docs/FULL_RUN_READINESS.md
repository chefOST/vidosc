# VidOSC full-run readiness — 2026-10-24

**Decision: not ready for an all-object, all-dataset launch.** Full-video inference
works on one video. Target-object selection and cross-activity consistency remain
unvalidated. No paid jobs were launched for this audit.

## Verified available annotation scope

Downloaded seven named ground-truth JSON files from chefOST/chefOST-data and checked
their sizes and SHA-256 hashes against the Git LFS pointers in the local checkout.
These are the available repository snapshot, not proof of the complete original
paper splits. Train pseudo-label files are excluded. Videos are not included.

| Dataset/task | Videos | Video/object pairs |
|---|---:|---:|
| CMU | 30 | 269 |
| EGTEA | 12 | 202 |
| EgoPER coffee | 10 | 126 |
| EgoPER oatmeal | 10 | 128 |
| EgoPER pinwheels | 9 | 119 |
| EgoPER quesadilla | 7 | 69 |
| EgoPER tea | 10 | 109 |
| **Total** | **88** | **1,022** |

Files: `data/moscato-release/annotations/ground_truth/`. Exact IDs, objects and
annotation lengths: `runs/readiness-audit/dataset_inventory.json`. Regenerate with
`.venv/bin/python scripts/audit_full_run_readiness.py` after downloading the files.
Only the CMU S29 Salad video is downloaded locally. Durations for the rest are
unverified; do not infer hours from annotation counts using a blanket FPS value.

## Existing full-video run

The previously launched full S29 Salad run **completed**. It covers 0–269.333 seconds
with 270 intervals. Raw state probabilities are 270×52 and action probabilities
270×28; both are finite and normalized. Detector crops exist in 202 intervals;
68 use zero-object features. Remote extraction/inference took 1,041.4 seconds;
local global extraction took approximately 491 seconds.

No local W&B upload receipt exists for this full-video run; publication is not
verified. The earlier 107-second W&B run is a separate experiment. Existing metric
files were inspected but no new metrics were computed during this audit.

Raw state argmax is background in 174 intervals and peeled in 96. This alone does
not establish consistently reasonable behavior across activities. The initial
107-second run and this full video are not independent sample videos.

## Output contract and changes

- Global and object features: preserved as tensors; crop decisions in feature manifest.
- **Raw state/action logits:** new `raw_logits.npz` export before softmax, trimmed to
  observed intervals. Historical full-S29 files do not contain logits; do not claim
  they were saved retroactively. They can later be recovered by local inference on
  saved features without repeating expensive extraction.
- State/action probabilities: existing `raw_probabilities.npz` retained.
- MOSCATO-mapped scores: existing `predictions/probabilities.npz` retained for CMU.
- Temporal post-processing: explicitly recorded as **not applied**. Mapping and
  softmax are distinguishable stages; there is no fabricated 'after decoder' file.
- Internal transformer activations are not exported; clarify their necessity before
  adding storage-intensive hooks if the team means those by 'layers'.
- New plans default to global extraction → inference → mapping. Metric evaluation
  and metric reports require explicit `--with-metrics`.

Validated with 22 local adapter/batch tests, including logits-to-softmax equivalence,
padding exclusion, resume checks and execution gates. Output-only plan dry run passed.
The modified remote exporter has not been run on Modal yet.

## Remaining implementation work before full launch

1. Retrieve and inspect videos for a small cross-activity pilot, then the frozen scope.
   Check source identity, duration, frame clocks, decode failures and storage needs.
2. Implement and validate target-object selection. Current hand-associated crops can
   contain tools or boards; one video-level stream must not be copied across every
   object. Decide how to handle unseen/non-manipulated objects and visibility gaps.
3. Define mappings separately for EGTEA and each EgoPER vocabulary. The current
   hash-checked mapping is CMU-only; the planner cannot execute these other datasets
   as-is. Preserve unmapped raw outputs for future evaluation.
4. Decouple feature extraction from object labels for cache reuse across objects.
   Generalize the CMU pilot planner and separate inference inputs from scoring inputs.
   Plan videos over 900 seconds explicitly; no silent truncation or unreviewed chunking.
5. Add an output-only W&B logger. The current uploader expects metric/report files,
   so it is not compatible with the new metric-free plan. Log completion, timings,
   object coverage, configuration/hashes, raw and mapped artifacts without scores.
6. Run several representative videos with metrics disabled; inspect crops and output
   stability. Confirm failures resume safely and output files are complete.
7. Estimate full cost from measured video duration/object count, cache reuse, setup,
   storage and transfer. Apply a contingency margin and secure the actual run budget.

## Cost assessment

Historical rates read on October 11 were $0.04730/CPU-hour and $0.008/GiB-hour.
At 4 CPUs and 16 GiB, the measured 1,041.4-second remote stage corresponds to about
$0.092 in worker compute, excluding setup/transfer and local extraction. This is
not a current quote or a full-dataset estimate. We cannot extrapolate to 1,022 object
streams without their durations and a working object-selection method.

The team's <$100 readiness criterion is separate from the last authorized personal
budget ($1, free credits only). This audit did not increase that budget.

## Honest team update now

> VidOSC full-video inference is working on the complete S29 Salad video. We have
> inventoried 88 videos / 1,022 video-object pairs in our available MOSCATO ground-truth
> snapshot and added pre-softmax logit export plus metric-free plan generation.
> The pipeline is not yet ready for the full all-object run: object-specific crops,
> cross-dataset mappings, and a diverse pilot are outstanding. After those checks,
> we will report a measured cost estimate and preserve outputs at every implemented
> processing stage. No full-dataset readiness claim yet.
