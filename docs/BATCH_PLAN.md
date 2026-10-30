# MOSCATO batch preparation

Stage 1 only: inventory and code preparation. No new inference, downloads, Modal jobs,
or W&B uploads were performed. Existing results remain the 107-second pilot.

## Inventory

30 annotated videos: Brownie 10, Eggs 7, Salad 6, Pizza 4, Sandwich 3.
Only S29 Salad is local: 8,080 metadata frames, 30 FPS, 269.333 seconds.
See [inventory](moscato_inventory.md) for per-object label coverage. Counts refer
to distinct labels, not a performance measure. Other durations are unknown.
Candidate archive URLs in `runs/batch-preparation/inventory.json` derive from
released `files_dict.json` paths; their availability has not been verified.

## Pilot proposal

1. Full S29 Salad, vegetables: use the existing video, 270 one-second intervals
   (last interval shortened). The previous clip began at frame 1469, whereas this
   plan begins at frame 0. Sampling windows differ, so its features are not reused.
2. S09 Eggs (`S09_Eggs_7150991-1050`), egg mixture: 3 of 5 distinct annotated
   states map to checkpoint labels. Download availability/duration remain unknown.
3. S24 Brownie (`S24_Brownie_7150991-483`), eggs: 3 of 6 states map. Same checks pending.

These are development/debugging choices based on activity diversity and label
coverage, not a representative test subset. Freeze the protocol before dataset-wide
evaluation. Do not treat a single active-object prediction as separate object-specific
predictions for every annotation object; object association needs review first.

## Prepare and inspect (no compute)

From the repository root:

```bash
.venv/bin/python scripts/inventory_moscato.py
.venv/bin/python scripts/prepare_full_video_plan.py
.venv/bin/python scripts/batch_vidosc.py --plan runs/batch-preparation/full-video-pilot-v1-S29_Salad_7150991-1.json
```

The prepared plan uses local CPU for global features, then Modal CPU for detection,
object features and VidOSC inference, then local mapping, evaluation and reporting.
It does not upload to W&B. After approval, execution requires both `--execute` and
`--allow-modal`. One job is allowed by default; larger batches require an explicit
`--max-jobs`. No automatic paid retries are configured.

Every completed stage has a receipt with command/environment and input/output
hashes. Matching stages are reused. Changed inputs or damaged outputs are rejected;
use a new experiment directory. Partial outputs without receipts require inspection
before retrying. Resume is between stages, not within a remote extraction. Do not
run the same plan concurrently. The remote stage has a one-hour timeout; this is
not a dollar spending cap. Larger videos need a measured runtime and resource plan.

The full-video planner requires a local video, matching metadata/annotation counts,
and a video-specific timing file. Matching counts alone do not verify alignment.
The current S29 plan explicitly retains provisional timing. Videos over 900 seconds
are blocked until a temporal chunking protocol is agreed, because chunking changes
the transformer's temporal context.

## Before launching

- Approve one full-video run and its compute budget. A rough CPU scaling estimate
  is about 40 minutes for the remote S29 stage, plus global extraction/setup; this
  is not a measured runtime or a dollar quote.
- Check annotation transitions and object crop relevance; preserve provisional status.
- Benchmark actual cost before approving the other two downloads/runs or a GPU trial.
- Upload each reviewed result to `chefOST/vidosc-experiments`, using the existing
  upload script's `--run-dir` option. No automatic W&B publication in the batch runner.
- Add experiment grouping and a pooled dataset evaluator before the full launch.
  Do not substitute a mean of per-video F1 scores for pooled F1.

Local validation: 35 tests passed, including dry-run/Modal gates, successful stage
reuse, changed-input rejection, partial-output rejection, and failed-stage handling.
The generalized remote script has not yet been executed on a full video.
