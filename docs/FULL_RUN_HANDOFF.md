# CMU30 active-object full runs: team handoff

This is the current entry point for the full-run code on `pritika/main`.
`BATCH_PLAN.md` and `FULL_RUN_READINESS.md` are historical preparation/audit
snapshots; their run counts, budgets and outstanding implementation lists are
not live status. Publishing this code does not launch compute.

## Scope and flow

The frozen IDs are in `configs/moscato_cmu30_v1.json`: 30 available CMU videos,
not all MOSCATO datasets or every annotated object. Each video produces one
unnamed hand-associated active-object stream. It may switch objects over time.
Ground-truth object labels are not inference inputs. Metrics are disabled.

1. `scripts/inventory_moscato.py`: inspect local video durations/FPS with ffprobe.
2. `scripts/prepare_full_video_plan.py`: generate explicit stages and file paths.
3. `moscato/extract_internvideo.py`: extract global features on local CPU.
4. `scripts/modal_clip.py`: detect hand-associated crops, extract object features,
   concatenate features and run the ChangeIt checkpoint on Modal CPU.
5. `moscato/import_modal_predictions.py`: map probabilities to CMU state labels.
6. `scripts/log_inference_wandb.py`: validate and upload outputs per video.

`scripts/run_cmu30_active.py` coordinates these stages sequentially, skips uploaded
videos and records failures. `scripts/batch_vidosc.py` checks stage receipts and
hashes. `moscato/progress_cache.py` supports remote extraction checkpoints every
10 intervals; `moscato/reuse_global.py` validates reusable global features.

## Setup on another machine

Clone `chefOST/vidosc` and switch to `pritika/main`. Follow the environment,
annotation/vocabulary and InternVideo setup in [MOSCATO_BASELINE.md](../MOSCATO_BASELINE.md).
Use a repository-local `.venv`, install Modal and W&B there, and authenticate each
with your own account. Install ffmpeg/ffprobe. The current scripts expect:

- `../internvideo-upstream/` (the sibling checkout described in the setup guide).
- `~/Downloads/InternVideo-MM-L-14.ckpt`.
- `checkpoints/changeit/state_best.ckpt`.
- `data/moscato/annotations/gt_annotations_cmu.json`.
- `data/moscato/vocabulary/files_dict.json` and `state_dict.json`.
- Videos at `data/moscato/videos/<video_id>.avi`.

These large/data files and account credentials are deliberately not in Git.
The download inventory is a historical availability/size snapshot, not a guarantee
that the source server is still available. To retrieve the frozen videos:

```bash
mkdir -p runs/batch-preparation
cp configs/cmu30_download_inventory.json runs/batch-preparation/download_inventory.json
.venv/bin/python scripts/download_cmu30.py
.venv/bin/python scripts/inventory_moscato.py
```

Before using the coordinator, create `runs/batch-preparation/modal_rates.json`
with numeric `cpu_hour_cost` and `mem_gib_hour_cost` fields from your current
Modal pricing. It deliberately has no bundled current-price claim.

Preview the batch without inference:

```bash
.venv/bin/python scripts/run_cmu30_active.py --budget-usd 25
```

When you are ready to incur compute charges and upload to the shared project:

```bash
.venv/bin/python scripts/run_cmu30_active.py --execute --budget-usd 25
```

Choose your own authorized budget (current script accepts up to $25). The budget
ledger estimates worker reservations; it is not a provider billing cap. Workers
have a 7200-second timeout, 4 CPUs, 16 GiB and no automatic retries. Keep the local
coordinator awake/connected for result retrieval and W&B upload. Do not run two
coordinators against the same experiment directory. `--wait-s08` and
`scripts/retry_s08.py` are recovery tools for the original local experiment, not
fresh-run prerequisites. The original finish-Brownie/shutdown watchers lived under
ignored `runs/`; they are not part of the reusable batch runner.

## Outputs and team reuse

Outputs live under `runs/batches/cmu30-active-object-v1/<video_id>/model/`.
The coordinator logs each video to a sibling `<video_id>.log`; `batch_status.json`
tracks progress. W&B defaults to entity `chefOST`, project `vidosc - full evals`,
group `cmu30-active-object-v1`. The upload receipt contains the actual run URL.

| File | Meaning |
| --- | --- |
| `raw_logits.npz` | State N×52 and action N×28 scores before softmax |
| `raw_probabilities.npz` | Probabilities over the checkpoint's state/action classes |
| `combined_features.pth.tar` | N×1536: global 768 followed by object 768 |
| `object_features.pth.tar` | N×768; zero fallback for missing object crops |
| `feature_manifest.json` | Timing windows, crop/provenance details and hashes |
| `predictions/probabilities.npz` | Scores mapped into the supported CMU vocabulary |
| `predictions/predictions_by_second.csv` | Human-readable predictions by interval |
| `predictions/manifest.json` | Mapping and prediction metadata |

N is the observed number of roughly one-second intervals. Match windows and hashes
when reusing tensors across models; do not assume annotation frames use a universal
FPS. Raw confidence is not accuracy. No temporal decoder or internal transformer
activation export is applied here. A video-level stream must not be presented as
separate predictions for every annotated object. Videos over 900 seconds and
other MOSCATO datasets require additional protocol/mapping work.

Validation for this handoff: 25 focused baseline, batch, cache and reuse tests pass.
The broader unittest discovery is blocked by a missing local `ffmpeg` Python module
in the legacy caption pipeline. No paid inference was launched for the handoff.
