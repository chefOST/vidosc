# Human-supervised accuracy follow-up

This follow-up improves the previous selected checkpoint on four new CMU
recordings. Validation selected `stable`: fresh mean target/video multi-label
F1 is 53.1%, versus 35.5% for the unchanged old checkpoint and 46.9% for the
same-training-data constant prior. The retrained original architecture scored
56.2%; the new architecture changes did not improve fresh-test F1 in this run.
See [CHANGES.md](CHANGES.md) for the implementation walkthrough and
[REPORT.md](REPORT.md) for failures, paired recording uncertainty and metrics.

## What changed

- Nine human-labelled training recordings cover six target categories, including
  egg mixture, cheese and batter. Five separate recordings provide validation.
- Training uses raw-frame-weighted soft targets within each two-second prediction
  bin, unweighted binary cross entropy, equal weighting per object/video and a
  fixed 100-epoch budget. CLIP and VidOSC features stay frozen.
- `original` retrains the existing text scorer. `stable` uses fixed object-query
  scaling and query-relative text centering. Prototypes and prior blends provide
  declared alternatives; every hyperparameter/model choice uses validation.
- Four new subject/recordings provide six annotated object/video test units.
  Original exposed tests are development data now. This is known-category
  transfer within one kitchen, not unseen-object or unseen-state recognition.

Both Michelle's pipeline and this module aim for semantic states over time.
Her pipeline describes VidOSC phases using Claude; this module learns to score
supplied candidate descriptions against video/temporal features. Abstract
states are a shared goal. These results are not a comparison with Michelle's
accuracy and do not establish algorithmic novelty.

## Inspecting the completed run

`results/selection-lock.json` records validation selection, source/model hashes
and the absence of fresh-test scores at selection. `prediction-lock.json` hashes
all methods' predictions saved before fresh-label export/scoring.
`evaluation.json` contains per-state and segment metrics; `audit.json` contains
paired recording bootstrap intervals and integrity/inference checks.

Models, features, video files and predictions in `.pt` form are local artifacts
excluded from Git. The module depends on the prepared workspace described in
the parent README: `work/vidosc-env`, original `work/VidOSC` code/weights,
`work/chefost-data/MOSCATO`, cached videos and `work/shray-broader` features/text
embeddings. It is not a standalone clone-and-run package.

From the prepared workspace root in PowerShell:

```powershell
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/v2/test_accuracy.py
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/v2/audit_results.py
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/v2/predict.py --video-id S17_Salad_7150991-822 --object vegetables --candidates work/chefost-vidosc/experiments/shray_multistate/broader/candidate-states-cached-example.json --output work/chefost-vidosc/experiments/shray_multistate/v2/local-predictions/salad-three-states.json
```

The completed inference example uses three supplied names; the trained network
also passed shared-score consistency checks with 3, 5 and 77 candidates. This
does not imply those states are automatically discovered or equally accurate.
Custom candidates can be strings or objects with `state` and a full
`description` prompt. Uncached descriptions require the CLIP text encoder.
Scores are not calibrated probabilities. Candidate selection changes which
labels can be returned; the globally selected threshold is not validated for
every custom vocabulary.

## Reproduction of the declared experiment

Read `PLAN.md` and `protocol.json` first. `prepare_experiment.py` prepares the
locked split/labels and `broader/download.py --protocol <v2 protocol path>
--manifest v2-sources.json` obtains missing recordings. Feature extraction uses
`broader/features.py --backbone clip --backend directml --protocol <v2 protocol
path>` with the existing verified ONNX cache; `--backend torch` is the CPU
alternative. `run_accuracy.py` trains, locks selection, waits for fresh features,
saves/hashes predictions and evaluates. It rejects an existing results directory.
Preserve this completed experiment; a later model-selection experiment needs
its own declared run and fresh test recordings rather than overwriting these.

Post-evaluation `predict.py` and `audit_results.py` are separate helpers added
after the training-code snapshot. They do not change training, selected settings
or stored fresh-test predictions. Implementation changes are confined to Shray's
own module; the contributor-branch overview also links the new evidence.
Large local checkpoints and datasets are excluded from Git.
