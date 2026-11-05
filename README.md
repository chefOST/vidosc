# VidOSC / ChefOST

ChefOST evaluates object-state tracking in procedural cooking videos. This branch
extends Pritika's full-video workflow to the pinned R2 release: 29 CMU and 12 EGTEA
videos available for inference, with 46 unavailable EgoPER videos reported separately.
Each video produces one unnamed hand-associated active-object stream. CMU outputs
include the existing alias mapping; EGTEA outputs remain raw, without a new mapping.
These inventory counts are not completed-run or benchmark results.

Global/object feature extraction runs on Modal CPU by default, or an L4 GPU with
batched clips using `--accelerator L4 --batch-size 8`. VidOSC inference uses CPU. The worker
reads videos from R2; the local machine supplies verified weights, coordinates
stages, and retrieves outputs. Full videos are not downloaded to the laptop.

Start with [setup and workflow](docs/FULL_RUN_HANDOFF.md),
[account setup](docs/account-setup.md), and [agent instructions](AGENTS.md).
The [documentation index](docs/README.md) links the remaining references.

## Metrics

The current workflow exports inference outputs and coverage; it does not compute
evaluation scores. ChefOST's evaluation definitions are:

For each object, `P_t` is the predicted state set and `T_t` the ground-truth state set at frame `t`.

- Precision: fraction of predicted states that are correct, `|P_t ∩ T_t| / |P_t|`, averaged over frames with predictions.
- Recall (detail): fraction of ground-truth states recovered, `|P_t ∩ T_t| / |T_t|`, averaged over frames where the object is present.

Average per-frame ratios equally; missing predictions give zero recall when GT states exist. Record the dataset, split and sampling; document empty-GT handling and aggregation across objects/videos. Keep published paper results distinct from ChefOST experiments.

## Context

Read the [project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals), then the [VidOSC paper notes](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#1-vidosc). Use the [organization guide](https://github.com/chefOST/.github/blob/main/profile/README.md) for shared project context and the [data repository](https://github.com/chefOST/chefOST-data) for annotations, vocabularies and shared R2 video access.

## Experiment tracking

W&B defaults to `chefOST/vidosc-experiments`; the full-batch group follows the
experiment name (`full-r2-v1` by default). See [W&B setup](weights_and_biases/README.md)
for configuration and upload commands.
