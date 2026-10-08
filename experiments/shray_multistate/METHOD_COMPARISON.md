# How Shray's experiment fits the team work

This comparison describes committed code and documentation reviewed on October
7, 2026. It compares scope, not leaderboard positions. Private Modal/W&B logs
and uncommitted runs may contain additional results.

## Team approaches

| Contributor / work | Goal and implementation | Relationship to this experiment |
| --- | --- | --- |
| Michelle: VidOSC + Claude | Predict temporal phases, caption representative segment frames with Claude, map descriptions to MOSCATO labels and evaluate. Pipeline code exists; the status document reports no completed live prediction-to-evaluation run. | Shray trains semantic state scores directly over time, without a separate Claude description stage. |
| Pritika: pretrained VidOSC baseline | Extract compatible InternVideo features, run the official checkpoint and explicitly map fixed outputs to MOSCATO states, without new training or Claude. Global extraction is verified; the documented full input and benchmark remain unfinished. | Shray changes and trains the output mechanism. Its CLIP global features and zero-filled object stream do not reproduce original VidOSC inputs. |
| Justin's TubeletGraph branch / Tingxuan Wang (`jstxw`) | Start from a target mask, recover transformation events and semantic text, map them to MOSCATO states and evaluate the sandwich/bread target. The branch is named `justin/...`; commits are credited to Tingxuan Wang. | TubeletGraph addresses object tracking and transformations. Shray predicts state timelines without physical-object identity tracking. |
| Tingxuan: SAM2 and VISOR experiments | SAM2 tracking and DAVIS mask evaluation, VISOR preparation, crop/VLM experiments and Modal/Comet/W&B infrastructure. | These can supply object-centric inputs missing from Shray's full-frame method. Mask quality and state-label accuracy measure different tasks. |
| Ryan: SPOC pseudo-labelling | Grounding DINO, SAM and DeAOT produce tracked regions; CLIP and temporal constraints label actionable/transformed regions. Pseudo-labelling and spatial evaluation are implemented; the separate learned SPOC transformer is not. | Shray predicts named semantic states and intervals, rather than spatially progressing changed/unchanged regions. |
| Ryan: OSCAR/MOSP adaptation | A temporal Transformer, optionally with action-context attention, produces a fixed object-by-state output tensor with multi-label supervision. | Both use learned temporal representations and simultaneous states. Shray uses candidate-text scoring in place of a fixed-size output head. |
| Raf: data generation and SAM2 + SmolVLM | Tracking, object crops, pretrained VLM state classification, smoothing and timelines. State lists are configurable in the tracking script. Some dataset-generation stages use heuristic substitutes; a separate Ego4D prototype uses fixed toast labels. | Raf also built pipeline-level code. Shray trains a temporal text scorer rather than primarily classifying individual crops and smoothing labels. |
| Harrison: shared foundations | MOSCATO annotations/vocabularies, repository organization, research goals and contribution guidance; the demo repository is a placeholder. | Provides data and project structure used by the experiments. |

## Shray's implemented contribution

Video → frozen CLIP image features → pretrained VidOSC temporal context →
an own projection conditioned on an object-name embedding → comparison with
candidate state text embeddings → overlapping state intervals.

The experiment includes scorer training, optional fine-tuning of a separate
encoder copy, a frozen SlowFast fusion comparison, inference and timeline
export, subject-disjoint protocols, constant-prior baselines and held-out-state
evaluation. Validation selects model/epoch/threshold choices; test predictions
are saved before test scoring.

Candidate count is an input. Candidate names/descriptions are still supplied;
the benchmark uses 77 provided CMU state names. Arbitrary descriptions work as
inputs, but their recognition accuracy is unverified. Scores are not calibrated
probabilities. Multi-label prediction and configurable state lists also appear
in teammates' work and are not unique contributions by themselves.

This training-to-evaluation experiment runs. The full ChefOST goal remains
unresolved: no physical-object tracker, crop feature stream or automatic
candidate generator is implemented. The encoder uses future video context,
so this is offline analysis rather than causal streaming.

On five fresh videos/seven target units, the validation-selected model scored
30.3% mean unit multi-label F1 versus 38.1% for the stronger constant prior.
Held-out peeled-state F1 was zero, and SlowFast did not improve the fresh test.
The earlier 60.7% single-salad highest-label score is not directly comparable.
See the [full report](broader/REPORT.md).

Several teammates built pipeline-level implementations. Shray contributes a
distinct trainable text-conditioned temporal experiment with measured
limitations, not a claim of being the only complete pipeline or an accuracy
improvement over the team.

## Reviewed sources

- [Michelle's pipeline and status](https://github.com/chefOST/vidosc/blob/11dcdf3abe20c088dd2da2ebc15d1e62853e3e7a/MOSCATO.md)
- [Pritika's baseline](https://github.com/chefOST/vidosc/blob/4d9c4192f9186155e9b430855f3ed4adbe8a8121/MOSCATO_BASELINE.md)
- [TubeletGraph experiment](https://github.com/chefOST/tubletegraph/blob/32fa88a898a3ea3cec716823753d0ac5a1f0eda8/README.md)
- [SAM2 evaluation](https://github.com/chefOST/chefOST-ml/blob/014bdcecdf869405c25feb0c4d97e66156a5677f/modal_sam2_eval.py)
- [SPOC implementation](https://github.com/chefOST/spoc/blob/6cb78330b1aac3b74b4a809ff0791774ce059281/README.md)
- [OSCAR/MOSP model](https://github.com/chefOST/oscar/blob/003308d3747846580fabc646e887cdf4e662f94d/moscato/model.py)
- [Raf's tracking implementation](https://github.com/chefOST/chefOST-ml/blob/5bfb5c4699a964d74be6b85e87baa9269d2d3b25/src/object_state_tracking.py)
- [Ego4D prototype](https://github.com/chefOST/ego4d-pseudo-labels/blob/01c287e3910b3b7e00b81af29a5875c613a7114f/README.md)
- [Shared data](https://github.com/chefOST/chefOST-data/blob/6b9fa8ddb66a1fcdf276d29fab63520db9f42312/README.md)
- [Project guide](https://github.com/chefOST/.github-private/blob/6530b4e6d38782676dabcb9632289fe719984d8c/profile/README.md)
