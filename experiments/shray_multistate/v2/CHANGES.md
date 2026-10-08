# What changed, and what improved

The earlier model struggled with unsupported object queries and missing training
coverage. This follow-up broadens human supervision and changes how the scorer
learns from overlapping labels. The selected model improves mean target/video
multi-label F1 by **17.6 percentage points**, from 35.5% to 53.1%, when both old
and new checkpoints are evaluated on the same four new recordings.

![Same-recording F1 comparison](comparison.png)

## Implementation walkthrough

The prediction path remains:

```text
Video → sampled frames → frozen CLIP image features → frozen VidOSC context
                                                        ↓
Object-name embedding ─────────────────────────→ trained text scoring layer
Candidate-description embeddings ──────────────→ scores for each candidate
                                                        ↓
                                      smoothing + threshold → overlapping intervals
```

There are no hosted LLM calls in this experiment. CLIP's text encoder embeds
supplied descriptions; it does not generate candidate descriptions. VidOSC and
CLIP are pretrained components. This contribution trains the scoring layer,
not a new foundation model from scratch.

| Change | Implementation | Why it was tested |
|---|---|---|
| Broader human training | `prepare_experiment.py`, `protocol.json` | Nine recordings / 13 target-video units now cover vegetables, eggs, egg mixture, bread slices, cheese and batter. Previously exposed tests become development data. |
| Align training with evaluation time | `run_accuracy.py:prepare_units` | Within each two-second prediction bin, use the fraction of known human-labelled frames carrying each state. Weight by known-frame coverage. |
| Train overlapping states independently | `run_accuracy.py:fit_neural` | Unweighted binary cross entropy lets several states be active at once. Each object/video receives equal weight. |
| Fixed object-query scaling | `models.py:StableQueryModel.initialize_statistics` | Avoid deriving per-coordinate query scaling from a small set of training object names. |
| Query-relative text centering | `models.py:StableQueryModel.forward` | Compare projected video features with normalized candidate-minus-object-query embeddings. Scores remain independent of candidate order/count. |
| Add controls | `run_accuracy.py:fit_prototypes`, `select_blend` | Compare learned visual/temporal prototypes, constant training-frequency guesses and prior blends. |
| Preserve selection and evaluation evidence | `selection-lock.json`, `prediction-lock.json`, `audit_results.py` | Select using validation only; save/hash all test predictions before fresh-label export/scoring. |

For example, if a two-second bin has 60 known frames and `chopped` applies to 45,
the training target for that state is 0.75. A second state can also have a positive
target. That represents changing/overlapping human labels within a bin; it is
not a claim that the model's output is a calibrated probability.

Both neural variants use the same new training data, loss and 100-epoch budget.
They retain frozen CLIP/VidOSC features. Validation selects epochs, thresholds,
smoothing and the overall method; `stable` won validation before test scoring.
The selected threshold is 0.25 with five-sample smoothing, and the selected
checkpoint is epoch 50. All 77 evaluation candidates are supplied vocabulary
entries, not states discovered automatically by the model.

## Direct comparison on the same test

| Method | Mean target/video F1 | Highest-label hit |
|---|---:|---:|
| Unchanged old selected checkpoint | 35.5% | 42.9% |
| Constant training-frequency prior | 46.9% | 57.0% |
| Original architecture retrained with new data/loss | 56.2% | 72.6% |
| Validation-selected revised architecture | **53.1%** | **72.9%** |
| Visual/temporal prototypes | 54.2% | 59.0% |

F1 penalizes extra and missed labels. Highest-label hit checks only whether the
single top-scoring candidate is among the human labels. Neither is the percentage
of frames with a completely correct state set: that figure is only **1.5%** for
the selected model. Segment F1 at IoU 0.5 is **26.5%**.

The unchanged old model was evaluated again on these new recordings. Its 35.5%
here is the valid before/after reference. The previously reported 30.3% came
from a different test set and must not be used to calculate this improvement.

## What the evidence does and does not establish

The overall new training setup improves the old checkpoint on this test. However,
the revised architecture scores 3.2 points below the retrained original
architecture. This run does not establish a benefit from the query-scaling or
text-centering changes. The data, supervision coverage and loss change together;
there is no isolated estimate of each one's contribution to the improvement.

The selected model beats the prior by 6.2 points on average, but a recording-level
bootstrap gives an exploratory 95% interval of -0.6 to +13.2 points. With only
four recordings the advantage over the constant baseline remains uncertain.
Brownie eggs regress from 75.4% to 58.3% F1 even though the overall average rises.
Pizza cheese rises from 0.0% to 59.0%, but its constant prior scores 59.5%:
that improvement does not by itself demonstrate visual recognition.

The fresh test comprises four separate subjects and six target/video units in
the same CMU kitchen. Recipes and target categories are represented in training.
This does not measure unseen-object/state or cross-kitchen generalization. Old
test recordings used in development are explicitly listed in the protocol.
Split planning inspected source annotations, so this is a custom pilot rather
than a fully blinded official benchmark. The next model-selection experiment
needs new held-out recordings.

Michelle and this contribution share the semantic-state goal. Her pipeline uses
Claude to describe VidOSC phases; this contribution learns state scores from
video/temporal features and supplied descriptions. No same-test comparison with
Michelle's pipeline has been performed, so superiority is not established.

## Why CLIP is cheaper in this setup

Our CLIP extractor processes **one 224×224 image per two-second sample**, batched
in groups of eight. Our SlowFast comparison processes **32 frames at 256×256 per
window**, with an eight-frame slow pathway and a 32-frame fast pathway. This adds
decoding/preprocessing and video-network work. These counts are not a 32× speedup
measurement: model architecture, batching, precision and hardware also matter.

We actually ran the SlowFast extraction and trained CLIP+SlowFast fusion in the
earlier pilot. [Saved paired extraction timings](results/extraction-timing.json)
cover 21 shared videos: SlowFast is slower on 17, with a **1.34× median time ratio**.
For `S07_Salad_7150991-2095`, CLIP extraction takes 93.5 seconds and SlowFast
takes 169.3 seconds. For `S20_Pizza_7150991-180`, they are almost equal:
239.3 versus 237.0 seconds. These are ordinary saved wall-clock extraction
timers, with different batching/precision and some large delays; they are not
a controlled benchmark or a universal speed claim. No pairs were discarded.
`compare_extraction.py` reproduces this summary from existing metadata.

[CLIP](https://github.com/openai/CLIP) provides matching image and text embedding
spaces, fitting candidate-description scoring directly. [SlowFast](https://ai.meta.com/research/publications/slowfast-networks-for-video-recognition/)
uses two temporal pathways to capture appearance and motion. SlowFast can retain
motion information absent from a single CLIP frame; it is not universally less
efficient or less accurate. Our VidOSC encoder adds context across sampled CLIP
frames, but cannot recover every motion between them. In the earlier pilot,
trained CLIP+SlowFast fusion scored **31.6% F1**, versus **32.5%** for trained
CLIP alone on the same test (see the [original report](../broader/REPORT.md)).
The previous SlowFast fusion comparison did not improve the earlier test; v2 does not rerun that
comparison or establish which backbone is best under the new training setup.

## Verification and artifacts

- 23 existing tests plus four follow-up tests pass.
- The trained scorer works with 3, 5 and 77 supplied candidates and preserves
  shared scores under subset/reordering (maximum observed difference 2.61e-8).
- Saved predictions reproduce; frozen training-code/model/prediction hashes and
  all previous experiment artifact hashes remain intact.
- JSON metrics, histories, locks, code and this chart are versioned. Large
  checkpoints, tensor predictions, feature caches and videos remain local;
  they are excluded from Git. See the setup requirements in [README.md](README.md).

Full per-state, per-target and interval results are in [REPORT.md](REPORT.md) and
[results/evaluation.json](results/evaluation.json). The chart and narrative are
post-evaluation reporting; trained models and locked selection were not changed.
