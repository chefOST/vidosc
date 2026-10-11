# First real-video VidOSC run on Modal

This runs the pretrained ChangeIt multitask VidOSC checkpoint on the 107-second
salad excerpt. It is an **exploratory transfer experiment**, not a reproduction
of VidOSC's published benchmark or the complete MOSCATO benchmark. No training,
Claude, hand-corrected crops, or test-label-guided mapping is used.

## Run

From the repository root, with the assets documented in MOSCATO_BASELINE.md:

```bash
.venv/bin/python -m modal run scripts/modal_clip.py
```

The Modal worker requests four CPU cores, 16 GiB memory, at most one container,
zero inference retries, and a one-hour timeout. It uses Modal credits and stops
after the run. The remote image inherits the detector environment, adding
InternVideo and VidOSC at pinned source commits. The video, previously extracted
global features, InternVideo weights from Downloads, and VidOSC weights are
uploaded. Ground-truth annotations are not uploaded. Results return to a new
UTC-timestamped folder under `runs/modal-clip/`.

## Declared crop and missing-feature policy

The released VidOSC code supplies the feature loader but not the complete crop
extraction implementation. This pilot therefore makes these choices explicit:

1. Use the existing eight-frame-per-second sampling grid, frames [1469,4679).
2. Detect hands and objects on the fifth sampled frame of each interval.
3. At confidence >0.5, associate contacting hands with objects using the original
   detector's endpoint/nearest-object-center rule (`filter_object`). Contact >0
   matches that upstream helper; this is not a vegetables-specific selection.
4. Take the union of associated object boxes and apply that fixed crop to all
   eight sampled frames in the interval, with no extra padding or propagation.
5. Encode valid cropped clips with the same InternVideo visual model and demo
   preprocessing; L2-normalize the 768-dimensional vectors.
6. Leave missing object intervals as zero vectors. This is explicitly supported
   by `ChangeItFeatDataset.load_feat` in upstream `dataset_changeit.py` (lines
   166–178): only provided object feature indices overwrite a zero-initialized
   array. Our checkpoint adapter still refuses silently padding arbitrary input.
7. Concatenate global then object features to obtain [107,1536] and run the real
   checkpoint's state/action heads. As in the original ChangeIt loader, zero-pad
   to 900 time steps without an attention mask, then retain only the 107 valid
   outputs. Apply softmax without an ordering decoder.

Do not infer exact extraction parity from the matching input dimensions. The
crop policy, detector variant, sampling, normalization, and old-op compatibility
have not all been validated against the original training feature generator.
The crops may cover tools or the board rather than the annotated vegetables.
All detected boxes and chosen crops are retained in the output manifest.

## Labels and MOSCATO mapping

`configs/changeit_moscato_alias_mapping.json` is generated from the released
44-row `state_description_vocab.csv` (Drive ID
`1ov1wwrTYkmJYB8URUa-wntOEqbF4gXJj`). Label ordering follows upstream
`build_vocab`: background, initial states in CSV order, then end states, removing
repeats in insertion order. This matches torchtext's documented `vocab` behavior
and reconstructs exactly 52 states and 28 actions. The original training
folder list is not bundled with this run, so ordering is a reconstruction from
released metadata, not an embedded label list from the checkpoint.

The mapping uses only exact canonical MOSCATO labels and its published aliases,
with whitespace trimming for lookup. For example, cut maps to chopped, ground
to grinded, and whisked to beaten. Unmatched labels, including background,
remain unmapped. No probability renormalization or extra guessed synonyms are
applied. Action outputs are reported separately and are not treated as states.

After the remote run finishes, name and map its saved outputs without inference:

```bash
.venv/bin/python -m moscato.import_modal_predictions \
  --run runs/modal-clip/REPLACE_WITH_RUN_DIRECTORY \
  --mapping configs/changeit_moscato_alias_mapping.json \
  --state-dict data/moscato/vocabulary/state_dict.json \
  --output runs/modal-clip/REPLACE_WITH_RUN_DIRECTORY/predictions
```

## Provisional scoring

The annotation file contains 8080 entries for 8080 raw video frames. Visual
spot checks support alignment, but exact event boundaries have not been
formally verified. Diagnostic scoring explicitly assumes zero-based raw-frame
alignment at 30 FPS. Timing metadata remains `verified: false`; it is not
silently promoted to verified just to make the evaluator run.

```bash
.venv/bin/python -m moscato.baseline evaluate \
  --run runs/modal-clip/REPLACE_WITH_RUN_DIRECTORY/predictions \
  --annotations data/moscato/annotations/gt_annotations_cmu.json \
  --state-dict data/moscato/vocabulary/state_dict.json \
  --timing configs/s29_provisional_timing.json \
  --allow-unverified-timing \
  --output runs/modal-clip/REPLACE_WITH_RUN_DIRECTORY/scoring
```

The opt-in flag marks scores provisional; default scoring still requires
verified timing. Full MOSCATO vocabulary is retained and unmapped positive
labels remain misses for thresholded F1. Mean AP uses standard sklearn ranking
semantics: a constant zero score can receive the class prevalence as AP due to
ties, so nonzero AP for an unmapped class does not show recognition ability.
F1-Max selects the best threshold on this clip for reporting; it is not a
held-out threshold estimate. Segment F1 uses the fixed 0.5 threshold.

The named prediction CSV gives per-second source state/action argmax labels
and their probabilities. These probabilities are model confidence, not an
accuracy score. The evaluation JSON contains provisional comparison metrics.

Create the readable report after scoring:

```bash
.venv/bin/python -m moscato.summarize_modal_run \
  --run runs/modal-clip/REPLACE_WITH_RUN_DIRECTORY
```

It writes `RESULTS.md` beside the raw outputs. Large tensors, raw predictions,
and video remain local/ignored; the code, fixed label mapping, and explicitly
unverified timing assumption are versioned on `pritika/main`.

## Completed result (October 11, 2026)

The run directory is `runs/modal-clip/20261011T184132341252Z/`.
The combined input is [107,1536], with 91 object-feature rows and 16 zero rows.
Extraction plus the initial inference took 935.9 seconds on Modal CPU. A brief
second Modal pass applied the original 900-step padding; the initial unpadded
probabilities are retained separately as `raw_probabilities_unpadded.npz`.
The current full-run script includes padding directly for future runs.

The final outputs contain [107,52] state and [107,28] action probabilities.
The source-state argmax is background for 67 seconds and peeled for 40 seconds.
Provisional frame micro F1-Max is 33.73% (selected threshold 0.5); macro AP is
51.52%; segment F1 is 33.33% at IoU 0.1, 0.25, and 0.5. These are not accuracy
percentages or official benchmark results. See the limitations above and
[the saved results summary](docs/s29_exploratory_results.md).

Remote runs:
- Feature extraction and initial inference: https://modal.com/apps/pritika2404/main/ap-qSGZYKNZTeYloOCdWKHZcu
- Final padded inference: https://modal.com/apps/pritika2404/main/ap-yBiZuGoy7Xbhk8sXXY8bys

31 relevant local tests pass, including explicit temporal-padding length and
provisional-scoring opt-in checks. Returned tensor shapes, normalization,
zero-feature masks, probability sums, and unchanged global features were checked.
