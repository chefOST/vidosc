# Paper evaluation audit — 2026-10-11

No additional paid inference was launched. Current uploaded scores remain exploratory.

## Timing evidence

- Sequentially decoded the complete local S29 Salad AVI: 8,080 frames.
- Video metadata: 30 FPS and 269.333 seconds; ground truth has 8,080 entries.
- A 10 Hz interpretation would span 808 seconds and cannot represent this entire
  local video without another transformation or source version.
- Inspected a contact sheet at annotation boundaries 1469, 2908, 3648, 4562,
  5315, 5923, 6116, 6800, 7193, 7445, 7800, and 7943. At frame 3648 the person
  is peeling cucumber; at 7445 a bowl of salad and dressing-related activity are
  visible. This supports broad temporal correspondence, not exact boundary accuracy.
- The published repository's annotation-format example says 10 FPS. The local
  metadata uses a separate shorter feature timeline: 2,694 feature entries for
  S29 versus 8,080 annotation entries. Across 29 of 30 videos, feature lengths are
  within one entry of floor(annotation_count/3)+1. S13 Brownie is a larger outlier
  (18,131 annotation entries; 6,838 features). This suggests separate clocks but
  does not prove their provenance or an official conversion rule.
- Keep the current timing file unverified until provenance/offset checks are resolved.

Local evidence: `runs/timing-audit/clock_audit.json` and
`runs/timing-audit/salad-boundaries.jpg`. These are ignored generated artifacts.

## Scope discrepancy

The paper and release README describe 36 CMU test videos, whereas the local
ground-truth JSON contains 30. Earlier references to "all 30 MOSCATO videos" were
too broad: these are the locally available CMU annotations, not the full MOSCATO
benchmark, which also includes EGTEA and EgoPER. Reconcile the release version and
missing videos before claiming the full published CMU split.

## Protocol to freeze before final runs

1. Identify the released annotation version, test IDs, and source-video versions.
   Resolve the 10/30 FPS discrepancy and S13 outlier; record per-video clock/offset.
2. Define the task explicitly: a ChangeIt-trained VidOSC transfer baseline.
   The existing model produces one active-object state stream, while MOSCATO
   evaluates multiple objects. Do not duplicate that stream across objects and
   describe it as object-specific recognition. Choose/document object association
   or define a restricted single-object evaluation with its own comparison scope.
3. Freeze source label order, alias mapping, object selection, feature extraction,
   and treatment of unsupported labels before final scores are inspected.
4. Implement pooled dataset metrics and validate against the released evaluator
   if obtainable. The paper pools TP/FP/FN over the full test set for F1. Its
   F1-Max protocol sweeps one threshold globally over 0.1,...,0.9; a single test-set
   sweep is the reported benchmark metric, not a calibrated deployment threshold.
5. Audit masks: current scoring skips empty per-object state lists. Confirm how
   the original benchmark scores absent/invisible objects and false positives.
6. Do not compare our state-only mAP on one object's positive classes directly
   against the paper's multi-object results. Freeze the object-state category axis
   and missing-class policy before comparison.
7. Run an approved cost-bounded pilot, then the frozen evaluation set; log per-video
   coverage and pooled results to W&B with configuration and file hashes.

Human-created benchmark annotations do not need to be proven error-free to report
research results. A documented, reproducible protocol and precise claims are needed.
Preserve original labels; report any audited disagreements separately.

## Draft clarification for the dataset authors (not sent)

We are evaluating a pretrained VidOSC checkpoint on the released MOSCATO CMU
annotations. Our local gt_annotations_cmu.json contains 30 videos; the paper lists
36 CMU test videos. For S29_Salad_7150991-1, the state/object arrays have 8,080
entries, matching an 8,080-frame, 30 FPS CMU AVI, while files_dict feature_len is
2,694 and the README describes annotation arrays at 10 FPS. Which release and
video version should we use, what conversion/offset maps the ground-truth arrays
to the video and feature timelines, and where can we obtain the complete test ID
list and official evaluator (including visibility masks and mAP class averaging)?

## Sources

- https://github.com/eelhami/iccv25_moscato — annotation examples and split counts.
- https://openaccess.thecvf.com/content/ICCV2025/papers/Zameni_MOSCATO_Predicting_Multiple_Object_State_Change_Through_Actions_ICCV_2025_paper.pdf — Sections 4 and 5.1: human test annotations, split counts and pooled metrics.
