# VidOSC real-video exploratory result

**Provisional single-clip transfer result; not an official benchmark score.**

Video: `S29_Salad_7150991-1`; target annotation object: `vegetables`.
Excerpt: 48.967–155.967 seconds.

Processed 107 intervals on Modal CPU in 935.9 seconds (excluding upload, image build, and the brief final padded inference pass).
Object features present for 91 intervals; 16 intervals used the upstream-supported zero-object fallback.

Final predictions use the original ChangeIt loader's 900-step zero padding,
retaining the first 107 outputs. The initial unpadded outputs are preserved separately.

## Provisional comparison with MOSCATO

| Metric | Value |
| --- | ---: |
| Frame micro F1-Max (threshold sweep on this clip) | 33.73% |
| Selected threshold | 0.5 |
| Macro average precision (positive GT classes) | 51.52% |
| Segment F1 at IoU 0.1, confidence 0.5 | 33.33% |
| Segment F1 at IoU 0.25, confidence 0.5 | 33.33% |
| Segment F1 at IoU 0.5, confidence 0.5 | 33.33% |

Evaluated annotation entries: 3183; empty entries skipped: 27.
GT labels without a source-label mapping: mixed, separated.

## Most frequent source predictions

These are per-second argmax outputs. Counts and probabilities are not accuracy.

| Source state | Seconds |
| --- | ---: |
| background | 67 |
| peeled | 40 |

| Source action | Seconds |
| --- | ---: |
| peeling | 64 |
| background | 43 |

## Limits

- Crop extraction is a documented adaptation, not verified against the original training pipeline.
- Object crops can cover tools/boards rather than vegetables; no manual GT-based corrections were made.
- Class indices are reconstructed from the released full multitask vocabulary.
- Scoring assumes one annotation per raw frame at 30 FPS; formal timing verification is incomplete.
- Missing label mappings remain misses for thresholded F1. Constant-zero scores can still have nonzero AP due to ties.
- F1-Max uses the best threshold on this clip; segment F1 uses the fixed 0.5 threshold.
- Raw softmax predictions are used without the original ordering decoder.

## Files

- `predictions/predictions_by_second.csv`: named state and action predictions and confidence.
- `scoring/evaluation.json`: detailed provisional metrics and provenance.
- `scoring/per_frame.csv`: aligned GT and predictions for inspection.
- `feature_manifest.json`: crop decisions, sampling, input identities, and model metadata.
- `combined_features.pth.tar`: the actual 107 × 1536 model input.
- `raw_probabilities.npz`: original Modal state and action probabilities.

Local output directory: `runs/modal-clip/20261011T184132341252Z/`.
