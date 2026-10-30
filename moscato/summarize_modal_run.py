"""Write a readable report for a completed exploratory run (no model execution)."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .baseline import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    run = Path(args.run)
    fm = read(run/'feature_manifest.json')
    prediction = read(run/'predictions/manifest.json')
    evaluation = read(run/'scoring/evaluation.json')
    with np.load(run/'predictions/probabilities.npz', allow_pickle=False) as data:
        state, action = data['state'], data['action']
    labels = prediction['mapping']['state_labels']
    actions = prediction['mapping']['action_labels']
    counts = Counter(labels[int(i)].strip() for i in state.argmax(1))
    action_counts = Counter(actions[int(i)] for i in action.argmax(1))
    lines = [
        '# VidOSC real-video exploratory result', '',
        '**Provisional single-clip transfer result; not an official benchmark score.**', '',
        f"Video: `{fm['video_id']}`; target annotation object: `{fm['object']}`.",
        f"Excerpt: {fm['clip_start_seconds']:.3f}–{fm['clip_end_seconds']:.3f} seconds.", '',
        f"Processed {len(state)} intervals on {fm['prediction_device']} in {fm['elapsed_seconds']:.1f} seconds "
        '(excluding upload and image build).',
        f"Object features present for {fm['object_windows']} intervals; "
        f"{fm['missing_object_windows']} intervals used the upstream-supported zero-object fallback.", '',
        '## Provisional comparison with MOSCATO', '',
        '| Metric | Value |', '| --- | ---: |',
        f"| Frame micro F1-Max (threshold sweep on this clip) | {100*evaluation['f1_max']['f1']:.2f}% |",
        f"| Selected threshold | {evaluation['f1_max']['threshold']:.1f} |",
        f"| Macro average precision (positive GT classes) | {100*evaluation['map']:.2f}% |",
    ]
    for overlap, value in evaluation['segment_f1'].items():
        lines.append(f"| Segment F1 at IoU {overlap}, confidence 0.5 | {100*value['f1']:.2f}% |")
    lines += ['', f"Evaluated annotation entries: {evaluation['evaluated_frames']}; "
              f"empty entries skipped: {evaluation['skipped_empty_frames']}.",
              f"GT labels without a source-label mapping: {', '.join(evaluation['unmapped_gt_states']) or 'none'}.", '',
              '## Most frequent source predictions', '',
              'These are per-second argmax outputs. Counts and probabilities are not accuracy.', '',
              '| Source state | Seconds |', '| --- | ---: |']
    lines += [f'| {name} | {n} |' for name,n in counts.most_common()]
    lines += ['', '| Source action | Seconds |', '| --- | ---: |']
    lines += [f'| {name} | {n} |' for name,n in action_counts.most_common()]
    lines += ['', '## Limits', '',
              '- Crop extraction is a documented adaptation, not verified against the original training pipeline.',
              '- Object crops can cover tools/boards rather than the annotated target; no manual GT-based corrections were made.',
              '- Class indices are reconstructed from the released full multitask vocabulary.',
              f'- Annotation FPS: {evaluation["timing"]["annotation_fps"]}; timing verified: {evaluation["timing_verified"]}.',
              '- Missing label mappings remain misses for thresholded F1. Constant-zero scores can still have nonzero AP due to ties.',
              '- F1-Max uses the best threshold on this clip; segment F1 uses the fixed 0.5 threshold.',
              '- Raw softmax predictions are used without the original ordering decoder.', '',
              '## Files', '',
              '- `predictions/predictions_by_second.csv`: named state and action predictions and confidence.',
              '- `scoring/evaluation.json`: detailed provisional metrics and provenance.',
              '- `scoring/per_frame.csv`: aligned GT and predictions for inspection.',
              '- `feature_manifest.json`: crop decisions, sampling, input identities, and model metadata.',
              f'- `combined_features.pth.tar`: the actual {len(state)} × 1536 model input.',
              '- `raw_probabilities.npz`: original Modal state and action probabilities.', '']
    path = run/'RESULTS.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__ == '__main__':
    main()
