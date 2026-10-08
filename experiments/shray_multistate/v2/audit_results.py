"""Post-evaluation audit; never refits or selects a model from test scores."""
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'broader'))
from protocol import WORKSPACE, DATA, read, write
from features import feature_path
from models import StableQueryModel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(8)
    results = read(HERE / 'results/evaluation.json')
    lock = read(HERE / 'results/selection-lock.json')
    selected = lock['winner']
    for name, digest in lock['code_sha256'].items():
        assert sha(HERE / name) == digest, name
    for name, digest in lock['models_sha256'].items():
        assert sha(HERE / 'results' / name) == digest, name
    pred_path = HERE / 'results/fresh-test-predictions.pt'
    assert sha(pred_path) == read(HERE / 'results/prediction-lock.json')['sha256']
    for path, digest in read(HERE / 'prior-artifact-hashes.json').items():
        assert sha(WORKSPACE / path) == digest, path
    rows = results[selected]['units']
    videos = sorted({key.split('::')[0] for key in rows})
    comparisons = {}
    for reference in ('legacy', 'prior', 'original'):
        differences = {v: [r['micro_f1'] - results[reference]['units'][key]['micro_f1']
                            for key, r in rows.items() if key.startswith(v + '::')]
                       for v in videos}
        # Resample whole recordings. Multiple targets from one video stay together.
        samples = [np.mean([d for v in sample for d in differences[v]])
                   for sample in itertools.product(videos, repeat=len(videos))]
        comparisons[reference] = {
            'mean_difference': float(np.mean([d for ds in differences.values() for d in ds])),
            'paired_recording_bootstrap_95_percentile_interval': np.quantile(samples, [.025, .975]).tolist(),
            'recordings_improved': int(sum(np.mean(ds) > 0 for ds in differences.values())),
            'recordings': len(videos),
        }
    table = []
    for key, row in rows.items():
        table.append({'unit': key, **{method: results[method]['units'][key]['micro_f1']
                                     for method in ('legacy', 'prior', 'original', selected)},
                      'missed_states': row['missed_state_names'], 'extra_states': row['extra_state_names']})
    # Exercise the trained network on real cached features with 3, 5 and all 77 candidates.
    text = torch.load(DATA / 'text_embeddings.pt', map_location='cpu', weights_only=True)
    checkpoint = torch.load(HERE / 'results/stable.pt', map_location='cpu', weights_only=True)
    model = StableQueryModel().eval()
    model.head.load_state_dict(checkpoint['head'])
    model.encoder.load_state_dict(checkpoint['encoder'])
    key = 'S17_Salad_7150991-822::vegetables'
    bundle = torch.load(feature_path(key.split('::')[0], 'clip'), map_location='cpu', weights_only=True)
    query = text['queries']['vegetables']; candidates = text['embeddings']['vegetables']
    # Locally generated, hash-verified artifact includes NumPy prediction arrays.
    prediction = torch.load(pred_path, map_location='cpu', weights_only=False)
    checks = {}
    with torch.inference_mode():
        full = model(bundle['visual'], query, candidates, temporal=bundle['temporal']).sigmoid()
        for size in (3, 5):
            indices = list(reversed(range(size)))
            subset = model(bundle['visual'], query, candidates[indices], temporal=bundle['temporal']).sigmoid()
            torch.testing.assert_close(subset, full[:, indices], rtol=1e-5, atol=1e-6)
            checks[str(size)] = float((subset-full[:, indices]).abs().max())
    np.testing.assert_allclose(full.numpy(), prediction['predictions']['stable'][key], rtol=1e-5, atol=1e-6)
    audit = {'paired_comparisons': comparisons, 'per_unit_f1': table,
             'all_candidate_count': len(text['states']), 'trained_subset_max_abs_errors': checks,
             'saved_prediction_reproduced': True, 'frozen_code_models_predictions_and_old_artifacts_verified': True,
             'uncertainty_note': 'Only four recordings. Percentile intervals are exploratory and cannot establish broad reliability.'}
    write(HERE / 'results/audit.json', audit)
    deliver = WORKSPACE / 'outputs/shray-accuracy-v2'
    write(deliver / 'audit.json', audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ['legacy', 'prior', 'original', 'stable', 'prototype']
    labels = ['Old checkpoint', 'Constant prior', 'Retrained original', 'Selected stable', 'Prototype']
    values = [100*results[m]['aggregate']['macro_over_object_video_units']['micro_f1'] for m in methods]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    bars = ax.barh(labels, values, color=['#9b9b9b', '#bc9759', '#52758c', '#27755d', '#88749d'])
    ax.bar_label(bars, fmt='%.1f%%', padding=5)
    ax.set(xlim=(0, 70), xlabel='Mean target/video multi-label F1 (%)',
           title='Same four new recordings · six annotated targets')
    ax.invert_yaxis(); ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(deliver / 'comparison.png', dpi=160)
    plt.close(fig)
    lines = ['\n## Audit and limitations\n',
             'Stable was selected on validation and remains the selected model. The retrained original architecture scored 56.2% F1 on the fresh test, versus stable’s 53.1%. The query-scaling/centering changes therefore did not improve test F1 in this run. Gains over the old model cannot be attributed to those changes: the training coverage and loss also changed.\n',
             '| Target/video | Old | Prior | Retrained original | Selected stable |',
             '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['unit']} | {100*row['legacy']:.1f}% | {100*row['prior']:.1f}% | {100*row['original']:.1f}% | {100*row[selected]:.1f}% |")
    lines += ['\nPaired percentile bootstrap intervals resample whole recordings, keeping targets from the same video together. With four recordings these intervals are exploratory.']
    for reference, comparison in comparisons.items():
        lo, hi = comparison['paired_recording_bootstrap_95_percentile_interval']
        lines.append(f"- Stable minus {reference}: {100*comparison['mean_difference']:+.1f} percentage points; exploratory 95% interval [{100*lo:+.1f}, {100*hi:+.1f}].")
    metric = results[selected]['aggregate']['macro_over_object_video_units']
    segment = results[selected]['aggregate']['segment_f1_macro_over_units']['0.5']
    lines += [f"\nThe highest-scoring label is in the human label set {100*metric['top1_frame_hit_accuracy']:.1f}% of evaluated time, but the complete predicted label set matches only {100*metric['exact_state_set_accuracy']:.1f}%. Segment F1 at IoU 0.5 is {100*segment:.1f}%. The model still misses states and predicts extra states; a high top-one hit rate does not mean the full state timeline is correct.",
              '\nF1 balances correct labels, extra labels and missed labels. It is not the percentage of frames that are completely correct. Frames without known human labels are excluded. Candidates are supplied, not discovered.',
              '\nThe trained scorer reproduced saved predictions and preserved shared candidate scores when using 3, 5 and 77 supplied candidates. This verifies software behavior, not accuracy on arbitrary new state descriptions.',
              '\nFrozen training code, models, saved test predictions and previous experiment artifacts passed SHA-256 integrity checks. Four unit tests passed. Implementation changes are confined to Shray’s own module. Large checkpoints, tensor predictions and datasets remain local and are excluded from Git.']
    addition = '\n'.join(lines) + '\n'
    for path in (HERE / 'REPORT.md', deliver / 'REPORT.md'):
        body = path.read_text(encoding='utf-8').split('\n## Audit and limitations')[0]
        if '![Same-recording F1 comparison]' not in body:
            body += '\n![Same-recording F1 comparison](comparison.png)\n\nSee [CHANGES.md](CHANGES.md) for the implementation walkthrough and interpretation.\n'
        path.write_text(body + addition, encoding='utf-8')
    print(audit, flush=True)


if __name__ == '__main__':
    main()
