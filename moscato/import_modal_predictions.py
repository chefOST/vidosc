"""Name and project saved Modal probabilities without rerunning the model or reading GT."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .baseline import read, write, digest, project_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'mapping', 'state-dict', 'output']:
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    source, out = Path(args.run), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    fm, mapping = read(source/'feature_manifest.json'), read(args.mapping)
    if fm.get('ground_truth_used') is not False or mapping.get('ground_truth_used_to_define_mapping') is not False:
        raise ValueError('Missing no-ground-truth provenance')
    if mapping['checkpoint_sha256'] != fm['vidosc_checkpoint_sha256']:
        raise ValueError('Wrong checkpoint vocabulary')
    if mapping['target_vocabulary_sha256'] != digest(args.state_dict):
        raise ValueError('MOSCATO vocabulary mismatch')
    if fm['feature_sha256'] != digest(source/'combined_features.pth.tar'):
        raise ValueError('Feature fingerprint mismatch')
    with np.load(source/'raw_probabilities.npz', allow_pickle=False) as data:
        state, action = data['state'], data['action']
    starts = fm['interval_starts_seconds']
    if state.shape != (len(starts), len(mapping['state_labels'])):
        raise ValueError('State dimensions do not match vocabulary/timeline')
    if action.shape != (len(starts), len(mapping['action_labels'])):
        raise ValueError('Action dimensions do not match vocabulary/timeline')
    if not np.isfinite(action).all() or (action < 0).any() or not np.allclose(action.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid action probabilities')
    vocabulary = sorted(read(args.state_dict)['s2i'])
    mapped = project_scores(state, mapping['source_index_to_states'], vocabulary)
    out.mkdir(parents=True)
    np.savez(out/'probabilities.npz', state=state, action=action, mapped=mapped)
    write(out/'manifest.json', {
        'method': 'vidosc-exploratory-fixed-alias-transfer',
        'ground_truth_used_for_inference': False, 'feature_manifest': fm,
        'model': fm['model_info'], 'transformer_heads': 4, 'mapping': mapping,
        'vocabulary': vocabulary, 'decoder': 'raw softmax; no ordering',
        'checkpoint_sha256': fm['vidosc_checkpoint_sha256'],
        'features_sha256': fm['feature_sha256'],
        'state_dict_sha256': digest(args.state_dict),
        'probabilities_sha256': digest(out/'probabilities.npz'),
        'source_probabilities_sha256': digest(source/'raw_probabilities.npz'),
        'feature_compatibility_verified': False,
    })
    with (out/'predictions_by_second.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_second', 'video_second', 'state_index', 'state_label',
                         'state_probability', 'action_label', 'action_probability',
                         'has_object_crop'])
        for i,t in enumerate(starts):
            s, a = int(state[i].argmax()), int(action[i].argmax())
            writer.writerow([t, fm['clip_start_seconds']+t, s, mapping['state_labels'][s].strip(),
                             float(state[i,s]), mapping['action_labels'][a], float(action[i,a]),
                             fm['crops'][i]['crop_xyxy_exclusive'] is not None])
    print(f'Named predictions saved to {out}; no annotations read.')


if __name__ == '__main__':
    main()
