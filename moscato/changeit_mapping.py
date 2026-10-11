"""Reconstruct published multitask labels and a conservative MOSCATO alias mapping.

Uses vocabulary metadata only; never reads test annotations or model outputs.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
from .baseline import read, write, digest


def build_mapping(csv_path, state_dict, checkpoint):
    rows = list(csv.DictReader(Path(csv_path).open()))
    states = list(Counter(['background'] + [r['Initial_State'] for r in rows]
                          + [r['End_State'] for r in rows]))
    actions = list(Counter(['background'] + [r['Action'] for r in rows]))
    if (len(rows), len(states), len(actions)) != (44, 52, 28):
        raise ValueError('Expected the released full 44-category multitask vocabulary')
    vocabulary = read(state_dict)
    target = set(vocabulary['s2i'])
    aliases = vocabulary['all2one']
    mapping = {}
    for i, state in enumerate(states):
        normalized = state.strip().lower()
        canonical = aliases.get(normalized, normalized)
        mapping[str(i)] = [canonical] if canonical in target else []
    return {
        'checkpoint_sha256': digest(checkpoint),
        'source_vocabulary_sha256': digest(csv_path),
        'target_vocabulary_sha256': digest(state_dict),
        'source_vocabulary_drive_id': '1ov1wwrTYkmJYB8URUa-wntOEqbF4gXJj',
        'state_labels': states, 'action_labels': actions,
        'source_index_to_states': mapping,
        'ground_truth_used_to_define_mapping': False,
        'rationale': 'Reconstruct all 44 multitask categories in published CSV row order, matching '
                     'ChangeItFeatDataset.build_vocab and torchtext.vocab(Counter(...)) insertion order. '
                     'Preserve source whitespace for index reconstruction; trim/lowercase only for '
                     'MOSCATO canonical-label or published all2one alias lookup. No added semantic '
                     'synonyms, no inference from test annotations, no action-to-state conversion. '
                     'Unmatched states map to []; their probability mass is not renormalized.',
        'source_order_evidence': 'facebookresearch/VidOSC@59575773c97878ea30a0228f28ab00a3ee2f1ea2 '
                                 'dataset_changeit.py:build_vocab; torchtext vocab preserves insertion order',
        'limitation': 'Reconstructed from released training metadata; checkpoint has no embedded labels '
                      'and the original training directory list is not bundled with this run.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['csv', 'state-dict', 'checkpoint', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    write(out, build_mapping(args.csv, args.state_dict, args.checkpoint))
    print(f'Mapping saved without reading test ground truth: {out}')


if __name__ == '__main__':
    main()
