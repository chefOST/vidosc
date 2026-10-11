from pathlib import Path
import pytest
from moscato.changeit_mapping import build_mapping

ROOT = Path(__file__).resolve().parents[1]


def test_released_mapping_reconstructs_index_order_and_aliases():
    csv = ROOT / 'data/moscato/vocabulary/changeit_state_description_vocab.csv'
    ckpt = ROOT / 'checkpoints/changeit/state_best.ckpt'
    if not csv.exists() or not ckpt.exists():
        pytest.skip('Local released assets unavailable')
    m = build_mapping(csv, ROOT / 'data/moscato/vocabulary/state_dict.json', ckpt)
    assert len(m['state_labels']) == 52 and len(m['action_labels']) == 28
    assert m['state_labels'][18] == 'dirty '
    assert m['source_index_to_states']['18'] == ['dirty']
    assert m['source_index_to_states']['49'] == ['chopped']
    assert m['source_index_to_states']['0'] == []
    assert m['source_index_to_states']['1'] == []  # unpeeled is not peeled
    assert m['ground_truth_used_to_define_mapping'] is False
