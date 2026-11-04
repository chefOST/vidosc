"""Mapping, artifact and checkpoint compatibility checks; no model downloads."""
from pathlib import Path

import numpy as np
import pytest
from moscato.baseline import project_scores, read

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints/changeit/state_best.ckpt'
UPSTREAM = ROOT.parent / 'vidosc-upstream'


def test_mapping_preserves_probability_mass_and_multilabel_targets():
    actual = project_scores([[.2, .3, .5]], {'0':['raw'], '1':['raw', 'cut'], '2':[]}, ['raw', 'cut'])
    np.testing.assert_allclose(actual, [[.5, .3]])


@pytest.mark.parametrize('mapping', [{'0':['raw']}, {'0':['invented'], '1':[]}, {'0':['raw','raw'], '1':[]}])
def test_invalid_mappings_rejected(mapping):
    with pytest.raises(ValueError):
        project_scores([[.4,.6]], mapping, ['raw'])


def test_lfs_pointer_error(tmp_path):
    p=tmp_path/'gt.json'; p.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError,match='Git LFS pointer'):
        read(p)


@pytest.mark.skipif(not CHECKPOINT.exists() or not (UPSTREAM/'model.py').exists(),reason='Official checkpoint/upstream not installed')
def test_downloaded_checkpoint_loads_and_runs_without_training_framework(tmp_path):
    import torch
    from moscato.checkpoint_adapter import predict, read_weights, inspect_weights
    info=inspect_weights(read_weights(CHECKPOINT))
    assert info['family']=='changeit'
    assert info['state_classes']==52
    # Synthetic features ONLY verify adapter shape and finite output, not accuracy.
    f=tmp_path/'synthetic.pth.tar'
    torch.save(torch.zeros(3,info['input_dim']),f)
    state,action,_=predict(CHECKPOINT,f,UPSTREAM)
    assert state.shape==(3,52)
    assert action.shape==(3,28)
    np.testing.assert_allclose(state.sum(axis=1),1,atol=1e-6)
    logits_path = tmp_path / 'logits.npz'
    padded, padded_action, padded_info = predict(CHECKPOINT, f, UPSTREAM, pad_to=5, raw_output=logits_path)
    with np.load(logits_path, allow_pickle=False) as logits:
        for name, expected in [('state', padded), ('action', padded_action)]:
            raw = logits[name + '_logits']
            assert raw.shape == expected.shape
            exp = np.exp(raw - raw.max(axis=-1, keepdims=True))
            np.testing.assert_allclose(exp / exp.sum(axis=-1, keepdims=True), expected, atol=1e-6)
    assert padded.shape == (3,52) and padded_action.shape == (3,28)
    assert padded_info['transformer_timesteps'] == 5
    with pytest.raises(ValueError, match='padding length'):
        predict(CHECKPOINT, f, UPSTREAM, pad_to=2)
    torch.save(torch.zeros(3,768),f)
    with pytest.raises(ValueError,match='No zero padding'):
        predict(CHECKPOINT,f,UPSTREAM)
