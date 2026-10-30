"""Metric, timestamp, leakage and real-checkpoint compatibility regression tests."""
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moscato.baseline import align_scores, project_scores, read, digest, evaluate_run
from moscato.baseline_metrics import evaluate_scores, segment_counts, segments


def test_alignment_uses_annotation_clock_and_noninteger_clip_offset():
    indices, bins, scores = align_scores(np.array([[.1], [.9]]), [0, 1], .15, 1.65, 10, 20)
    assert indices.tolist() == list(range(2, 17))
    assert bins.tolist() == [0] * 10 + [1] * 5
    assert scores[-1, 0] == .9


def test_alignment_does_not_treat_30fps_video_as_10fps_annotations():
    indices, bins, _ = align_scores(np.eye(2), [0, 1], 10, 12, 30, 360)
    assert indices.tolist() == list(range(300, 360))
    assert bins.tolist() == [0] * 30 + [1] * 30


@pytest.mark.parametrize('starts,end', [([1, 2], 3), ([0, 0], 2), ([0, 2], 2), ([0, float('nan')], 3)])
def test_bad_interval_metadata_rejected(starts, end):
    with pytest.raises(ValueError):
        align_scores(np.eye(2), starts, 0, end, 10, 100)


def test_mapping_preserves_probability_mass_and_multilabel_targets():
    actual = project_scores([[.2, .3, .5]], {'0':['raw'], '1':['raw', 'cut'], '2':[]}, ['raw', 'cut'])
    np.testing.assert_allclose(actual, [[.5, .3]])


@pytest.mark.parametrize('mapping', [{'0':['raw']}, {'0':['invented'], '1':[]}, {'0':['raw','raw'], '1':[]}])
def test_invalid_mappings_rejected(mapping):
    with pytest.raises(ValueError):
        project_scores([[.4,.6]], mapping, ['raw'])


def test_perfect_metrics_and_empty_gap_policy():
    truth = [[1,0], [0,0], [1,0], [0,1]]
    result = evaluate_scores(truth, [[.9,.1], [.99,.99], [.8,.1], [.1,.9]])
    assert result['map'] == 1
    assert result['f1_max']['f1'] == 1
    assert result['skipped_empty_frames'] == 1
    assert result['segment_f1']['0.5'] == {'tp':3,'fp':0,'fn':0,'f1':1}


def test_multilabel_missing_state_counts_as_false_negative():
    result = evaluate_scores([[1,1]], [[.9,0]])
    assert result['f1_max']['recall'] == .5
    assert result['f1_max']['f1'] == pytest.approx(2/3)


def test_single_state_metric_does_not_count_true_negatives_as_hits():
    # Every annotated row has a positive label; all are missed.
    assert evaluate_scores([[1],[1]], [[0],[0]])['f1_max']['f1'] == 0


def test_iou_threshold_and_one_to_one_matching():
    truth = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])
    pred = np.array([[0],[0],[1],[1],[1],[1],[0],[0]])
    valid = np.ones(8,dtype=bool)
    assert segment_counts(truth,pred,valid,.25)['tp'] == 1
    assert segment_counts(truth,pred,valid,.5)['tp'] == 0


def test_no_annotations_and_nan_fail():
    with pytest.raises(ValueError,match='No annotated'):
        evaluate_scores([[0,0]], [[.1,.2]])
    with pytest.raises(ValueError,match='finite'):
        evaluate_scores([[1,0]], [[float('nan'),.2]])


def test_lfs_pointer_error(tmp_path):
    p=tmp_path/'gt.json'; p.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError,match='Git LFS pointer'):
        read(p)


def test_scoring_real_schema_and_integrity_checks(tmp_path):
    run=tmp_path/'run'; run.mkdir()
    vocabulary=tmp_path/'vocab.json'
    vocabulary.write_text(json.dumps({'s2i':{'whole':'0','cut':'1'},'all2one':{'sliced':'cut'}}))
    gt=tmp_path/'gt.json'
    gt.write_text(json.dumps({'v':{'state':{'onion':[['whole'],['whole'],['sliced'],['cut']]}}}))
    timing=tmp_path/'timing.json'
    timing.write_text(json.dumps({'verified':True,'evidence':'fixture','video_id':'v','video_sha256':'fixture','annotation_fps':2,'annotation_count':4}))
    np.savez(run/'probabilities.npz',mapped=np.array([[.1,.9],[.9,.1]])) # sorted vocab cut,whole
    manifest={'ground_truth_used_for_inference':False,'probabilities_sha256':digest(run/'probabilities.npz'),
        'state_dict_sha256':digest(vocabulary),'feature_manifest':{'video_id':'v','object':'onion','video_sha256':'fixture',
        'interval_starts_seconds':[0,1],'clip_start_seconds':0,'clip_end_seconds':2},
        'vocabulary':['cut','whole'],'method':'fixture','mapping':{'source_index_to_states':{'0':['cut'],'1':['whole']}}}
    (run/'manifest.json').write_text(json.dumps(manifest))
    args=Namespace(run=str(run),state_dict=str(vocabulary),annotations=str(gt),timing=str(timing),threshold=.5,output=str(tmp_path/'scores'))
    before=digest(run/'probabilities.npz')
    evaluate_run(args)
    assert read(tmp_path/'scores/evaluation.json')['f1_max']['f1']==1
    assert digest(run/'probabilities.npz')==before
    provisional = read(timing)
    provisional['verified'] = False
    timing.write_text(json.dumps(provisional))
    args.output = str(tmp_path/'diagnostic')
    with pytest.raises(ValueError, match='allow-unverified-timing'):
        evaluate_run(args)
    args.allow_unverified_timing = True
    evaluate_run(args)
    diagnostic = read(tmp_path/'diagnostic/evaluation.json')
    assert diagnostic['provisional'] is True
    assert diagnostic['timing_verified'] is False
    np.savez(run/'probabilities.npz',mapped=np.zeros((2,2)))
    with pytest.raises(ValueError,match='changed after inference'):
        evaluate_run(args)


ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'checkpoints/changeit/state_best.ckpt'
UPSTREAM=ROOT.parent/'vidosc-upstream'


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
