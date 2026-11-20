"""Prediction runners retain outputs while external scoring owns GT comparisons."""
import ast
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class Source:
    src_fps = 30
    videos = ['video']
    gt = {'video': {'state': {'egg': []}}}

    def candidate_states(self, source): return {'egg': ['raw', 'cooked']}
    def candidates_for(self, candidates, source, video, obj): return candidates[obj]
    def num_frames(self, video): return 60
    def objects(self, video): return ['egg']
    def state_prior(self, video, obj): return {'raw': 2, 'cooked': 1}
    def gt_states(self, *args): raise AssertionError('Prediction runner read GT label values')


class Encoder:
    name = 'fixture'
    video_level = True
    logit_scale = 1.0
    def encode_texts(self, labels):
        return np.asarray([[1., 0.] if 'raw' in x else [0., 1.] for x in labels])


@pytest.fixture
def source_setup(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace())
    from pipeline import data, encoders
    monkeypatch.setattr(data, 'Moscato', lambda *a, **k: Source())
    monkeypatch.setattr(data, 'cmu_video', lambda *a: 'video.mp4')
    monkeypatch.setattr(encoders, 'build_encoder', lambda *a: Encoder())
    monkeypatch.setattr(encoders, 'cached_video_embeddings', lambda *a: (np.eye(2), False))
    return tmp_path


def args(**kwargs):
    base = dict(data_root='unused', dataset='CMU', videos=['video'], objects=['egg'], fps=1.,
                encoder='fixture', templates='basic', decoder='argmax', name='fixture', wandb=False,
                state_source='pseudo_train', ckpt='unused', adapt_steps=1, adapt_lr=.001, adapt_keep=.5,
                seed=0, temperature=1., clusters=None, within='prior', smooth=1, adapt_target='frame')
    return SimpleNamespace(**{**base, **kwargs})


def test_state_runner_saves_unthresholded_probabilities(source_setup, monkeypatch):
    from pipeline.states import run
    monkeypatch.setattr(run, 'ROOT', str(source_setup))
    monkeypatch.setattr(run, 'parse_args', lambda: args(method='zeroshot'))
    monkeypatch.setattr(run, 'internvideo_frames', lambda _: 60)
    run.main()
    result = json.loads((source_setup/'runs/fixture/predictions.json').read_text())
    row = result['per_pair']['video/egg']
    assert row['source_frame_indices'] == [0, 30]
    assert row['pred'] == ['raw', 'cooked']
    assert np.asarray(row['probabilities']).shape == (2, 2)
    assert not {'precision', 'recall', 'gt', 'overall'} & result.keys()
    assert 'gt' not in row


def test_embedding_runner_saves_vectors_and_alignment(source_setup, monkeypatch):
    from pipeline.embedding import run
    monkeypatch.setattr(run, 'ROOT', str(source_setup))
    monkeypatch.setattr(run, 'parse_args', lambda: args(methods=['zeroshot']))
    monkeypatch.setattr(run, 'video_size', lambda _: (1, 1, 60))
    run.main()
    result = json.loads((source_setup/'runs/fixture/predictions.json').read_text())
    row = result['methods']['zeroshot']['per_pair']['video/egg']
    saved = np.load(source_setup/'runs/fixture'/row['embedding_file'])
    np.testing.assert_allclose(saved['embeddings'], np.eye(2))
    assert saved['source_frame_indices'].tolist() == [0, 30]
    assert not {'top1', 'mrr', 'precision'} & result['methods']['zeroshot'].keys()


def test_attribute_runner_saves_each_frame_and_probability(source_setup, monkeypatch):
    from pipeline.attributes import run
    schema = SimpleNamespace(attrs={'form': ['raw', 'cooked']},
                             phrases={'form': {'raw': 'raw', 'cooked': 'cooked'}})
    monkeypatch.setattr(run, 'ROOT', str(source_setup))
    monkeypatch.setattr(run, 'parse_args', lambda: args(methods=['zeroshot'], values='all'))
    monkeypatch.setattr(run, 'video_size', lambda _: (1, 1, 60))
    monkeypatch.setattr(run.attributes, 'Schema', lambda _: schema)
    monkeypatch.setattr(run.attributes, 'candidates', lambda *a: {'form': ['raw', 'cooked']})
    monkeypatch.setattr(run.attributes, 'from_states', lambda *a: None)
    run.main()
    result = json.loads((source_setup/'runs/fixture/predictions.json').read_text())
    row = result['methods']['zeroshot']['per_pair']['video/egg']
    assert row['timeline'] == [{'t': 0., 'pred': 'form=raw'}, {'t': 1., 'pred': 'form=cooked'}]
    assert np.asarray(row['attributes']['form']['probabilities']).shape == (2, 2)


def test_discovery_runner_saves_inferred_state_outputs(source_setup, monkeypatch):
    from pipeline.discovery import run
    monkeypatch.setattr(run, 'ROOT', str(source_setup))
    monkeypatch.setattr(run, 'parse_args', lambda: args(feats='siglip', split='all', debias=0.,
                                                      methods=['frame', 'discover'], quiet=True))
    monkeypatch.setattr(run, 'split_videos', lambda *a: ['video'])
    monkeypatch.setattr(run, 'tracks', lambda *a: iter([('video', 'egg', np.eye(2), np.ones(2), [0, 30])]))
    monkeypatch.setattr(run, 'TextBank', lambda *a: lambda *b: np.eye(2))
    monkeypatch.setattr(run, 'outputs', lambda *a: ({'frame': np.eye(2), 'discover': np.eye(2)}, np.array([0, 1])))
    run.main()
    result = json.loads((source_setup/'runs/fixture/predictions.json').read_text())
    assert result['predictions'][0]['methods']['frame']['predicted_states'] == ['raw', 'cooked']
    assert 'ranking' not in result and 'discovery' not in result


def test_training_retains_losses_and_inference_without_benchmark_callbacks():
    for path in ('VidOSC/task.py', 'VidOSC/task_changeit.py'):
        tree = ast.parse((ROOT/path).read_text())
        methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert {'training_step', 'configure_optimizers', 'train_dataloader'} <= methods
        assert not {'validation_step', 'test_step', 'on_validation_epoch_end', 'on_test_epoch_end'} & methods
    for path in ('VidOSC/train.py', 'VidOSC/train_changeit.py'):
        assert not (ROOT/path).exists()


def test_no_standalone_local_benchmark_entrypoints():
    for path in ('pipeline/metrics.py', 'VidOSC/data_scripts/evaluator.py', 'VidOSC/eval_changeit_all.py',
                 'experiments/attrs/evaluate.py', 'experiments/statespace/score_textmatch.py'):
        assert not (ROOT/path).exists()


def load_function(path, name, namespace):
    tree = ast.parse((ROOT/path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_attribute_track_loader_does_not_filter_on_gt():
    method = SimpleNamespace(schema_texts=lambda schema: [], text_emb=lambda *a: {},
                             feats=lambda *a: np.eye(2), norm_attr=lambda a: a,
                             evidence=lambda *a: (np.eye(2), np.eye(2), np.ones(2)))
    namespace = {'M': method, 'ds': Source(), 'recipe': lambda _: 'recipe', 'HERE': 'unused'}
    load = load_function('experiments/attrs/predict.py', 'load_tracks', namespace)
    schema = {'recipe': {'egg': {'visible': 'egg', 'attributes': [
        {'name': 'form', 'values': ['raw', 'cooked'], 'script': ['raw', 'cooked']}]}}}
    tracks = load(schema, 'encoder')
    assert len(tracks) == 1 and tracks[0]['V'] == ['raw', 'cooked']
    assert 'gt' not in tracks[0]


def test_slider_evidence_keeps_tracks_without_gt_state_values():
    source = Source()
    source.videos = ['S1_recipe_clip']
    sl = SimpleNamespace(ds=source, texts=lambda *a: {}, feats=lambda *a: np.eye(2),
                         ASSIGN={'egg': ['done']}, SL={'done': (['raw'], ['cooked'], ['heating'], True)},
                         mean_t=lambda embeddings, labels, obj: np.array([1., 0.]))
    load = load_function('experiments/sliders/evidence.py', 'load',
                         {'sl': sl, 'EXTRA': ['visible', 'held'], 'RTEXT': [], 'RA': {}, 'np': np})
    tracks, _ = load()
    assert len(tracks) == 1 and tracks[0]['T'] == 2
    assert 'y' not in tracks[0]
