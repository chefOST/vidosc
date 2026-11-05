"""Exercise the cloud workflow offline with tiny deterministic encoder doubles."""
import ast
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from moscato.checkpoint_adapter import inference_timesteps

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('count,expected', [(1, 900), (270, 900), (900, 900),
                                            (901, 901), (2516, 2516), (5000, 5000)])
def test_padding_preserves_short_runs_and_accepts_long_videos(count, expected):
    assert inference_timesteps(count, 5000) == expected


@pytest.mark.parametrize('count', [0, -1, 5001])
def test_checkpoint_capacity_is_not_silently_truncated(count):
    with pytest.raises(ValueError):
        inference_timesteps(count, 5000)


class Tensor(np.ndarray):
    """Only the tensor operations the orchestration needs; no model simulation."""
    def cpu(self):
        return self

    def float(self):
        return self.astype(np.float32)

    def to(self, device):
        assert device == 'cpu'
        return self

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)


def tensor(value):
    return np.asarray(value, dtype=np.float32).view(Tensor)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    from moscato import baseline, extract_internvideo, checkpoint_adapter, object_crops

    def local_path(path):
        path = Path(path)
        if path.is_relative_to(tmp_path):
            return path
        if str(path).startswith(('/tmp/', '/progress', '/root/', '/inputs/')):
            path = tmp_path / str(path).lstrip('/')
        return path

    # Serialize only test-generated arrays, never external checkpoint data.
    def save(value, path):
        path = local_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            pickle.dump(value, stream)

    def load(path, **kwargs):
        with local_path(path).open('rb') as stream:
            return pickle.load(stream)

    fake_torch = SimpleNamespace(
        set_num_threads=lambda n: None, inference_mode=nullcontext,
        isfinite=np.isfinite, zeros=lambda shape: tensor(np.zeros(shape)),
        cat=lambda arrays, dim=0: tensor(np.concatenate(arrays, axis=dim)),
        save=save, load=load,
        nn=SimpleNamespace(functional=SimpleNamespace(
            normalize=lambda arr, dim: tensor(arr / np.linalg.norm(arr, axis=dim, keepdims=True)))))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    # Execute the real progress-cache functions using the test tensor API.
    cache_namespace = {'__name__': '_test_progress_cache'}
    exec(compile((ROOT / 'moscato/progress_cache.py').read_text(), 'progress_cache.py', 'exec'), cache_namespace)
    monkeypatch.setitem(sys.modules, 'moscato.progress_cache', SimpleNamespace(**cache_namespace))

    calls = {'detector_load': 0, 'encoder_load': 0, 'encoded': [], 'predictions': []}
    manifest = {
        'video_id': 'test-video', 'video_sha256': 'video', 'checkpoint_sha256': 'encoder',
        'upstream_commit': 'pinned-commit', 'sampled_global_frame_indices': [[i] * 8 for i in range(3)],
        'interval_starts_seconds': [0, 1, 2], 'feature_backbone': 'InternVideo-MM-L-14',
        'clip_start_seconds': 0, 'clip_end_seconds': 3,
    }

    def prepare(*args):
        assert args[1:3] == ('test-video', 'active-object')
        assert args[5:9] == (0, 3, 1.0, 'cpu')
        return copy.deepcopy(manifest)

    def load_detector(*args):
        calls['detector_load'] += 1
        return None, None

    monkeypatch.setitem(sys.modules, 'moscato.hand_object', SimpleNamespace(
        load_detector=load_detector, detect_frame=lambda model, cfg, frame, device: int(frame[0, 0, 0])))
    monkeypatch.setattr(object_crops, 'choose_crop', lambda detections, w, h: None if detections == 1 else [0, 0, 1, 1])
    monkeypatch.setattr(extract_internvideo, 'prepare_manifest', prepare)
    monkeypatch.setattr(extract_internvideo, 'sampled_clips',
                        lambda path, windows: ([np.full((2, 2, 3), i, dtype=np.uint8)] * 8 for i in range(len(windows))))

    def load_encoder(*args):
        calls['encoder_load'] += 1

        def encode(value):
            calls['encoded'].append(float(value[0, 0]))
            return tensor([np.arange(768) + 1 + value[0, 0]])

        return SimpleNamespace(encode_video=encode), lambda frames: tensor([frames[0].mean()])

    monkeypatch.setattr(extract_internvideo, 'load_encoder', load_encoder)
    monkeypatch.setattr(checkpoint_adapter, 'read_weights', lambda path: {})
    capacity = {'max_timesteps': 5000}
    monkeypatch.setattr(checkpoint_adapter, 'inspect_weights', lambda weights: capacity)

    def predict(checkpoint, features, upstream, pad_to, raw_output):
        combined = load(features)
        calls['predictions'].append((combined.copy(), pad_to))
        state, action = np.zeros((len(combined), 52)), np.zeros((len(combined), 28))
        np.savez_compressed(local_path(raw_output), state_logits=state, action_logits=action)
        return state, action, {'max_timesteps': capacity['max_timesteps'], 'transformer_timesteps': pad_to}

    monkeypatch.setattr(checkpoint_adapter, 'predict', predict)
    original_digest = baseline.digest
    monkeypatch.setattr(baseline, 'digest', lambda path: original_digest(local_path(path))
                        if local_path(path).is_file() else str(path))

    # Load only the function: importing the deployment module would contact Modal.
    tree = ast.parse((ROOT / 'scripts/modal_clip.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_clip')
    function.decorator_list = []
    namespace = {'Path': local_path, 'progress_volume': SimpleNamespace(commit=lambda: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'modal_clip.py', 'exec'), namespace)
    return namespace['run_clip'], calls, load, capacity


def test_cloud_extraction_outputs_and_resume(worker):
    run, calls, load, _ = worker
    outputs = run('test-video', 3, 1.0)
    assert set(outputs) == {'feature_manifest.json',
                            'object_features.pth.tar', 'combined_features.pth.tar',
                            'raw_logits.npz', 'raw_probabilities.npz'}
    global_features = load('/tmp/global_features.pth.tar')
    object_features = load('/tmp/object.pth.tar')
    combined, padding = calls['predictions'][0]
    assert padding == 900
    np.testing.assert_array_equal(combined[:, :768], global_features)
    np.testing.assert_array_equal(combined[:, 768:], object_features)
    np.testing.assert_allclose(np.linalg.norm(global_features, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(object_features[1], 0)
    assert calls['encoded'] == [0, 1, 2, 0, 2]  # global, then detected crops
    manifest = json.loads(outputs['feature_manifest.json'])
    assert manifest['feature_extraction_device'] == 'Modal CPU'
    assert manifest['global_intervals_reused'] == manifest['object_intervals_reused'] == 0
    assert set(manifest['stage_elapsed_seconds']) == {'setup', 'detection', 'encoder_load',
                                                     'global_features', 'object_features', 'inference'}
    resumed = json.loads(run('test-video', 3, 1.0)['feature_manifest.json'])
    assert resumed['global_intervals_reused'] == resumed['object_intervals_reused'] == 3
    assert calls['encoder_load'] == calls['detector_load'] == 1
    np.testing.assert_array_equal(calls['predictions'][1][0], combined)


def test_capacity_failure_precedes_any_extraction(worker):
    run, calls, _, capacity = worker
    capacity['max_timesteps'] = 2
    with pytest.raises(ValueError, match='checkpoint supports 2'):
        run('test-video', 3, 1.0)
    assert calls['encoder_load'] == calls['detector_load'] == 0
