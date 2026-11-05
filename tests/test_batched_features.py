"""Check batching with small real Torch tensors; no pretrained model or GPU needed."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from moscato import extract_internvideo as extraction


class TinyEncoder:
    def __init__(self, corrupt_batch=False):
        self.batch_sizes = []
        self.corrupt_batch = corrupt_batch

    def encode_video(self, clips):
        self.batch_sizes.append(len(clips))
        value = clips.reshape(len(clips), -1).mean(dim=1, keepdim=True)
        output = torch.arange(1, 769, dtype=torch.float32).unsqueeze(0) + value
        if self.corrupt_batch and len(clips) > 1:
            output[1, 0] += 1000
        return output


@pytest.fixture
def clips(monkeypatch):
    decoded = []

    def sample(video, windows):
        decoded.append(1)
        for window in windows:
            yield [np.full((2, 2, 3), window[0], dtype=np.float32)] * 8

    monkeypatch.setattr(extraction, 'sampled_clips', sample)
    return decoded


def run_features(count, batch_size, missing=(), global_completed=0, object_completed=0,
                 corrupt_batch=False, callback=None):
    windows = [[i] * 8 for i in range(count)]
    crops = [{'crop_xyxy_exclusive': None if i in missing else [0, 0, 1, 1]} for i in range(count)]
    expected = torch.nn.functional.normalize(torch.arange(1, 769).float().unsqueeze(0)
                                            + torch.arange(count).float().unsqueeze(1), dim=-1)
    global_features = torch.zeros((count, 768))
    object_features = torch.zeros((count, 768))
    global_features[:global_completed] = expected[:global_completed]
    object_features[:object_completed] = expected[:object_completed]
    for i in missing:
        object_features[i] = 0
    saved = []

    def checkpoint(name, features, completed):
        saved.append((name, completed, features.clone()))
        if callback:
            callback(name, features, completed)

    model = TinyEncoder(corrupt_batch)
    report, elapsed = extraction.extract_feature_batches(
        'fixture', windows, crops, model, lambda frames: torch.tensor([frames[0].mean()]),
        'cpu', batch_size, global_features, object_features, global_completed, object_completed, checkpoint)
    return global_features, object_features, expected, model, saved, report


@pytest.mark.parametrize('batch_size', [1, 4, 8, 16])
def test_tail_order_missing_rows_and_one_decode(clips, batch_size):
    global_features, object_features, expected, model, saved, report = run_features(
        19, batch_size, missing={1, 5, 11, 18})
    torch.testing.assert_close(global_features, expected)
    for index in range(19):
        torch.testing.assert_close(object_features[index], torch.zeros(768) if index in {1, 5, 11, 18} else expected[index])
    assert len(clips) == 1
    assert max(model.batch_sizes) <= batch_size
    assert {name: completed for name, completed, _ in saved} == {'global': 19, 'object': 19}
    assert all(check['batch_vs_singleton']['passed'] for check in report['checks'])
    assert report['checks'][0]['batch_size'] == batch_size
    assert len(report['checks'][0]['window_indices']) == batch_size


def test_unequal_resume_positions_preserve_prefixes(clips):
    global_features, object_features, expected, _, saved, _ = run_features(
        23, 8, missing={2, 12, 21}, global_completed=10, object_completed=5)
    torch.testing.assert_close(global_features, expected)
    for index in range(23):
        torch.testing.assert_close(object_features[index], torch.zeros(768) if index in {2, 12, 21} else expected[index])
    assert all(position > (10 if name == 'global' else 5) for name, position, _ in saved)
    assert len(clips) == 1


def test_all_missing_object_crops_finish_with_zero_rows(clips):
    _, objects, _, _, saved, report = run_features(23, 8, missing=set(range(23)))
    assert torch.count_nonzero(objects) == 0
    assert {name: completed for name, completed, _ in saved}['object'] == 23
    assert [check['stream'] for check in report['checks']] == ['global']


def test_first_batch_checks_interior_rows_before_checkpoint(clips):
    saved = []
    with pytest.raises(ValueError, match='diverged'):
        run_features(12, 8, corrupt_batch=True, callback=lambda *args: saved.append(args))
    assert saved == []


def test_checkpoints_never_claim_pending_unencoded_rows(clips):
    def inspect(name, features, completed):
        assert torch.all(torch.linalg.vector_norm(features[:completed], dim=1) > .99)
        assert torch.count_nonzero(features[completed:]) == 0

    run_features(25, 8, callback=inspect)


@pytest.mark.parametrize('accelerator,batch', [('T4', 8), ('L4', 0), ('L4', 17), ('cpu', True)])
def test_invalid_compute_settings_fail(accelerator, batch):
    with pytest.raises(ValueError):
        extraction.validate_compute(accelerator, batch)
