import torch
import pytest
from moscato.progress_cache import save_features, load_features


def test_resume_distinguishes_missing_crop_zero_from_unprocessed(tmp_path):
    path = tmp_path / 'features.pt'
    features, done = load_features(path, 5)
    assert done == 0
    features[0, 0] = 1
    save_features(path, features, 3)
    restored, done = load_features(path, 5)
    assert done == 3  # processed zero rows are still complete
    assert torch.equal(restored, features)
    with pytest.raises(ValueError):
        load_features(path, 6)


def test_reject_invalid_progress(tmp_path):
    with pytest.raises(ValueError):
        save_features(tmp_path / 'bad.pt', torch.zeros((2, 768)), 3)
