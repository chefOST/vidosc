import json
import torch
import pytest
from moscato.baseline import digest
from moscato.extract_internvideo import window_indices
from moscato.reuse_global import validate_global


def test_reuse_requires_matching_video_and_features(tmp_path):
    video, checkpoint = tmp_path / 'video', tmp_path / 'checkpoint'
    video.write_bytes(b'video'); checkpoint.write_bytes(b'checkpoint')
    features = torch.zeros((1, 768)); features[0, 0] = 1
    path = tmp_path / 'global_features.pth.tar'; torch.save(features, path)
    manifest = dict(video_sha256=digest(video), checkpoint_sha256=digest(checkpoint),
        feature_sha256=digest(path), sampled_global_frame_indices=window_indices(0, 30, 30),
        feature_layout='global_only_768', clip_start_seconds=0, clip_end_seconds=1)
    (tmp_path / 'feature_manifest.json').write_text(json.dumps(manifest))
    assert validate_global(tmp_path, video, checkpoint, 30, 30) == manifest
    video.write_bytes(b'different video')
    with pytest.raises(ValueError, match='provenance'):
        validate_global(tmp_path, video, checkpoint, 30, 30)
