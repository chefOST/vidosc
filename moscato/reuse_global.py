"""Validate completed global features before reusing them after an interruption."""
import json
from pathlib import Path
import torch
from .baseline import digest
from .extract_internvideo import window_indices


def validate_global(directory, video, checkpoint, frames, fps):
    directory = Path(directory)
    manifest = json.loads((directory / 'feature_manifest.json').read_text())
    features_path = directory / 'global_features.pth.tar'
    expected = window_indices(0, frames, fps)
    if (manifest['video_sha256'] != digest(video)
            or manifest['checkpoint_sha256'] != digest(checkpoint)
            or manifest['feature_sha256'] != digest(features_path)
            or manifest['sampled_global_frame_indices'] != expected
            or manifest['feature_layout'] != 'global_only_768'
            or manifest['clip_start_seconds'] != 0
            or abs(manifest['clip_end_seconds'] - frames / fps) > 1e-6):
        raise ValueError('Cached global feature provenance differs')
    features = torch.load(features_path, weights_only=True, map_location='cpu')
    if (features.shape != (len(expected), 768) or not torch.isfinite(features).all()
            or not torch.allclose(features.norm(dim=1), torch.ones(len(expected)), atol=1e-4)):
        raise ValueError('Invalid cached global features')
    return manifest
