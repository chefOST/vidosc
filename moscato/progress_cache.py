"""Atomic, validated intermediate feature checkpoints for interrupted inference."""
from pathlib import Path
import torch


def save_features(path, features, completed):
    if not 0 <= completed <= len(features) or not torch.isfinite(features).all():
        raise ValueError('Invalid intermediate features')
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save({'features': features.cpu(), 'completed': completed}, temp)
    temp.replace(path)


def load_features(path, count):
    path = Path(path)
    if not path.exists():
        return torch.zeros((count, 768)), 0
    data = torch.load(path, map_location='cpu', weights_only=True)
    features, completed = data['features'], data['completed']
    if (features.shape != (count, 768) or not isinstance(completed, int)
            or not 0 <= completed <= count or not torch.isfinite(features).all()):
        raise ValueError('Invalid saved intermediate features')
    return features, completed
