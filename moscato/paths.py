"""Shared local assets, independent of any experiment checkout."""
import os
from pathlib import Path


def checkpoint_path(name, root):
    directory = os.environ.get("VIDOSC_CHECKPOINT_DIR")
    directory = Path(directory).expanduser() if directory else Path(root).parent / "data/checkpoints/vidosc"
    return directory.resolve() / name


def state_vocabulary_path(root):
    configured = os.environ.get("VIDOSC_STATE_VOCABULARY")
    path = Path(configured).expanduser() if configured else Path(root).parent / "chefOST-data/MOSCATO/vocabulary/CMU/state_dict.json"
    return path.resolve()
