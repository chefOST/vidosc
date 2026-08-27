"""Config loading for the MOSCATO experiment, with a ground-truth firewall.

The experiment is split into stages. Only the *evaluation* stage is allowed to
see ``gt_annotations_cmu.json``. Every other stage receives a config produced by
:func:`inference_config`, from which the annotation path has been removed, so a
leak would have to be written deliberately rather than happening by accident.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keys that point at MOSCATO ground truth. Never present in inference configs.
GROUND_TRUTH_KEYS = ("annotations_path",)

REQUIRED_KEYS = (
    "video_id",
    "video_path",
    "annotations_path",
    "files_dict_path",
    "state_dict_path",
    "target_object",
    "clip_start_global",
    "clip_end_global",
    "expected_raw_frames",
    "fps",
    "width",
    "height",
    "work_dir",
    "runs_dir",
)


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(config: dict[str, Any], key: str) -> Path:
    """Resolve a config path relative to the repository root when relative."""
    value = Path(config[key])
    return value if value.is_absolute() else REPO_ROOT / value


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    clip_start = int(config["clip_start_global"])
    clip_end = int(config["clip_end_global"])
    raw_frames = int(config["expected_raw_frames"])
    if not 0 <= clip_start < clip_end <= raw_frames:
        raise ValueError(
            f"Invalid clip [{clip_start}, {clip_end}) for {raw_frames} raw frames"
        )
    if int(config["fps"]) <= 0:
        raise ValueError("fps must be positive")

    config["config_path"] = str(path)
    return config


def inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the config with every ground-truth pointer stripped.

    Prepare / inference / description / vocabulary-mapping stages must be given
    only this. Passing the full config to those stages is the single most likely
    way to leak ground truth into a prediction, so it is made structurally hard.
    """
    stripped = {
        key: value for key, value in config.items() if key not in GROUND_TRUTH_KEYS
    }
    stripped["ground_truth_available_to_stage"] = False
    return stripped


def assert_no_ground_truth(config: dict[str, Any], stage: str) -> None:
    """Fail loudly if a prediction-side stage was handed a ground-truth path."""
    leaked = [key for key in GROUND_TRUTH_KEYS if key in config]
    if leaked:
        raise RuntimeError(
            f"Ground-truth leakage: stage {stage!r} was given {leaked}. "
            "Prediction stages must receive moscato.config.inference_config(cfg)."
        )
    if config.get("ground_truth_available_to_stage") is not False:
        raise RuntimeError(
            f"Stage {stage!r} did not receive an inference_config()-derived config."
        )


def canonical_states(state_dict: dict[str, Any]) -> list[str]:
    """Ported verbatim from chefOST-ml scripts/evaluate.py::canonical_states."""
    states = sorted(state.strip().lower() for state in state_dict["s2i"])
    if not states:
        raise ValueError("MOSCATO state_dict.s2i is empty")
    return states


def vocabulary_digest(allowed: list[str]) -> str:
    """Ported from chefOST-ml scripts/map_event_states.py provenance block."""
    return hashlib.sha256(
        json.dumps(allowed, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
