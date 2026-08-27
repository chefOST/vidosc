"""MOSCATO (CMU-MMAC) evaluation harness for the VidOSC + Claude pipeline.

The MOSCATO-specific data handling, vocabulary mapping and evaluation logic in
this package is ported from the TubeletGraph x MOSCATO experiment in the
chefOST-ml repository, branch ``feat/tubeletgraph-moscato-cmu-s12``
(``scripts/inspect_video.py``, ``scripts/prepare_frames.py``,
``scripts/map_event_states.py``, ``scripts/evaluate.py``,
``scripts/log_wandb.py``).

Behaviour is preserved deliberately: the point of this package is that
TubeletGraph and VidOSC+Claude are scored by the *same* evaluation protocol, so
the numbers are comparable. Each module names its upstream source at the top.

Nothing in this package modifies the rest of VIDOSC-pipeline; the VidOSC and
Claude stages are imported from ``run_inference.py``, ``extract_features.py``
and ``describe_states.py`` unchanged.
"""

__all__ = [
    "config",
    "evaluate",
    "frame_mapping",
    "inspect_video",
    "log_wandb",
    "map_states",
    "prepare_clip",
    "segments_to_events",
    "visualize",
]
