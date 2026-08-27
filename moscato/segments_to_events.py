"""Express VidOSC segment predictions in TubeletGraph's MOSCATO event schema.

This is the whole trick that keeps the two experiments comparable. TubeletGraph
emitted::

    {"initial_state": ..., "transitions": [{"local_frame": n, "state": ...}]}

(see chefOST-ml ``scripts/build_event_draft.py``) and ``moscato/evaluate.py``
densifies that onto MOSCATO's raw-frame timeline. Rather than writing a second
evaluator for VidOSC's per-second segments, we convert the segments into exactly
that schema and hand it to the same unmodified code.

Conversion rule, deterministic and lossless:

    segment 0                -> initial_state
    every later segment      -> a transition at local frame ``start_second * fps``

Segment boundaries land on whole seconds, so each transition frame is an exact
multiple of ``fps``. The evaluator's forward fill then reproduces, frame by
frame, the same assignment ``frame_mapping.timestep_for_local_frame`` gives --
which :func:`build_events` asserts before returning.
"""

from __future__ import annotations

from typing import Any

from . import frame_mapping


def build_events(
    *,
    video_id: str,
    target_object: str,
    clip_start_global: int,
    num_local_frames: int,
    fps: int,
    num_timesteps: int,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the MOSCATO event JSON from mapped VidOSC segments.

    Each segment dict must carry ``start`` / ``end`` (inclusive second indices),
    ``vidosc_state`` and ``mapped_moscato_state``.
    """
    if not segments:
        raise ValueError("No VidOSC segments to convert")

    ordered = sorted(segments, key=lambda item: int(item["start"]))
    if int(ordered[0]["start"]) != 0:
        raise ValueError(
            f"The first segment starts at second {ordered[0]['start']}, not 0. "
            "Every local frame must be covered, so no segment may be skipped "
            "(run the description stage with skip_states empty)."
        )

    expected_start = 0
    for segment in ordered:
        if int(segment["start"]) != expected_start:
            raise ValueError(
                f"Segment gap: expected a segment starting at second "
                f"{expected_start}, found {segment['start']}"
            )
        if int(segment["end"]) < int(segment["start"]):
            raise ValueError(f"Segment {segment} ends before it starts")
        if not segment.get("mapped_moscato_state"):
            raise ValueError(
                f"Segment starting at second {segment['start']} has no mapped "
                "MOSCATO state; every segment must be mapped before evaluation."
            )
        expected_start = int(segment["end"]) + 1
    if expected_start != num_timesteps:
        raise ValueError(
            f"Segments cover {expected_start} timesteps but VidOSC produced "
            f"{num_timesteps}"
        )

    transitions = []
    for segment in ordered[1:]:
        local_frame = int(segment["start"]) * fps
        if not 0 <= local_frame < num_local_frames:
            raise ValueError(
                f"Transition at local frame {local_frame} is outside "
                f"[0, {num_local_frames})"
            )
        transitions.append(
            {
                "local_frame": local_frame,
                "global_frame": clip_start_global + local_frame,
                "timestamp_s": round(local_frame / fps, 6),
                "vidosc_timestep": int(segment["start"]),
                "state": segment["mapped_moscato_state"],
                "raw_vidosc_state": segment["vidosc_state"],
                "raw_claude_description": segment.get("claude_description"),
            }
        )

    events = {
        "schema_version": 1,
        "video_id": video_id,
        "object": target_object,
        "clip_start_global": clip_start_global,
        "num_local_frames": num_local_frames,
        "initial_state": ordered[0]["mapped_moscato_state"],
        "initial_raw": {
            "vidosc_timestep": 0,
            "raw_vidosc_state": ordered[0]["vidosc_state"],
            "raw_claude_description": ordered[0].get("claude_description"),
        },
        "transitions": transitions,
        "provenance": {
            "source": "VidOSC segment predictions + Claude descriptions",
            "schema_source": (
                "chefOST-ml scripts/build_event_draft.py "
                "(feat/tubeletgraph-moscato-cmu-s12)"
            ),
            "state_mapping_status": "mapped",
            "ground_truth_used_for_inference": False,
            "ground_truth_used_for_mapping": False,
            "temporal_rule": frame_mapping.TIMESTEP_RULE,
            "densification": "forward fill from initial_state through transitions",
        },
    }

    _assert_matches_frame_mapping(
        events=events,
        segments=ordered,
        fps=fps,
        num_local_frames=num_local_frames,
        num_timesteps=num_timesteps,
    )
    return events


def dense_states(events: dict[str, Any]) -> list[str]:
    """Forward-fill the events to one state per local frame (no evaluator, no GT)."""
    num_local_frames = int(events["num_local_frames"])
    dense = [events["initial_state"]] * num_local_frames
    for transition in sorted(
        events["transitions"], key=lambda item: int(item["local_frame"])
    ):
        for local in range(int(transition["local_frame"]), num_local_frames):
            dense[local] = transition["state"]
    return dense


def _assert_matches_frame_mapping(
    *,
    events: dict[str, Any],
    segments: list[dict[str, Any]],
    fps: int,
    num_local_frames: int,
    num_timesteps: int,
) -> None:
    """Densified events must agree with the frame mapping on every frame."""
    by_timestep = {}
    for segment in segments:
        for timestep in range(int(segment["start"]), int(segment["end"]) + 1):
            by_timestep[timestep] = segment["mapped_moscato_state"]

    dense = dense_states(events)
    for local in range(num_local_frames):
        timestep = frame_mapping.timestep_for_local_frame(local, fps, num_timesteps)
        expected = by_timestep[timestep]
        if dense[local] != expected:
            raise AssertionError(
                f"Local frame {local} (timestep {timestep}): event densification "
                f"gives {dense[local]!r} but the frame mapping gives {expected!r}"
            )
