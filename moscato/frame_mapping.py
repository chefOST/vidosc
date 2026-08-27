"""The one deterministic mapping between VidOSC time and MOSCATO frames.

Four coordinate systems are in play:

    vidosc_timestep t   integer second index into VidOSC's per-second output
    timestamp           seconds from the start of the *clip*
    local_frame         zero-based frame index into the clip
    global_frame        MOSCATO / raw-video frame index

and exactly two rules connect them:

    global_frame = clip_start_global + local_frame
    timestep(local_frame) = min(local_frame // fps, num_timesteps - 1)

The clamp in the second rule is the only concession to the fact that the clip
does not divide evenly into whole seconds: the final timestep absorbs the
leftover tail frames. This is a forward fill, not interpolation and not
smoothing -- nothing here invents a value between two predictions.

Frames are never re-timed anywhere else in this package. ``prepare_clip`` builds
the clip MP4 out of the extracted local JPEGs at the source FPS, so clip frame i
is local frame i by construction, and VidOSC's 1-fps decode of that MP4 samples
frame ``t * fps``.
"""

from __future__ import annotations

from typing import Any

LOCAL_TO_GLOBAL_RULE = "global_frame = clip_start_global + local_frame"
TIMESTEP_RULE = "vidosc_timestep = min(local_frame // fps, num_timesteps - 1)"
TAIL_POLICY = (
    "the final VidOSC timestep covers every remaining local frame, so the "
    "timestep spans partition [0, num_local_frames) exactly"
)


def global_frame(local_frame: int, clip_start_global: int) -> int:
    return clip_start_global + local_frame


def timestamp_seconds(local_frame: int, fps: int) -> float:
    return local_frame / fps


def timestep_for_local_frame(local_frame: int, fps: int, num_timesteps: int) -> int:
    if local_frame < 0:
        raise ValueError(f"local_frame must be non-negative; got {local_frame}")
    return min(local_frame // fps, num_timesteps - 1)


def local_frame_span(
    timestep: int, fps: int, num_local_frames: int, num_timesteps: int
) -> tuple[int, int]:
    """Return the half-open ``[start, end)`` local-frame span of one timestep."""
    if not 0 <= timestep < num_timesteps:
        raise ValueError(f"timestep {timestep} outside [0, {num_timesteps})")
    start = timestep * fps
    if timestep == num_timesteps - 1:
        end = num_local_frames
    else:
        end = min((timestep + 1) * fps, num_local_frames)
    if start >= end:
        raise ValueError(
            f"Timestep {timestep} has an empty frame span [{start}, {end}). "
            f"num_timesteps={num_timesteps} is too large for "
            f"{num_local_frames} frames at {fps} fps."
        )
    return start, end


def representative_local_frame(timestep: int, fps: int, num_local_frames: int) -> int:
    """The local frame VidOSC's 1-fps decode actually sampled for this timestep.

    ``extract_features.extract_frames_1fps`` uses ffmpeg's ``fps=1`` filter,
    which emits the frame at the start of each whole second.
    """
    return min(timestep * fps, num_local_frames - 1)


def build_mapping(
    *,
    num_local_frames: int,
    clip_start_global: int,
    clip_end_global: int,
    fps: int,
    num_timesteps: int,
) -> dict[str, Any]:
    """Build and *verify* the full mapping. Raises if it is not a partition."""
    if num_local_frames <= 0:
        raise ValueError("num_local_frames must be positive")
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if clip_end_global - clip_start_global != num_local_frames:
        raise ValueError(
            f"Clip [{clip_start_global}, {clip_end_global}) is "
            f"{clip_end_global - clip_start_global} frames but "
            f"{num_local_frames} local frames were prepared"
        )

    timesteps = []
    covered = 0
    for timestep in range(num_timesteps):
        start, end = local_frame_span(timestep, fps, num_local_frames, num_timesteps)
        if start != covered:
            raise AssertionError(
                f"Timestep {timestep} starts at local frame {start}, "
                f"leaving a gap after {covered}"
            )
        covered = end
        rep = representative_local_frame(timestep, fps, num_local_frames)
        timesteps.append(
            {
                "vidosc_timestep": timestep,
                "timestamp_start_s": round(start / fps, 6),
                "timestamp_end_s": round(end / fps, 6),
                "local_frame_start": start,
                "local_frame_end": end,
                "global_frame_start": global_frame(start, clip_start_global),
                "global_frame_end": global_frame(end, clip_start_global),
                "representative_local_frame": rep,
                "representative_global_frame": global_frame(rep, clip_start_global),
            }
        )

    if covered != num_local_frames:
        raise AssertionError(
            f"Timestep spans cover {covered} of {num_local_frames} local frames"
        )

    # Round-trip every frame through the forward rule as a belt-and-braces check.
    for local in range(num_local_frames):
        timestep = timestep_for_local_frame(local, fps, num_timesteps)
        span = timesteps[timestep]
        if not span["local_frame_start"] <= local < span["local_frame_end"]:
            raise AssertionError(
                f"local frame {local} maps to timestep {timestep} whose span is "
                f"[{span['local_frame_start']}, {span['local_frame_end']})"
            )

    return {
        "schema_version": 1,
        "fps": fps,
        "clip_start_global": clip_start_global,
        "clip_end_global": clip_end_global,
        "num_local_frames": num_local_frames,
        "num_vidosc_timesteps": num_timesteps,
        "rules": {
            "local_to_global": LOCAL_TO_GLOBAL_RULE,
            "timestep": TIMESTEP_RULE,
            "tail": TAIL_POLICY,
            "timestamp": "timestamp_s = local_frame / fps (clip-relative)",
            "interpolation": "none; segment predictions are forward-filled only",
        },
        "timesteps": timesteps,
    }


def frame_rows(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the mapping into one row per local frame, for the per-frame CSV."""
    fps = int(mapping["fps"])
    clip_start = int(mapping["clip_start_global"])
    num_local_frames = int(mapping["num_local_frames"])
    num_timesteps = int(mapping["num_vidosc_timesteps"])
    rows = []
    for local in range(num_local_frames):
        timestep = timestep_for_local_frame(local, fps, num_timesteps)
        rows.append(
            {
                "local_frame": local,
                "global_moscato_frame": global_frame(local, clip_start),
                "timestamp": round(timestamp_seconds(local, fps), 6),
                "vidosc_timestep": timestep,
            }
        )
    return rows
