#!/usr/bin/env python3
"""Validate the exact CMU video against its MOSCATO annotation timeline.

Ported from chefOST-ml (branch feat/tubeletgraph-moscato-cmu-s12),
``scripts/inspect_video.py``. Behaviour is preserved; the only changes are
repo-relative path resolution and richer reporting.

This is the Step-4 alignment gate. It asserts

    video raw frame count == MOSCATO object timeline length
                          == MOSCATO state timeline length

and refuses to continue otherwise, so a misaligned experiment fails here rather
than silently producing meaningless metrics later.

Ground truth *is* read here. That is deliberate and safe: this stage produces no
predictions, it only validates that the data can be compared at all.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import load_config, load_json, resolve, write_json

DEFAULT_CONFIG = Path("configs/moscato/s12_sandwich_7150991-2470.json")


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,"
        "width,height,duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required; install FFmpeg first.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip() or "ffprobe failed") from error

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}; found {len(streams)}")
    return streams[0]


def stream_frame_count(stream: dict[str, Any]) -> int:
    """Prefer the exact decoded count; AVI containers often omit ``nb_frames``."""
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "", "N/A"):
            return int(value)
    raise ValueError("ffprobe reported no usable frame count for this video")


def validate(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    video_path = resolve(config, "video_path")
    annotations_path = resolve(config, "annotations_path")
    files_dict_path = resolve(config, "files_dict_path")
    state_dict_path = resolve(config, "state_dict_path")
    video_id = config["video_id"]
    target_object = config["target_object"]

    for required in (video_path, annotations_path, files_dict_path, state_dict_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing. Run scripts/setup_moscato_data.sh first."
            )

    if video_path.stem != video_id:
        raise ValueError(
            f"AVI stem {video_path.stem!r} does not match video_id {video_id!r}"
        )

    annotations = load_json(annotations_path)
    files_dict = load_json(files_dict_path)
    state_dict = load_json(state_dict_path)
    if video_id not in annotations:
        raise KeyError(f"{video_id} is missing from {annotations_path}")
    if video_id not in files_dict:
        raise KeyError(f"{video_id} is missing from {files_dict_path}")

    video_gt = annotations[video_id]
    object_frames = len(video_gt["object"])
    state_lengths = {
        object_name: len(timeline) for object_name, timeline in video_gt["state"].items()
    }
    mismatched = {
        object_name: length
        for object_name, length in state_lengths.items()
        if length != object_frames
    }
    if mismatched:
        raise ValueError(
            f"State timelines do not match object timeline ({object_frames}): {mismatched}"
        )
    if target_object not in video_gt["state"]:
        raise KeyError(
            f"Target {target_object!r} is unavailable; "
            f"choose from {sorted(video_gt['state'])}"
        )

    stream = probe_video(video_path)
    raw_frames = stream_frame_count(stream)
    fps = Fraction(stream["avg_frame_rate"])
    checks = {
        "raw_frames_equal_gt_object_timeline": raw_frames == object_frames,
        "raw_frames_equal_gt_state_timelines": all(
            length == raw_frames for length in state_lengths.values()
        ),
        "raw_frames_equal_manifest": raw_frames == int(config["expected_raw_frames"]),
        "fps_equal_manifest": fps == Fraction(int(config["fps"]), 1),
        "dimensions_equal_manifest": (
            int(stream["width"]) == int(config["width"])
            and int(stream["height"]) == int(config["height"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Alignment validation failed: "
            + ", ".join(failed)
            + f" (video={raw_frames} frames @ {float(fps)} fps "
            f"{stream['width']}x{stream['height']}; "
            f"gt_object={object_frames}; gt_states={state_lengths})"
        )

    clip_start = int(config["clip_start_global"])
    clip_end = int(config["clip_end_global"])
    if not 0 <= clip_start < clip_end <= raw_frames:
        raise ValueError(
            f"Invalid clip [{clip_start}, {clip_end}) for {raw_frames} raw frames"
        )

    entry = files_dict[video_id]
    original_path = entry.get("video_path", "")
    if Path(original_path).stem != video_id:
        raise ValueError(
            f"MOSCATO files_dict points at an unexpected video: {original_path!r}"
        )

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "moscato_original_video_path": original_path,
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps),
        "raw_frames": raw_frames,
        "moscato_object_timeline_frames": object_frames,
        "moscato_state_timeline_frames": state_lengths,
        # MOSCATO's own feature stride. NOT the raw frame count -- recorded here
        # purely so the distinction stays visible.
        "moscato_feature_len": entry.get("feature_len"),
        "target_object": target_object,
        "available_objects": sorted(video_gt["state"]),
        "num_canonical_states": len(state_dict["s2i"]),
        "clip_start_global": clip_start,
        "clip_end_global": clip_end,
        "num_local_frames": clip_end - clip_start,
        "local_to_global_rule": "global_frame = clip_start_global + local_frame",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, help="Optional metadata JSON output")
    args = parser.parse_args()

    report = validate(args.config)
    print(json.dumps(report, indent=2))
    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
