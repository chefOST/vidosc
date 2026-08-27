#!/usr/bin/env python3
"""Decode the AVI, cut the configured clip, and re-encode it for VidOSC.

Ported from chefOST-ml (branch feat/tubeletgraph-moscato-cmu-s12),
``scripts/prepare_frames.py``. ``decode_all`` and ``materialize_clip`` keep the
original behaviour and the original contiguity assertions, so the frames handed
to VidOSC are byte-for-byte the frames TubeletGraph was given.

The one addition is ``encode_clip_video``: VIDOSC-pipeline's feature extractor
consumes a video file, not a frame directory. Building that video *from the
already-extracted local JPEGs* -- rather than seeking into the AVI -- is what
makes clip frame i identical to local frame i by construction, with no
container-level seeking error to reason about.

This stage never reads ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .config import (
    assert_no_ground_truth,
    inference_config,
    load_config,
    resolve,
    sha256,
    write_json,
)
from . import frame_mapping

DEFAULT_CONFIG = Path("configs/moscato/s12_sandwich_7150991-2470.json")


def numbered_frames(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg"))


def assert_numbering(frames: list[Path], expected_count: int) -> None:
    if len(frames) != expected_count:
        raise ValueError(f"Expected {expected_count} frames; found {len(frames)}")
    for index, frame in enumerate(frames):
        if frame.name != f"{index:07d}.jpg":
            raise ValueError(f"Non-contiguous frame sequence at {index}: found {frame.name}")


def decode_all(video_path: Path, raw_dir: Path, expected_count: int) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = numbered_frames(raw_dir)
    if existing:
        assert_numbering(existing, expected_count)
        print(f"Using {len(existing)} existing raw frames in {raw_dir}")
        return existing

    if not video_path.is_file():
        raise FileNotFoundError(
            f"{video_path} is missing. See scripts/setup_moscato_data.sh for how to "
            "obtain the CMU-MMAC AVI."
        )

    command = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vsync", "0",
        "-q:v", "2",
        "-start_number", "0",
        str(raw_dir / "%07d.jpg"),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required; install FFmpeg first.") from error

    frames = numbered_frames(raw_dir)
    assert_numbering(frames, expected_count)
    print(f"Decoded {len(frames)} exact raw frames into {raw_dir}")
    return frames


def materialize_clip(
    raw_frames: list[Path], clip_dir: Path, start: int, end: int
) -> list[Path]:
    expected_count = end - start
    clip_dir.mkdir(parents=True, exist_ok=True)
    existing = numbered_frames(clip_dir)
    if existing:
        assert_numbering(existing, expected_count)
        print(f"Using {len(existing)} existing clip frames in {clip_dir}")
        return existing

    for local_index, global_index in enumerate(range(start, end)):
        source = raw_frames[global_index]
        target = clip_dir / f"{local_index:07d}.jpg"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    frames = numbered_frames(clip_dir)
    assert_numbering(frames, expected_count)
    print(f"Prepared {len(frames)} local frames: local 0 maps to global {start}")
    return frames


def encode_clip_video(clip_dir: Path, clip_video: Path, fps: int, expected_count: int) -> Path:
    """Re-encode the local frames into an MP4 at the source frame rate.

    ``extract_features.py`` decodes whatever it is given at 1 fps, so the clip
    video must run at the true 30 fps for timestep t to land on local frame
    ``t * fps``. The frame count is verified after encoding.
    """
    if clip_video.is_file():
        print(f"Using existing clip video {clip_video}")
        return clip_video

    clip_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(clip_dir / "%07d.jpg"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        # Constant frame rate, so clip frame i stays local frame i. Do not add
        # -vsync/-fps_mode here: recent ffmpeg rejects it alongside -r.
        "-r", str(fps),
        str(clip_video),
    ]
    subprocess.run(command, check=True)

    encoded = probe_frame_count(clip_video)
    if encoded != expected_count:
        raise ValueError(
            f"Clip video has {encoded} frames but {expected_count} were expected; "
            "refusing to continue with a misaligned clip."
        )
    print(f"Encoded {encoded}-frame clip video at {fps} fps: {clip_video}")
    return clip_video


def probe_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])


def prepare(config_path: Path) -> dict[str, Any]:
    full_config = load_config(config_path)
    config = inference_config(full_config)
    assert_no_ground_truth(config, "prepare_clip")

    video_path = resolve(config, "video_path")
    work_dir = resolve(config, "work_dir")
    raw_dir = work_dir / "raw_frames"
    clip_dir = work_dir / "frames"
    clip_video = work_dir / f"{config['video_id']}_clip.mp4"

    expected_count = int(config["expected_raw_frames"])
    clip_start = int(config["clip_start_global"])
    clip_end = int(config["clip_end_global"])
    fps = int(config["fps"])

    raw_frames = decode_all(video_path, raw_dir, expected_count)
    clip_frames = materialize_clip(raw_frames, clip_dir, clip_start, clip_end)
    shutil.copy2(clip_frames[0], work_dir / "first_frame.jpg")
    encode_clip_video(clip_dir, clip_video, fps, len(clip_frames))

    metadata = {
        "schema_version": 1,
        "video_id": config["video_id"],
        "target_object": config["target_object"],
        "source_video": str(video_path),
        "source_video_sha256": sha256(video_path),
        "raw_frames_dir": str(raw_dir),
        "clip_frames_dir": str(clip_dir),
        "clip_video": str(clip_video),
        "clip_start_global": clip_start,
        "clip_end_global": clip_end,
        "num_local_frames": len(clip_frames),
        "fps": fps,
        "clip_duration_s": round(len(clip_frames) / fps, 6),
        "local_to_global_rule": frame_mapping.LOCAL_TO_GLOBAL_RULE,
        "ground_truth_used": False,
    }
    write_json(work_dir / "metadata.json", metadata)
    print(f"Metadata: {work_dir / 'metadata.json'}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()
