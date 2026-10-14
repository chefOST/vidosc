#!/usr/bin/env python3
"""Run S12_Sandwich_7150991-2470 through VidOSC + Claude and score it on MOSCATO.

Stages, each independently runnable:

    inspect    ffprobe vs. MOSCATO timeline alignment gate (reads GT; no predictions)
    prepare    decode the AVI, cut [clip_start, clip_end), build the clip MP4
    predict    features -> VidOSC -> segments -> Claude -> MOSCATO vocabulary
    evaluate   score the saved predictions with the ported TubeletGraph evaluator
    all        inspect + prepare + predict + evaluate

The split is what enforces experimental integrity: `predict` never receives the
annotation path (see moscato/config.inference_config), writes every artifact to
disk, and only then does `evaluate` open the ground truth.

Usage:
    python -m moscato.run_experiment inspect
    python -m moscato.run_experiment prepare
    python -m moscato.run_experiment predict --ckpt ../VidOSC/checkpoints/chopping.ckpt \
        --run-name baseline
    python -m moscato.run_experiment evaluate --run-name baseline
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moscato import decode as moscato_decode
from moscato import frame_mapping, prepare_clip, segments_to_events, visualize
from moscato.config import (
    assert_no_ground_truth,
    inference_config,
    load_config,
    load_json,
    resolve,
    sha256,
    write_json,
)
from moscato.evaluate import score_run
from moscato.inspect_video import validate as validate_alignment
from moscato.log_wandb import WANDB_ENTITY, WANDB_PROJECT, WANDB_PROJECTS
from moscato.map_states import DEFAULT_MAPPER_MODEL, ClaudeMapper, map_segments

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs/moscato/s12_sandwich_7150991-2470.json"

PREDICTION_CSV_COLUMNS = [
    "local_frame",
    "global_moscato_frame",
    "timestamp",
    "vidosc_timestep",
    "vidosc_state",
    "claude_description",
    "mapped_moscato_state",
]


def _run_name(value: str) -> str:
    if value:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError(f"Unsafe run name: {value!r}")
        return value
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(
    config_path: Path,
    *,
    ckpt: Path,
    run_name: str,
    obj_feat: Path | None = None,
    vlm_model: str | None = None,
    mapper_model: str | None = None,
    sample_rate: str | None = None,
    skip_states: str | None = None,
    prompt_mode: str | None = None,
    decode_mode: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Feature extraction -> VidOSC -> segments -> Claude -> MOSCATO vocabulary."""
    import numpy as np

    from describe_states import group_segments

    from moscato import describe as moscato_describe
    from extract_features import extract_clip_features
    from run_inference import STATE_NAMES, run_inference as vidosc_run_inference

    full_config = load_config(config_path)
    config = inference_config(full_config)
    assert_no_ground_truth(config, "predict")

    video_id = config["video_id"]
    target_object = config["target_object"]
    fps = int(config["fps"])
    clip_start = int(config["clip_start_global"])
    clip_end = int(config["clip_end_global"])
    work_dir = resolve(config, "work_dir")
    frames_dir = work_dir / "frames"
    clip_video = work_dir / f"{video_id}_clip.mp4"
    metadata_path = work_dir / "metadata.json"

    for required in (metadata_path, clip_video):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing; run the prepare stage first."
            )
    prepare_metadata = load_json(metadata_path)
    num_local_frames = int(prepare_metadata["num_local_frames"])
    if num_local_frames != clip_end - clip_start:
        raise ValueError(
            f"Prepared clip has {num_local_frames} frames but the config asks for "
            f"{clip_end - clip_start}"
        )

    resolved_run = _run_name(run_name)
    run_dir = resolve(config, "runs_dir") / resolved_run
    if run_dir.exists() and not force:
        raise FileExistsError(
            f"{run_dir} already exists. Pick another --run-name or pass --force."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()

    # -- 1. features ------------------------------------------------------
    ckpt = Path(ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    feat_path = work_dir / f"{video_id}_clip.pth.tar"
    if feat_path.is_file():
        print(f"Using existing features {feat_path}")
    else:
        extract_clip_features(str(clip_video), str(feat_path))

    # -- 2. VidOSC --------------------------------------------------------
    argmax_predicted, probs = vidosc_run_inference(
        str(feat_path), str(ckpt), str(obj_feat) if obj_feat else None, use_wandb=False
    )
    num_timesteps = int(len(argmax_predicted))

    # Decode BEFORE segment grouping: everything downstream -- segments, the
    # frames Claude sees, the MOSCATO events -- consumes `predicted`, so the
    # decode mode has to be applied here and nowhere else.
    effective_decode_mode = decode_mode or config.get("decode", "argmax")
    predicted, decode_info = moscato_decode.decode(effective_decode_mode, probs)
    print(
        f"Decode mode: {effective_decode_mode} "
        f"({decode_info['num_segments']} segments over {num_timesteps} timesteps)"
    )
    if effective_decode_mode == "argmax" and not np.array_equal(predicted, argmax_predicted):
        raise AssertionError(
            "argmax decoding diverged from run_inference's own argmax; the "
            "baseline path must stay byte-identical to the pre-existing behaviour."
        )

    import torch

    raw_feat = torch.load(feat_path, map_location="cpu")
    feature_dim = int(raw_feat.shape[-1])
    ckpt_state = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_raw = ckpt_state.get("state_dict", ckpt_state)
    ckpt_input_dim = int(ckpt_raw["model.proj.weight"].shape[1])
    obj_feat_padded = obj_feat is None and feature_dim == 768 and ckpt_input_dim == 1536

    # -- 3. frame mapping -------------------------------------------------
    mapping = frame_mapping.build_mapping(
        num_local_frames=num_local_frames,
        clip_start_global=clip_start,
        clip_end_global=clip_end,
        fps=fps,
        num_timesteps=num_timesteps,
    )
    write_json(run_dir / "frame_mapping.json", mapping)

    write_json(
        run_dir / "vidosc_predictions.json",
        {
            "schema_version": 1,
            "video_id": video_id,
            "checkpoint": str(ckpt),
            "num_vidosc_timesteps": num_timesteps,
            "decode": decode_info,
            "state_names": {str(k): v for k, v in STATE_NAMES.items()},
            "predictions": [
                {
                    "vidosc_timestep": int(timestep),
                    "state_index": int(state_index),
                    "state": STATE_NAMES[int(state_index)],
                    "argmax_state": STATE_NAMES[int(argmax_predicted[timestep])],
                    "confidence": float(probs[timestep, state_index]),
                    "probabilities": {
                        STATE_NAMES[index]: float(probs[timestep, index])
                        for index in STATE_NAMES
                    },
                }
                for timestep, state_index in enumerate(predicted)
            ],
        },
    )

    # -- 4. segments ------------------------------------------------------
    raw_segments = group_segments(np.asarray(predicted))
    skip = {
        item.strip()
        for item in (skip_states if skip_states is not None else config.get("skip_states", "")).split(",")
        if item.strip()
    }
    if skip:
        raise ValueError(
            "MOSCATO evaluation requires a mapped state for every frame, so no "
            f"segment may be skipped. Got skip_states={sorted(skip)}."
        )

    effective_sample_rate = sample_rate or config.get("sample_rate", "per_segment")
    if effective_sample_rate == "per_second":
        raw_segments = [(t, t, state) for start, end, state in raw_segments for t in range(start, end + 1)]
    elif effective_sample_rate != "per_segment":
        raise ValueError(f"Unknown sample_rate {effective_sample_rate!r}")

    segments: list[dict[str, Any]] = []
    for start, end, state_index in raw_segments:
        rep_local = frame_mapping.representative_local_frame(
            (start + end) // 2, fps, num_local_frames
        )
        segments.append(
            {
                "start": int(start),
                "end": int(end),
                "vidosc_state_index": int(state_index),
                "vidosc_state": STATE_NAMES[int(state_index)],
                "representative_vidosc_timestep": (start + end) // 2,
                "representative_local_frame": rep_local,
                "representative_global_frame": clip_start + rep_local,
                "representative_timestamp_s": round(rep_local / fps, 6),
                "local_frame_start": frame_mapping.local_frame_span(
                    int(start), fps, num_local_frames, num_timesteps
                )[0],
                "local_frame_end": frame_mapping.local_frame_span(
                    int(end), fps, num_local_frames, num_timesteps
                )[1],
            }
        )
    write_json(
        run_dir / "vidosc_segments.json",
        {
            "schema_version": 1,
            "video_id": video_id,
            "sample_rate": effective_sample_rate,
            "num_segments": len(segments),
            "segments": segments,
        },
    )
    print(f"\n{len(segments)} VidOSC segment(s) at sample_rate={effective_sample_rate}")

    # -- 5. Claude descriptions ------------------------------------------
    # Read the full-resolution local frame straight from the prepared frames
    # directory -- the SAME file the vocabulary mapper reads.
    #
    # This deliberately does NOT use extract_frames_1fps: that helper scales to
    # 224x224 for CLIP's input, which for a 1024x768 source is both a heavy
    # downscale and an aspect-ratio squash. Captioning those thumbnails produced
    # descriptions like "red cabbage" and "liquified in a bottle" for a carrot
    # and cucumber salad, while the mapper -- reading full-resolution frames --
    # saw something entirely different. Both stages now see identical pixels.
    from PIL import Image

    def representative_frame(local: int) -> "np.ndarray":
        path = frames_dir / f"{local:07d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; re-run the prepare stage."
            )
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    UNINFORMATIVE = ("not visible", "not clearly visible", "cannot see", "too blurry")

    def is_uninformative(description: str) -> bool:
        head = description.strip().lower()[:60]
        return any(marker in head for marker in UNINFORMATIVE)

    def candidate_local_frames(segment: dict[str, Any]) -> list[int]:
        """Midpoint first, then 25% and 75% within the segment.

        A single representative frame is fragile in egocentric video: the
        midpoint of a segment can land on a motion-blurred frame where the
        object genuinely is not visible. That gets *worse* with ordered
        decoding, because 3 long segments mean 3 samples for the whole clip
        while argmax's 12 short ones average the risk away. Fall back to other
        frames in the same segment rather than forcing a guess on a blur.
        """
        start, end = int(segment["start"]), int(segment["end"])
        offsets = [0.5, 0.25, 0.75] if end > start else [0.5]
        seen, frames = set(), []
        for offset in offsets:
            timestep = start + int(round((end - start) * offset))
            local = frame_mapping.representative_local_frame(
                timestep, fps, num_local_frames
            )
            if local not in seen:
                seen.add(local)
                frames.append(local)
        return frames

    probe = representative_frame(int(segments[0]["representative_local_frame"]))
    description_frame_size = {"height": int(probe.shape[0]), "width": int(probe.shape[1])}
    print(
        f"Description frames: {description_frame_size['width']}x"
        f"{description_frame_size['height']} (full resolution, as the mapper sees)"
    )

    vlm_model_name = vlm_model or config.get("vlm_model", "claude-haiku-4-5-20251001")
    effective_prompt_mode = prompt_mode or config.get("prompt_mode", "neutral")
    if effective_prompt_mode not in moscato_describe.PROMPT_MODES:
        raise ValueError(
            f"prompt_mode must be one of {moscato_describe.PROMPT_MODES}; "
            f"got {effective_prompt_mode!r}"
        )
    backend = (
        None if dry_run
        else moscato_describe.build_backend(effective_prompt_mode, vlm_model_name)
    )
    print(f"Description prompt mode: {effective_prompt_mode}")
    claude_raw: list[dict[str, Any]] = []
    for segment in segments:
        timestep = int(segment["representative_vidosc_timestep"])
        attempts: list[dict[str, Any]] = []
        if dry_run:
            local = int(segment["representative_local_frame"])
            description = "(dry run - no VLM call)"
        else:
            for local in candidate_local_frames(segment):
                description = backend.describe(
                    representative_frame(local), target_object, segment["vidosc_state"]
                )
                attempts.append({"local_frame": local, "description": description})
                if not is_uninformative(description):
                    break
            else:
                print(
                    f"  (every sampled frame in t={segment['start']}-{segment['end']}s "
                    f"was uninformative; keeping the last)"
                )
        # The frame that produced the kept description, which may not be the midpoint.
        segment["representative_local_frame"] = int(local)
        segment["representative_global_frame"] = clip_start + int(local)
        segment["representative_timestamp_s"] = round(local / fps, 6)
        segment["description_frame_attempts"] = attempts
        segment["claude_description"] = description
        claude_raw.append(
            {
                "segment_start_second": segment["start"],
                "segment_end_second": segment["end"],
                "vidosc_state": segment["vidosc_state"],
                "representative_vidosc_timestep": timestep,
                "representative_local_frame": segment["representative_local_frame"],
                "representative_global_frame": segment["representative_global_frame"],
                "prompt_mode": effective_prompt_mode,
                "prompt": moscato_describe.prompt_for(
                    effective_prompt_mode, target_object, segment["vidosc_state"]
                ),
                "raw_claude_description": description,
            }
        )
        label = f"t={segment['start']}s" if segment["start"] == segment["end"] else f"t={segment['start']}-{segment['end']}s"
        print(f"{label:<14} [{segment['vidosc_state']:<14}] -> \"{description}\"")

    write_json(
        run_dir / "claude_raw.json",
        {
            "schema_version": 1,
            "video_id": video_id,
            "target_object": target_object,
            "vlm_backend": "claude",
            "vlm_model": vlm_model_name,
            "prompt_mode": effective_prompt_mode,
            "vidosc_state_shown_to_vlm": effective_prompt_mode == "stated_state",
            "ground_truth_shown_to_vlm": False,
            "responses": claude_raw,
        },
    )

    # -- 6. closed-vocabulary mapping -------------------------------------
    state_dict = load_json(resolve(config, "state_dict_path"))
    if dry_run:
        from moscato.config import canonical_states, vocabulary_digest

        allowed = canonical_states(state_dict)
        for segment in segments:
            segment["mapped_moscato_state"] = allowed[0]
            segment["mapping"] = {"model": "(dry run)", "raw_responses": [], "mapped_state": allowed[0]}
        mapping_provenance = {
            "state_mapping_status": "dry-run placeholder",
            "state_mapping_model": None,
            "ground_truth_used_for_mapping": False,
            "allowed_states_sha256": vocabulary_digest(allowed),
            "num_allowed_states": len(allowed),
        }
    else:
        mapping_provenance = map_segments(
            config=config,
            state_dict=state_dict,
            segments=segments,
            frame_jpeg_for=lambda local: (frames_dir / f"{local:07d}.jpg").read_bytes(),
            mapper=ClaudeMapper(mapper_model or config.get("mapper_model", DEFAULT_MAPPER_MODEL)),
        )
    for segment in segments:
        print(
            f"t={segment['start']}s  {segment['vidosc_state']:<14} "
            f"\"{segment['claude_description'][:48]}\" -> {segment['mapped_moscato_state']}"
        )

    # -- 7. MOSCATO event predictions -------------------------------------
    events = segments_to_events.build_events(
        video_id=video_id,
        target_object=target_object,
        clip_start_global=clip_start,
        num_local_frames=num_local_frames,
        fps=fps,
        num_timesteps=num_timesteps,
        segments=segments,
    )
    events["provenance"].update(mapping_provenance)
    write_json(run_dir / "moscato_predictions.json", events)

    # -- 8. per-frame CSV (no ground truth) -------------------------------
    dense = segments_to_events.dense_states(events)
    timestep_segment = {}
    for segment in segments:
        for timestep in range(int(segment["start"]), int(segment["end"]) + 1):
            timestep_segment[timestep] = segment

    import csv

    csv_path = run_dir / "per_frame_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_COLUMNS)
        writer.writeheader()
        for row in frame_mapping.frame_rows(mapping):
            segment = timestep_segment[row["vidosc_timestep"]]
            writer.writerow(
                {
                    **row,
                    "vidosc_state": segment["vidosc_state"],
                    "claude_description": segment["claude_description"],
                    "mapped_moscato_state": dense[row["local_frame"]],
                }
            )
    print(f"Per-frame predictions: {csv_path}")

    # -- 9. visualizations -------------------------------------------------
    vis_dir = run_dir / "visualizations"
    vidosc_per_frame = [
        timestep_segment[
            frame_mapping.timestep_for_local_frame(local, fps, num_timesteps)
        ]["vidosc_state"]
        for local in range(num_local_frames)
    ]
    try:
        visualize.timeline(
            out_path=vis_dir / "timeline.png",
            fps=fps,
            num_local_frames=num_local_frames,
            vidosc_per_frame=vidosc_per_frame,
            mapped_per_frame=dense,
            segments=segments,
            title=f"{video_id} - {target_object} - global [{clip_start}, {clip_end})",
        )
        visualize.representative_frames(
            out_path=vis_dir / "representative_frames.jpg",
            segments=segments,
            frames_dir=frames_dir,
        )
        print(f"Visualizations: {vis_dir}")
    except ImportError as error:
        print(f"Skipping visualizations ({error}); pip install matplotlib")

    # -- 10. manifest ------------------------------------------------------
    manifest = {
        "schema_version": 1,
        "run_name": resolved_run,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "method": "vidosc+claude",
        "dataset": config.get("dataset", "MOSCATO/CMU-MMAC"),
        "video_id": video_id,
        "target_object": target_object,
        "clip_start_global": clip_start,
        "clip_end_global": clip_end,
        "num_local_frames": num_local_frames,
        "fps": fps,
        "num_vidosc_timesteps": num_timesteps,
        "dry_run": dry_run,
        "config_path": str(config_path),
        "source_video_sha256": prepare_metadata.get("source_video_sha256"),
        "clip_video": str(clip_video),
        "vidosc": {
            "decode": decode_info,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": sha256(ckpt),
            "checkpoint_expects_input_dim": ckpt_input_dim,
            "checkpoint_osc_note": (
                "The released VidOSC checkpoints (chopping/browning/rolling) were "
                "trained on HowToChange OSC categories. None of them corresponds to "
                "the sandwich-assembly transformation MOSCATO annotates for this "
                "clip, so the temporal segmentation is out-of-domain."
            ),
            "upstream_repo_commit": _git_commit(Path(os.environ.get("VIDOSC_ROOT", Path(__file__).resolve().parent.parent.parent / "VidOSC"))),
            "vocab_size": int(ckpt_raw["model.head.weight"].shape[0]),
        },
        "features": {
            "source": "extracted by this repo (extract_features.py)",
            "backbone": "CLIP ViT-L/14 (openai weights)",
            "trained_with_backbone": "InternVideo-MM-L14",
            "backbone_mismatch": True,
            "dim": feature_dim,
            "object_centric_features_used": bool(obj_feat),
            "object_centric_features_zero_padded": obj_feat_padded,
            "sample_rate": "1 fps (ffmpeg fps=1 filter over the clip MP4)",
            "path": str(feat_path),
            "warning": (
                "Two stacked approximations: CLIP stands in for InternVideo, and "
                "the object-centric half of the 1536-dim input is zero-filled."
                if obj_feat_padded
                else None
            ),
        },
        "claude": {
            "backend": "anthropic",
            "description_model": vlm_model_name,
            "sample_rate": effective_sample_rate,
            "skip_states": "",
            "prompt_mode": effective_prompt_mode,
            "vidosc_state_shown_to_vlm": effective_prompt_mode == "stated_state",
            "ground_truth_shown_to_vlm": False,
        },
        "state_mapping": mapping_provenance,
        "temporal_alignment": mapping["rules"],
        "integrity": {
            "ground_truth_used_for_inference": False,
            "ground_truth_available_to_predict_stage": False,
            "predictions_written_before_evaluation": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "repo_commit": _git_commit(Path(__file__).resolve().parent.parent),
        },
        "artifacts": sorted(
            str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
        ),
    }
    write_json(run_dir / "manifest.json", manifest)

    # -- 11. W&B ------------------------------------------------------------
    if use_wandb:
        from moscato.log_wandb import log_prediction

        try:
            result = log_prediction(
                manifest=manifest,
                segments=segments,
                vidosc_predictions=load_json(run_dir / "vidosc_predictions.json")["predictions"],
                project=wandb_project,
                entity=wandb_entity,
                run_name=resolved_run,
                run_dir=run_dir,
            )
            print(f"W&B (prediction): {result['run_url']}")
        except Exception as error:
            # Logging must never destroy a completed run's artifacts.
            print(f"W&B logging skipped: {type(error).__name__}: {error}")

    print(f"\nRun directory: {run_dir}")
    return manifest


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def evaluate(
    config_path: Path,
    *,
    run_name: str,
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
) -> dict[str, Any]:
    config = load_config(config_path)
    resolved_run = _run_name(run_name)
    run_dir = resolve(config, "runs_dir") / resolved_run
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"{run_dir} does not exist; pass the --run-name used by the predict stage."
        )
    summary = score_run(config_path, run_dir)

    print(json.dumps({k: v for k, v in summary.items() if k != "f1_threshold_curve"}, indent=2))

    # Regenerate the timeline with the ground-truth lane now that it is allowed.
    try:
        import csv

        with (run_dir / "per_frame_predictions_scored.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        segments = load_json(run_dir / "vidosc_segments.json")["segments"]
        for segment in segments:
            segment.setdefault("claude_description", "")
        visualize.timeline(
            out_path=run_dir / "visualizations" / "timeline_with_gt.png",
            fps=int(config["fps"]),
            num_local_frames=len(rows),
            vidosc_per_frame=[row["vidosc_state"] for row in rows],
            mapped_per_frame=[row["mapped_moscato_state"] for row in rows],
            segments=segments,
            gt_per_frame=[(row["ground_truth_state"].split("|")[0] or "(none)") for row in rows],
            title=f"{config['video_id']} - {config['target_object']} - prediction vs. MOSCATO GT",
        )
    except ImportError as error:
        print(f"Skipping GT timeline ({error}); pip install matplotlib")

    if use_wandb:
        from moscato.log_wandb import log_evaluation, run_config

        with (run_dir / "per_frame_predictions_scored.csv").open(encoding="utf-8", newline="") as handle:
            import csv

            rows = list(csv.DictReader(handle))
        result = log_evaluation(
            summary=summary,
            rows=rows[:: max(1, len(rows) // 4000)],
            project=wandb_project,
            entity=wandb_entity,
            run_name=f"vidosc-claude-{resolved_run}",
            artifact_paths={
                "manifest": run_dir / "manifest.json",
                "predictions": run_dir / "moscato_predictions.json",
                "evaluation": run_dir / "evaluation.json",
                "frames": run_dir / "per_frame_predictions_scored.csv",
            },
            config=run_config(load_json(run_dir / "manifest.json")),
        )
        print(json.dumps(result, indent=2))

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["inspect", "prepare", "predict", "evaluate", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, help="VidOSC checkpoint (required for predict)")
    parser.add_argument("--obj_feat", type=Path, default=None, help="Object-centric features for det=1 checkpoints")
    parser.add_argument("--run-name", default="", help="Run directory name (default: UTC timestamp)")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--mapper-model", default=None)
    parser.add_argument("--sample-rate", choices=["per_segment", "per_second"], default=None)
    parser.add_argument("--decode", dest="decode_mode", choices=["argmax", "ordered"], default=None,
                        help="argmax (default): independent per-timestep argmax, the pre-existing "
                             "baseline. ordered: VidOSC's ordering-constrained DP, which forbids "
                             "backward state transitions.")
    parser.add_argument("--prompt-mode", choices=["neutral", "stated_state"], default=None,
                        help="neutral (default): withhold the VidOSC label from Claude. "
                             "stated_state: upstream behaviour, names the label in the prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Skip every VLM call")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip Weights & Biases logging (on by default)")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT, choices=WANDB_PROJECTS,
                        help=f"W&B project (default: {WANDB_PROJECT!r}; "
                             "use 'vidosc - full evals' for full evaluations)")
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY,
                        help=f"W&B entity (default: {WANDB_ENTITY!r})")
    args = parser.parse_args()

    resolved_run = _run_name(args.run_name)

    if args.stage in ("inspect", "all"):
        print(json.dumps(validate_alignment(args.config), indent=2))
    if args.stage in ("prepare", "all"):
        prepare_clip.prepare(args.config)
    if args.stage in ("predict", "all"):
        if not args.ckpt:
            parser.error("--ckpt is required for the predict stage")
        predict(
            args.config,
            ckpt=args.ckpt,
            run_name=resolved_run,
            obj_feat=args.obj_feat,
            vlm_model=args.vlm_model,
            mapper_model=args.mapper_model,
            sample_rate=args.sample_rate,
            prompt_mode=args.prompt_mode,
            decode_mode=args.decode_mode,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            dry_run=args.dry_run,
            force=args.force,
        )
    if args.stage in ("evaluate", "all"):
        evaluate(
            args.config,
            run_name=resolved_run,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )


if __name__ == "__main__":
    main()
