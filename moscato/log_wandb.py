#!/usr/bin/env python3
'Publish saved prediction artifacts to Weights & Biases. Benchmark scoring lives in chefOST/chefOST-data.'

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


# chefOST/vidosc W&B destinations. Development runs go to experiments; pass
# --wandb-project "vidosc - full evals" only for intended full evaluations.
# The space in the full-evals name is part of the name.
WANDB_ENTITY = "chefOST"
WANDB_PROJECTS = ("vidosc-experiments", "vidosc - full evals")
WANDB_PROJECT = WANDB_PROJECTS[0]



def assert_wandb_credentials() -> str:
    """Return how W&B will authenticate, or raise with how to fix it."""
    if os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY"
    netrc = Path.home() / ".netrc"
    if netrc.is_file() and "api.wandb.ai" in netrc.read_text(errors="ignore"):
        return "~/.netrc (wandb login)"
    raise RuntimeError(
        "No W&B credentials found. Run `wandb login`, or export WANDB_API_KEY. "
        "Pass --no-wandb to skip logging entirely."
    )


def run_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Flatten the manifest into the axes worth grouping and filtering runs by.

    These are the knobs that differ between comparable runs -- decode mode,
    prompt mode, checkpoint, clip -- so they belong in W&B config rather than
    buried in an artifact.
    """
    vidosc = manifest.get("vidosc", {})
    claude = manifest.get("claude", {})
    features = manifest.get("features", {})
    return {
        "video_id": manifest.get("video_id"),
        "target_object": manifest.get("target_object"),
        "dataset": manifest.get("dataset"),
        "clip_start_global": manifest.get("clip_start_global"),
        "clip_end_global": manifest.get("clip_end_global"),
        "num_local_frames": manifest.get("num_local_frames"),
        "num_vidosc_timesteps": manifest.get("num_vidosc_timesteps"),
        "fps": manifest.get("fps"),
        "decode_mode": vidosc.get("decode", {}).get("mode"),
        "num_segments": vidosc.get("decode", {}).get("num_segments"),
        "checkpoint": Path(str(vidosc.get("checkpoint", ""))).name,
        "prompt_mode": claude.get("prompt_mode"),
        "vlm_model": claude.get("description_model"),
        "mapper_model": manifest.get("state_mapping", {}).get("state_mapping_model"),
        "sample_rate": claude.get("sample_rate"),
        "feature_backbone": features.get("backbone"),
        "object_features_zero_padded": features.get("object_centric_features_zero_padded"),
        "dry_run": manifest.get("dry_run"),
    }




def _artifact_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not component:
        raise ValueError(f"Cannot form a W&B artifact component from {value!r}")
    return component[:128]




def log_prediction(
    *,
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    vidosc_predictions: list[dict[str, Any]],
    project: str,
    entity: str,
    run_name: str,
    run_dir: Path,
) -> dict[str, str]:
    """Log the predict stage: decode, segments, Claude descriptions, timelines.

    Ground truth is not available at this point and none is logged. The
    evaluation stage logs the metrics separately, into its own run.
    """
    assert_wandb_credentials()

    import wandb

    decode = manifest.get("vidosc", {}).get("decode", {})
    resolved_name = _artifact_component(run_name)
    with wandb.init(
        project=project,
        entity=entity or None,
        name=f"{resolved_name}-predict",
        job_type="prediction",
        config=run_config(manifest),
        tags=[
            "vidosc", "claude", "moscato", "prediction",
            f"decode:{decode.get('mode', 'unknown')}",
            f"prompt:{manifest.get('claude', {}).get('prompt_mode', 'unknown')}",
        ],
    ) as run:
        # Headline numbers, so argmax-vs-ordered is visible without opening a table.
        for key, value in {
            "decode/num_segments": decode.get("num_segments"),
            "decode/num_timesteps": decode.get("num_timesteps"),
            "decode/emits_background": int(bool(decode.get("emits_background"))),
            "claude/num_calls": len(segments),
        }.items():
            if value is not None:
                run.summary[key] = value

        states = [entry["state"] for entry in vidosc_predictions]
        for name in ("background", "initial_state", "transitioning", "end_state"):
            run.summary[f"vidosc/seconds_{name}"] = states.count(name)
        run.summary["vidosc/transitioning_selected"] = int("transitioning" in states)

        indices = decode.get("detection_indices") or {}
        for state, index in indices.items():
            run.summary[f"decode/detection_index_{state}"] = index

        # Per-second predictions, with the raw argmax alongside so the effect of
        # the decode mode is inspectable rather than asserted.
        per_second = wandb.Table(
            columns=["timestep", "decoded_state", "argmax_state", "confidence"],
            data=[
                [
                    entry["vidosc_timestep"],
                    entry["state"],
                    entry.get("argmax_state", ""),
                    entry["confidence"],
                ]
                for entry in vidosc_predictions
            ],
        )

        segment_table = wandb.Table(
            columns=[
                "segment", "start_s", "end_s", "vidosc_state",
                "representative_local_frame", "representative_global_frame",
                "claude_description", "mapped_moscato_state", "frame",
            ]
        )
        frames_dir = Path(manifest.get("clip_video", "")).parent / "frames"
        for index, segment in enumerate(segments):
            local = int(segment["representative_local_frame"])
            frame_path = frames_dir / f"{local:07d}.jpg"
            image = (
                wandb.Image(
                    str(frame_path),
                    caption=segment.get("claude_description", "")[:180],
                )
                if frame_path.is_file()
                else None
            )
            segment_table.add_data(
                index,
                segment["start"],
                segment["end"],
                segment["vidosc_state"],
                local,
                segment["representative_global_frame"],
                segment.get("claude_description", ""),
                segment.get("mapped_moscato_state", ""),
                image,
            )

        payload: dict[str, Any] = {
            "predictions/per_second": per_second,
            "predictions/segments": segment_table,
        }
        timeline = run_dir / "visualizations" / "timeline.png"
        if timeline.is_file():
            payload["predictions/timeline"] = wandb.Image(
                str(timeline), caption=f"{run_name} — decode={decode.get('mode')}"
            )
        frames_sheet = run_dir / "visualizations" / "representative_frames.jpg"
        if frames_sheet.is_file():
            payload["predictions/representative_frames"] = wandb.Image(str(frames_sheet))
        run.log(payload)

        artifact = wandb.Artifact(
            _artifact_component(f"moscato-prediction-{resolved_name}"),
            type="prediction",
            metadata={**run_config(manifest), "ground_truth_used_for_inference": False},
        )
        for name in (
            "manifest.json", "frame_mapping.json", "vidosc_predictions.json",
            "vidosc_segments.json", "claude_raw.json", "moscato_predictions.json",
            "per_frame_predictions.csv",
        ):
            path = run_dir / name
            if path.is_file():
                artifact.add_file(str(path), name=name)
        run.log_artifact(artifact, aliases=["latest"])
        return {"run_url": run.url, "run_path": run.path}
