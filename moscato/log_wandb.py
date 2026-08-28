#!/usr/bin/env python3
"""Publish MOSCATO runs to Weights & Biases.

Ported from chefOST-ml (branch feat/tubeletgraph-moscato-cmu-s12),
``scripts/log_wandb.py``. The ``word_accuracy/*`` metric namespace and the
threshold-curve / frame-table layout are unchanged, so a VidOSC+Claude run and a
TubeletGraph run land in comparable W&B panels.

Two additions on top of the port:

* :func:`log_prediction` logs the *predict* stage — decode mode, segments,
  Claude descriptions, timeline images. The original only logged evaluation,
  which meant a run that failed before scoring left no trace.
* Credentials no longer require ``WANDB_API_KEY`` specifically. The original
  ran on Modal where a secret was injected; locally ``wandb login`` writes
  ``~/.netrc`` instead, and demanding the env var rejected a perfectly good
  login. :func:`assert_wandb_credentials` accepts either.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


# Defaults for this project. Note the project name contains a literal "+";
# the W&B UI shows it URL-encoded as %2B, which is not the name.
WANDB_ENTITY = "chefOST"
WANDB_PROJECT = "VidOSC + MOSCATO"

PRIMARY_METRICS = ("precision", "accuracy", "f1", "f1_max")


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


def metric_payload(summary: dict[str, Any]) -> dict[str, float]:
    """Map evaluation output to one clearly grouped W&B metric namespace."""
    missing = [key for key in PRIMARY_METRICS if key not in summary]
    if missing:
        raise KeyError(f"Evaluation summary is missing metrics: {missing}")
    metrics = {
        f"word_accuracy/{key}": float(summary[key]) for key in PRIMARY_METRICS
    }
    metrics.update(
        {
            "word_accuracy/f1_max_threshold": float(
                summary["f1_max_threshold"]
            ),
            "word_accuracy/evaluated_frames": float(
                summary["evaluated_frames"]
            ),
        }
    )
    return metrics


def _artifact_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not component:
        raise ValueError(f"Cannot form a W&B artifact component from {value!r}")
    return component[:128]


def log_evaluation(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    project: str,
    entity: str,
    run_name: str,
    artifact_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, str]:
    """Log scalar cards, threshold curve, frame table, and source artifacts."""
    assert_wandb_credentials()

    import wandb

    metrics = metric_payload(summary)
    resolved_name = _artifact_component(run_name)
    with wandb.init(
        project=project,
        entity=entity or None,
        name=resolved_name,
        job_type="evaluation",
        config=config,
        # decode:/prompt: mirror the prediction run's tags so a comparison can be
        # filtered on one axis across both job types.
        tags=[
            "vidosc", "claude", "moscato", "cmu", "evaluation",
            f"decode:{config.get('decode_mode', 'unknown')}",
            f"prompt:{config.get('prompt_mode', 'unknown')}",
        ],
    ) as run:
        for key, value in metrics.items():
            run.summary[key] = value
        run.log(metrics)

        curve = summary["f1_threshold_curve"]
        curve_table = wandb.Table(
            columns=["threshold", "precision", "recall", "f1"],
            data=[
                [
                    point["threshold"],
                    point["precision"],
                    point["recall"],
                    point["f1"],
                ]
                for point in curve
            ],
        )
        log_items: dict[str, Any] = {
            "word_accuracy/f1_threshold_curve": curve_table,
            "word_accuracy/f1_by_threshold": wandb.plot.line(
                curve_table,
                "threshold",
                "f1",
                title="MOSCATO micro-F1 by global threshold",
            ),
        }
        if rows:
            columns = list(rows[0])
            log_items["evaluation/frame_predictions"] = wandb.Table(
                columns=columns,
                data=[[row[column] for column in columns] for row in rows],
            )
        run.log(log_items)

        artifact_name = _artifact_component("-".join(
            [
                "moscato-evaluation",
                _artifact_component(str(summary["video_id"])),
                _artifact_component(str(summary["object"])),
                resolved_name,
            ]
        ))
        artifact = wandb.Artifact(
            artifact_name,
            type="evaluation",
            metadata={
                "video_id": summary["video_id"],
                "object": summary["object"],
                "metrics": metrics,
                "ground_truth_used_for_inference": False,
            },
        )
        for artifact_key, path in artifact_paths.items():
            if path.is_file():
                artifact.add_file(str(path), name=f"{artifact_key}/{path.name}")
        logged_artifact = run.log_artifact(artifact, aliases=["latest"])
        result = {
            "run_url": run.url,
            "run_path": run.path,
            "artifact_name": logged_artifact.name,
        }
    return result


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
