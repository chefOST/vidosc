"""Visual sanity checks to run *before* trusting any metric.

Produces two artifacts in ``<run_dir>/visualizations/``:

``timeline.png``
    Stacked lanes over clip time -- VidOSC temporal states, Claude's
    descriptions at their segment boundaries, and the mapped MOSCATO states.
    The evaluation stage regenerates it with a fourth ground-truth lane.

``representative_frames.jpg``
    The actual frames sent to Claude, captioned with their local/global frame
    index and the state that came back. This is how you check that Claude was
    describing the frames you think it was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

VIDOSC_COLORS = {
    "background": "#9e9e9e",
    "initial_state": "#4caf50",
    "transitioning": "#ff9800",
    "end_state": "#2196f3",
}
# Deliberately avoids the VIDOSC_COLORS hues so a MOSCATO state can never be
# mistaken for a VidOSC temporal state in the shared legend.
PALETTE = [
    "#7b3294", "#c2a5cf", "#d01c8b", "#5e3c99", "#e66101",
    "#b2182b", "#4d004b", "#8073ac", "#f1b6da", "#542788",
]


def _state_colors(states: list[str]) -> dict[str, str]:
    unique = sorted({state for state in states if state})
    return {state: PALETTE[index % len(PALETTE)] for index, state in enumerate(unique)}


def _draw_lane(ax, spans, colors, label, fps):
    for start, end, value in spans:
        ax.barh(
            0, (end - start) / fps, left=start / fps, height=0.8,
            color=colors.get(value, "#cccccc"), edgecolor="white", linewidth=0.4,
        )
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, labelpad=12)
    ax.spines[:].set_visible(False)


def _runs(values: list[str]) -> list[tuple[int, int, str]]:
    """Collapse a per-frame sequence into (start, end_exclusive, value) runs."""
    spans = []
    if not values:
        return spans
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            spans.append((start, index, values[start]))
            start = index
    return spans


def timeline(
    *,
    out_path: Path,
    fps: int,
    num_local_frames: int,
    vidosc_per_frame: list[str],
    mapped_per_frame: list[str],
    segments: list[dict[str, Any]],
    gt_per_frame: list[str] | None = None,
    title: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lanes = 3 + (1 if gt_per_frame is not None else 0)
    duration = num_local_frames / fps
    fig, axes = plt.subplots(
        lanes, 1, figsize=(max(12, duration * 0.1), 1.5 * lanes + 1.6), sharex=True
    )

    mapped_colors = _state_colors(mapped_per_frame + (gt_per_frame or []))

    _draw_lane(axes[0], _runs(vidosc_per_frame), VIDOSC_COLORS, "VIDOSC", fps)
    axes[0].set_title(title or "VidOSC + Claude on MOSCATO", fontsize=11, pad=14)

    # Claude lane: one labelled marker per segment boundary.
    axes[1].set_yticks([])
    axes[1].set_ylabel("CLAUDE", rotation=0, ha="right", va="center", fontsize=9, labelpad=12)
    axes[1].spines[:].set_visible(False)
    axes[1].set_ylim(0, 1)
    for index, segment in enumerate(segments):
        start_s = int(segment["start"])
        axes[1].axvline(start_s, color="#666666", linewidth=0.7, alpha=0.6)
        text = (segment.get("claude_description") or "")[:44]
        axes[1].text(
            start_s, 0.62 if index % 2 == 0 else 0.18, f"{start_s}s  {text}",
            fontsize=6.5, va="center", ha="left", color="#222222",
        )

    _draw_lane(axes[2], _runs(mapped_per_frame), mapped_colors, "MOSCATO\nmapped", fps)
    if gt_per_frame is not None:
        _draw_lane(axes[3], _runs(gt_per_frame), mapped_colors, "MOSCATO\nGT", fps)

    axes[-1].set_xlabel("Clip time (seconds)")
    axes[-1].set_xlim(0, duration)

    handles = [Patch(color=color, label=name) for name, color in VIDOSC_COLORS.items()]
    handles += [Patch(color=color, label=name) for name, color in mapped_colors.items()]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def representative_frames(
    *, out_path: Path, segments: list[dict[str, Any]], frames_dir: Path, max_frames: int = 24
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    shown = segments[:max_frames]
    columns = min(6, max(1, len(shown)))
    rows = (len(shown) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.9 * rows))
    axes = [axes] if rows * columns == 1 else list(axes.flat)

    for ax in axes:
        ax.axis("off")
    for ax, segment in zip(axes, shown):
        local = int(segment["representative_local_frame"])
        ax.imshow(Image.open(frames_dir / f"{local:07d}.jpg"))
        ax.set_title(
            f"t={segment['start']}s  local {local} / global "
            f"{segment['representative_global_frame']}\n"
            f"{segment['vidosc_state']} -> {segment.get('mapped_moscato_state', '?')}",
            fontsize=7,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
