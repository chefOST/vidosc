"""
VLM captioning on top of VidOSC's per-second state predictions.

Runs the existing VidOSC inference (see run_inference.py) to get a per-second
background/initial_state/transitioning/end_state sequence for a clip, groups
consecutive seconds that share a predicted state into contiguous segments, and
sends one representative frame per segment to a vision-language model with an
object-specific prompt asking it to describe what the object actually looks
like at that point (texture, color, shape).

This script is read-only with respect to the rest of the VidOSC pipeline: it
imports `run_inference()` and `extract_frames_1fps()` rather than duplicating
or modifying them, and never touches task.py / dataset.py / evaluator.py /
read_ann.py.

Usage:
    # Cheap prototyping with a hosted API model:
    python describe_states.py \
        --feat data/feats/browning_apple.pth.tar \
        --ckpt checkpoints/browning.ckpt \
        --video data/clips/browning_apple/clip_st0_dur20.mp4 \
        --vlm_backend claude

    # Local LLaVA-NeXT:
    python describe_states.py \
        --feat data/feats/browning_apple.pth.tar \
        --ckpt checkpoints/browning.ckpt \
        --video data/clips/browning_apple/clip_st0_dur20.mp4 \
        --vlm_backend llava

    # Finer-grained (one VLM call per second instead of per segment):
    python describe_states.py ... --sample_rate per_second

    # See the segments/frames that would be sent, without calling any VLM
    # (no extra deps required):
    python describe_states.py ... --dry_run
"""
import argparse
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from extract_features import extract_frames_1fps
from run_inference import (
    STATE_NAMES,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_PROJECTS,
    run_inference as vidosc_run_inference,
)

DEFAULT_VLM_MODELS = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "llava": "llava-hf/llava-v1.6-mistral-7b-hf",
}


# --------------------------------------------------------------------------
# Object name derivation (reads VidOSC's "osc" folder convention, e.g.
# browning_apple -> object "apple"; see read_ann.py / dataset.py / pseudo_label.py)
# --------------------------------------------------------------------------

def derive_object_name(video_path: str, osc_override: str | None, object_override: str | None) -> tuple[str, str | None]:
    """Return (object_name, osc) for prompt construction.

    Priority: --object > --osc > parent directory of --video (VidOSC's
    convention of storing clips under <clip_dir>/<osc>/<file>.mp4, where
    osc = "<verb>_<object>", e.g. "browning_apple").
    """
    osc = osc_override
    if osc is None:
        parent = Path(video_path).resolve().parent.name
        if "_" in parent:
            osc = parent

    if object_override:
        return object_override, osc

    if osc and "_" in osc:
        _verb, obj = osc.split("_", 1)
        return obj.replace("_", " "), osc

    # Fall back to the clip's filename stem; not reliable, so warn.
    stem = Path(video_path).stem
    print(
        f"Warning: could not infer object name from '{video_path}' "
        f"(expected parent dir like 'browning_apple'). Falling back to "
        f"'{stem}'. Pass --object to override.",
        flush=True,
    )
    return stem, osc


# --------------------------------------------------------------------------
# Segmenting the per-second state sequence
# --------------------------------------------------------------------------

def group_segments(predicted: np.ndarray) -> list[tuple[int, int, int]]:
    """Group consecutive seconds sharing the same predicted state.

    Returns a list of (start_sec, end_sec, state_idx), end_sec inclusive.
    """
    segments = []
    n = len(predicted)
    start = 0
    for t in range(1, n + 1):
        if t == n or predicted[t] != predicted[start]:
            segments.append((start, t - 1, int(predicted[start])))
            start = t
    return segments


def expand_to_per_second(segments: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Split each (start, end, state) segment into one-second segments."""
    out = []
    for start, end, state_idx in segments:
        for t in range(start, end + 1):
            out.append((t, t, state_idx))
    return out


# --------------------------------------------------------------------------
# VLM backends
# --------------------------------------------------------------------------

def build_prompt(object_name: str, state_name: str) -> str:
    readable_state = state_name.replace("_", " ")
    article = "an" if object_name[:1].lower() in "aeiou" else "a"
    return (
        f"This frame is from a cooking video showing {article} {object_name} "
        f"during the '{readable_state}' stage of a state change. "
        f"Describe the visual state of the {object_name} in this image in a "
        f"few words — texture, color, shape changes. One concise phrase, no "
        f"preamble."
    )


def frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ClaudeBackend:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 80):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for --vlm_backend claude. "
                "Install with: pip install anthropic"
            ) from exc
        self.client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY env var
        self.model = model
        self.max_tokens = max_tokens

    def describe(self, frame: np.ndarray, object_name: str, state_name: str) -> str:
        img_b64 = base64.b64encode(frame_to_jpeg_bytes(frame)).decode("ascii")
        prompt = build_prompt(object_name, state_name)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


class OpenAIBackend:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 80):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for --vlm_backend openai. "
                "Install with: pip install openai"
            ) from exc
        self.client = OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY env var
        self.model = model
        self.max_tokens = max_tokens

    def describe(self, frame: np.ndarray, object_name: str, state_name: str) -> str:
        img_b64 = base64.b64encode(frame_to_jpeg_bytes(frame)).decode("ascii")
        prompt = build_prompt(object_name, state_name)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()


class LlavaBackend:
    """Local LLaVA-NeXT inference (matches the model family used elsewhere in chefOST)."""

    def __init__(self, model: str, max_tokens: int = 80):
        try:
            import torch
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' (and 'torch') packages are required for "
                "--vlm_backend llava. Install with: pip install transformers accelerate"
            ) from exc

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._torch = torch
        self.processor = LlavaNextProcessor.from_pretrained(model)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        self.max_tokens = max_tokens

    def describe(self, frame: np.ndarray, object_name: str, state_name: str) -> str:
        prompt = build_prompt(object_name, state_name)
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        decoded = self.processor.decode(output[0], skip_special_tokens=True)

        # LLaVA-NeXT chat templates echo the prompt; take whatever follows the
        # last assistant turn marker if present, else the full decoded text.
        for marker in ("ASSISTANT:", "[/INST]"):
            if marker in decoded:
                decoded = decoded.rsplit(marker, 1)[-1]
        return decoded.strip()


def build_backend(args):
    model = args.vlm_model or DEFAULT_VLM_MODELS[args.vlm_backend]
    if args.vlm_backend == "claude":
        return ClaudeBackend(model, api_key=args.api_key, max_tokens=args.max_tokens)
    if args.vlm_backend == "openai":
        return OpenAIBackend(model, api_key=args.api_key, max_tokens=args.max_tokens)
    if args.vlm_backend == "llava":
        return LlavaBackend(model, max_tokens=args.max_tokens)
    raise ValueError(f"Unknown --vlm_backend {args.vlm_backend!r}")


# --------------------------------------------------------------------------
# W&B logging
# --------------------------------------------------------------------------

def _log_captions_to_wandb(
    *,
    feat_path: str,
    video_path: str,
    object_name: str,
    vlm_backend: str,
    vlm_model: str,
    sample_rate: str,
    results: list[dict],
    entity: str,
    project: str,
    run_name: str | None,
) -> str | None:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required for --wandb. Install with: pip install wandb") from exc

    video_id = Path(feat_path).stem
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name or f"vidosc-describe-{vlm_backend}-{video_id}",
        tags=["vidosc", "describe_states", vlm_backend],
        config={
            "feat_path": feat_path,
            "video_path": video_path,
            "object_name": object_name,
            "vlm_backend": vlm_backend,
            "vlm_model": vlm_model,
            "sample_rate": sample_rate,
        },
    )

    table = wandb.Table(columns=["segment", "start_s", "end_s", "state", "description", "frame"])
    for r in results:
        label = f"t={r['start']}s" if r["start"] == r["end"] else f"t={r['start']}-{r['end']}s"
        table.add_data(
            label, r["start"], r["end"], r["state"], r["description"],
            wandb.Image(r["frame"], caption=r["description"]),
        )
    wandb.log({"state_descriptions": table})

    url = run.url
    wandb.finish()
    return url


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def describe_states(args):
    predicted, _probs = vidosc_run_inference(
        args.feat, args.ckpt, args.obj_feat, use_wandb=False,
    )

    print(f"\nExtracting frames from {args.video} ...")
    frames = extract_frames_1fps(args.video)

    n = min(len(frames), len(predicted))
    if len(frames) != len(predicted):
        print(
            f"Warning: {len(frames)} frames vs {len(predicted)} predicted "
            f"seconds; truncating both to {n}.",
            flush=True,
        )
    frames, predicted = frames[:n], predicted[:n]

    object_name, osc = derive_object_name(args.video, args.osc, args.object)
    print(f"Object: '{object_name}'" + (f"  (osc={osc})" if osc else ""))

    segments = group_segments(predicted)
    if args.sample_rate == "per_second":
        segments = expand_to_per_second(segments)

    skip_states = {s.strip() for s in args.skip_states.split(",") if s.strip()}

    backend = None if args.dry_run else build_backend(args)

    print(f"\n{len(segments)} segment(s)  (sample_rate={args.sample_rate}, "
          f"skipping: {sorted(skip_states) or 'none'})")
    print("-" * 60)

    results = []
    for start, end, state_idx in segments:
        state_name = STATE_NAMES[state_idx]
        label = f"t={start}s" if start == end else f"t={start}-{end}s"

        if state_name in skip_states:
            print(f"{label:<12} [{state_name:<14}] (skipped)")
            continue

        mid = (start + end) // 2
        frame = frames[mid]

        if args.dry_run:
            desc = "(dry run — no VLM call)"
        else:
            desc = backend.describe(frame, object_name, state_name)

        print(f"{label:<12} [{state_name:<14}] → \"{desc}\"")
        results.append({
            "start": start, "end": end, "state": state_name,
            "description": desc, "frame": frame,
        })

    if args.wandb and not args.dry_run:
        url = _log_captions_to_wandb(
            feat_path=args.feat,
            video_path=args.video,
            object_name=object_name,
            vlm_backend=args.vlm_backend,
            vlm_model=args.vlm_model or DEFAULT_VLM_MODELS[args.vlm_backend],
            sample_rate=args.sample_rate,
            results=results,
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        )
        print(f"\nW&B run: {url or '(offline — run wandb sync wandb/offline-run-* to upload)'}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feat", required=True, help="Path to .pth.tar feature file")
    parser.add_argument("--ckpt", required=True, help="Path to VidOSC .ckpt checkpoint")
    parser.add_argument("--obj_feat", default=None, help="Optional object-centric feature file (.pth.tar) for det=1 checkpoints")
    parser.add_argument("--video", required=True, help="Path to the source clip (same video --feat was extracted from)")

    parser.add_argument("--object", default=None, help="Object name for the VLM prompt (e.g. 'apple'); overrides --osc/auto-detection")
    parser.add_argument("--osc", default=None, help="Override the osc category (e.g. 'browning_apple') used to derive the object name")

    parser.add_argument("--sample_rate", choices=["per_segment", "per_second"], default="per_segment",
                        help="One VLM call per contiguous state segment (default), or one per second (more expensive)")
    parser.add_argument("--skip_states", default="background",
                        help="Comma-separated state names to skip captioning (default: 'background'; pass '' to caption every state)")

    parser.add_argument("--vlm_backend", choices=["claude", "openai", "llava"], default="claude",
                        help="claude/openai = cheap hosted API prototyping; llava = local LLaVA-NeXT")
    parser.add_argument("--vlm_model", default=None, help=f"Override the default model per backend: {DEFAULT_VLM_MODELS}")
    parser.add_argument("--api_key", default=None, help="API key for claude/openai backends (defaults to ANTHROPIC_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--max_tokens", type=int, default=80, help="Max tokens for each VLM description")

    parser.add_argument("--dry_run", action="store_true", help="Print segments/frames without calling any VLM (no extra deps needed)")

    parser.add_argument("--wandb", action="store_true", default=bool(os.getenv("WANDB_API_KEY")),
                        help="Log results to Weights & Biases (default: on when WANDB_API_KEY is set)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY, help="W&B entity")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT, choices=WANDB_PROJECTS,
                        help="W&B project")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")

    args = parser.parse_args()
    args.wandb = args.wandb and not args.no_wandb
    describe_states(args)


if __name__ == "__main__":
    main()
