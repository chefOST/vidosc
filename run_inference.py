"""
Run VidOSC inference on pre-extracted features without needing the CSV dataset.

Loads a pytorch-lightning checkpoint, runs a forward pass on a .pth.tar feature
file, and prints per-second state predictions. Optionally logs results to W&B.

Usage:
    python run_inference.py \
        --feat data/feats/dicing_onion.pth.tar \
        --ckpt checkpoints/chopping.ckpt

    # W&B (requires WANDB_API_KEY in env, or wandb login):
    python run_inference.py \
        --feat data/feats/dicing_onion.pth.tar \
        --ckpt checkpoints/chopping.ckpt \
        --wandb

States:
    0 = background (no state change happening)
    1 = initial_state  (object before transformation)
    2 = transitioning  (change in progress)
    3 = end_state      (object after transformation)
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Upstream VidOSC lives in a separate checkout, not in this repo -- see README.md.
from vidosc_path import ensure_on_path

ensure_on_path()
from model import FeatTimeTransformer

STATE_NAMES = {
    0: "background",
    1: "initial_state",
    2: "transitioning",
    3: "end_state",
}
STATE_SYMBOLS = {0: "·", 1: "█", 2: "▒", 3: "░"}
STATE_COLORS = {
    0: "#9e9e9e",
    1: "#4caf50",
    2: "#ff9800",
    3: "#2196f3",
}

# chefOST/vidosc W&B destinations. Development runs go to experiments; pass
# --wandb-project "vidosc - full evals" only for intended full evaluations.
WANDB_ENTITY = "chefOST"
WANDB_PROJECTS = ("vidosc-experiments", "vidosc - full evals")
WANDB_PROJECT = WANDB_PROJECTS[0]


def _checkpoint_config(ckpt_path: str) -> tuple[int, int]:
    """Return (input_dim, vocab_size) inferred from checkpoint weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    proj_weight = raw["model.proj.weight"]
    head_weight = raw["model.head.weight"]
    return proj_weight.shape[1], head_weight.shape[0]


def load_model(ckpt_path: str, input_dim: int | None = None, vocab_size: int | None = None) -> FeatTimeTransformer:
    """Extract FeatTimeTransformer weights from a pytorch-lightning checkpoint."""
    import argparse as _ap

    ckpt_input_dim, ckpt_vocab_size = _checkpoint_config(ckpt_path)
    input_dim = input_dim or ckpt_input_dim
    vocab_size = vocab_size or ckpt_vocab_size
    if input_dim != ckpt_input_dim:
        raise ValueError(
            f"Requested input_dim={input_dim}, but checkpoint expects {ckpt_input_dim}"
        )
    if vocab_size != ckpt_vocab_size:
        raise ValueError(
            f"Requested vocab_size={vocab_size}, but checkpoint expects {ckpt_vocab_size}"
        )

    args = _ap.Namespace(
        input_dim=input_dim,
        vocab_size=vocab_size,
        transformer_heads=4,
        transformer_layers=3,
        transformer_dim=512,
        transformer_dropout=0.1,
        det=1 if input_dim > 768 else 0,
    )
    model = FeatTimeTransformer(args)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    weights = {k[len("model."):]: v for k, v in raw.items()
               if k.startswith("model.")}
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def _prepare_features(feat: torch.Tensor, input_dim: int, obj_feat_path: str | None) -> torch.Tensor:
    """Match training-time feature layout (768-dim video, optionally +768-dim object-centric)."""
    if feat.shape[-1] == input_dim:
        return feat

    if feat.shape[-1] != 768 or input_dim != 1536:
        raise ValueError(
            f"Feature dim {feat.shape[-1]} does not match checkpoint input_dim {input_dim}"
        )

    obj_features = torch.zeros_like(feat)
    if obj_feat_path:
        obj_feat = torch.load(obj_feat_path, map_location="cpu")
        obj_idx_path = obj_feat_path.replace(".pth.tar", ".npy")
        if os.path.exists(obj_idx_path):
            obj_idx = np.load(obj_idx_path)
            obj_idx = obj_idx[obj_idx < len(obj_features)]
            obj_features[obj_idx] = obj_feat[: len(obj_idx)]
        else:
            obj_features[: len(obj_feat)] = obj_feat[: len(obj_features)]
    else:
        print(
            "Warning: checkpoint expects object-centric features (det=1) but none were "
            "provided; padding with zeros (results may be degraded)."
        )

    return torch.cat((feat, obj_features), dim=-1)


def _state_timeline_image(predicted: np.ndarray, n_frames: int):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(max(10, n_frames * 0.12), 1.2))
    for t, state_idx in enumerate(predicted):
        ax.barh(0, 1, left=t, color=STATE_COLORS[state_idx], edgecolor="none")
    ax.set_xlim(0, n_frames)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("VidOSC per-second state predictions")
    ax.legend(
        handles=[Patch(color=STATE_COLORS[i], label=STATE_NAMES[i])
                 for i in STATE_NAMES],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig, plt


def _log_to_wandb(
    *,
    feat_path: str,
    ckpt_path: str,
    obj_feat_path: str | None,
    device: str,
    input_dim: int,
    obj_feat_padded: bool,
    predicted: np.ndarray,
    probs: np.ndarray,
    entity: str,
    project: str,
    run_name: str | None,
) -> str | None:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for --wandb. Install with: pip install wandb"
        ) from exc

    video_id = Path(feat_path).stem
    ckpt_name = Path(ckpt_path).stem
    n_frames = len(predicted)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name or f"vidosc-{ckpt_name}-{video_id}",
        tags=["vidosc", "inference", ckpt_name],
        config={
            "feat_path": feat_path,
            "ckpt_path": ckpt_path,
            "obj_feat_path": obj_feat_path,
            "device": device,
            "input_dim": input_dim,
            "det": 1 if input_dim > 768 else 0,
            "obj_feat_padded": obj_feat_padded,
            "n_frames": n_frames,
            "video_id": video_id,
            "checkpoint": ckpt_name,
        },
    )

    pred_table = wandb.Table(
        columns=["second", "state", "confidence", *STATE_NAMES.values()])
    for t, state_idx in enumerate(predicted):
        row = [t, STATE_NAMES[state_idx], float(probs[t, state_idx])]
        row.extend(float(probs[t, i]) for i in STATE_NAMES)
        pred_table.add_data(*row)

    summary_rows = []
    for idx, name in STATE_NAMES.items():
        count = int((predicted == idx).sum())
        if count:
            pct = count / n_frames * 100
            run.summary[f"seconds/{name}"] = count
            run.summary[f"pct/{name}"] = pct
            summary_rows.append([name, count, pct])

    summary_table = wandb.Table(data=summary_rows, columns=[
                                "state", "seconds", "pct"])
    log_payload = {
        "predictions": pred_table,
        "state_distribution": wandb.plot.bar(
            summary_table, "state", "seconds", title="State duration (seconds)"
        ),
    }
    try:
        timeline_fig, plt = _state_timeline_image(predicted, n_frames)
        log_payload["state_timeline"] = wandb.Image(
            timeline_fig, caption=f"{video_id} — {ckpt_name}"
        )
        wandb.log(log_payload)
        plt.close(timeline_fig)
    except ImportError:
        wandb.log(log_payload)

    url = run.url
    wandb.finish()
    return url


def run_inference(
    feat_path: str,
    ckpt_path: str,
    obj_feat_path: str | None = None,
    *,
    use_wandb: bool = False,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
    wandb_run_name: str | None = None,
):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # (n_frames, 768) or (n_frames, 1536)
    feat = torch.load(feat_path, map_location="cpu")
    input_dim, vocab_size = _checkpoint_config(ckpt_path)
    obj_feat_padded = obj_feat_path is None and feat.shape[-1] == 768 and input_dim == 1536
    feat = _prepare_features(feat, input_dim, obj_feat_path)
    print(f"Features: {feat.shape}  ({feat.shape[0]} seconds of video)")
    feat = feat.unsqueeze(0).to(device)              # (1, n_frames, input_dim)

    model = load_model(ckpt_path, input_dim=input_dim,
                       vocab_size=vocab_size).to(device)

    with torch.no_grad():
        pred = model(feat)                            # (n_frames, vocab_size)
        prob = torch.softmax(pred, dim=-1)

    predicted = prob.argmax(dim=-1).squeeze(0).cpu().numpy()   # (n_frames,)
    probs = prob.squeeze(0).cpu().numpy()

    n_frames = len(predicted)
    print(f"\nPer-second predictions  ({n_frames}s total)")
    print("-" * 55)
    for t, state_idx in enumerate(predicted):
        name = STATE_NAMES[state_idx]
        conf = probs[t, state_idx]
        bar = STATE_SYMBOLS[state_idx] * int(conf * 20)
        print(f"  t={t:3d}s  [{bar:<20}]  {name}  ({conf:.2f})")

    print("\nSummary:")
    for idx, name in STATE_NAMES.items():
        count = int((predicted == idx).sum())
        if count:
            pct = count / n_frames * 100
            print(f"  {name:<20} {count:3d}s  ({pct:.0f}%)")

    if use_wandb:
        url = _log_to_wandb(
            feat_path=feat_path,
            ckpt_path=ckpt_path,
            obj_feat_path=obj_feat_path,
            device=device,
            input_dim=input_dim,
            obj_feat_padded=obj_feat_padded,
            predicted=predicted,
            probs=probs,
            entity=wandb_entity,
            project=wandb_project,
            run_name=wandb_run_name,
        )
        print(
            f"\nW&B run: {url or '(offline — run wandb sync wandb/offline-run-* to upload)'}")

    return predicted, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", required=True,
                        help="Path to .pth.tar feature file")
    parser.add_argument("--ckpt", required=True,
                        help="Path to VidOSC .ckpt checkpoint")
    parser.add_argument(
        "--obj_feat",
        default=None,
        help="Optional object-centric feature file (.pth.tar) for det=1 checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=bool(os.getenv("WANDB_API_KEY")),
        help="Log results to Weights & Biases (default: on when WANDB_API_KEY is set)",
    )
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument(
        "--wandb-entity", default=WANDB_ENTITY, help="W&B entity")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        choices=WANDB_PROJECTS, help="W&B project")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Optional W&B run name")
    args = parser.parse_args()
    run_inference(
        args.feat,
        args.ckpt,
        args.obj_feat,
        use_wandb=args.wandb and not args.no_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
