"""Load official HowToChange or ChangeIt models without training dependencies."""
from argparse import Namespace
import importlib.util
from pathlib import Path
import re

import torch


def read_weights(path):
    # Do not silently fall back to arbitrary pickle execution for downloaded files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw = checkpoint.get("state_dict", checkpoint)
    weights = {k.removeprefix("model."): v for k, v in raw.items()
               if isinstance(v, torch.Tensor) and (k.startswith("model.") or "state_dict" not in checkpoint)}
    if "proj.weight" not in weights:
        raise ValueError("Checkpoint has no recognizable VidOSC projection weights")
    return weights


def inspect_weights(weights):
    dual = "state_head.weight" in weights
    head = "state_head.weight" if dual else "head.weight"
    if head not in weights:
        raise ValueError("Checkpoint has neither a ChangeIt nor a HowToChange state head")
    layers = {int(m.group(1)) for key in weights
              if (m := re.match(r"transformer_encoder.layers.(\d+)\.", key))}
    if not layers or layers != set(range(len(layers))):
        raise ValueError("Invalid transformer layer layout")
    return {"family": "changeit" if dual else "howtochange",
            "input_dim": int(weights["proj.weight"].shape[1]),
            "transformer_dim": int(weights["proj.weight"].shape[0]),
            "transformer_layers": len(layers),
            "state_classes": int(weights[head].shape[0]),
            "action_classes": int(weights["action_head.weight"].shape[0]) if dual else 0,
            "max_timesteps": int(weights["pos_encoder.pe"].shape[1])}


def predict(checkpoint, features, upstream, heads=4, pad_to=0):
    weights = read_weights(checkpoint)
    info = inspect_weights(weights)
    path = Path(upstream) / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official VidOSC model.py not found at {path}")
    spec = importlib.util.spec_from_file_location("_official_vidosc_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(**{k: info[k] for k in (
        "input_dim", "transformer_dim", "transformer_layers", "state_classes", "action_classes")},
        transformer_heads=heads, transformer_dropout=0.1, vocab_size=info["state_classes"])
    cls = module.FeatTimeTransformerDualHead if info["family"] == "changeit" else module.FeatTimeTransformer
    model = cls(args).eval()
    model.load_state_dict(weights, strict=True)
    feat = torch.load(features, map_location="cpu", weights_only=True)
    if not isinstance(feat, torch.Tensor) or feat.ndim != 2 or not len(feat):
        raise ValueError("Features must be a nonempty [time, input_dim] tensor")
    if feat.shape[1] != info["input_dim"]:
        raise ValueError(f"Expected {info['input_dim']} feature dimensions; got {feat.shape[1]}. No zero padding is performed.")
    if len(feat) > info["max_timesteps"] or not torch.isfinite(feat).all():
        raise ValueError("Features exceed positional capacity or contain non-finite values")
    length = len(feat)
    if pad_to:
        if pad_to < length or pad_to > info['max_timesteps']:
            raise ValueError('Invalid temporal padding length')
        feat = torch.cat((feat, feat.new_zeros((pad_to-length, feat.shape[1]))))
    info['input_timesteps'] = length
    info['transformer_timesteps'] = len(feat)
    with torch.inference_mode():
        result = model(feat.float().unsqueeze(0))
        if isinstance(result, dict):
            state = torch.softmax(result["state"], -1).numpy()[:length]
            action = torch.softmax(result["action"], -1).numpy()[:length]
        else:
            state = torch.softmax(result, -1).numpy()[:length]
            action = None
    return state, action, info
