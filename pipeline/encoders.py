"""Frame/text encoders. Add new ones to ENCODERS; spec format '<kind>:<args>'."""
import os

import numpy as np
import torch
from PIL import Image


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class CLIPEncoder:
    """open_clip model, e.g. 'clip:ViT-L-14/openai' (the CLIP family VidOSC used for ChangeIt pseudo-labels)."""

    def __init__(self, arch="ViT-L-14/openai", device=None):
        import open_clip
        model, pretrained = arch.split("/")
        self.name = f"clip-{model}-{pretrained}"
        self.device = device or default_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    @torch.no_grad()
    def encode_images(self, paths, batch=32):
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            e = self.model.encode_image(x.to(self.device)).float()
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def encode_texts(self, texts):
        e = self.model.encode_text(self.tokenizer(texts).to(self.device)).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def _internvideo(*args):
    from .internvideo import InternVideoEncoder
    return InternVideoEncoder(*args)


ENCODERS = {"clip": CLIPEncoder, "internvideo": _internvideo}


def build_encoder(spec):
    kind, _, arg = spec.partition(":")
    return ENCODERS[kind](arg) if arg else ENCODERS[kind]()


def cached_image_embeddings(encoder, paths, cache_path):
    """Returns (embeddings, cache_hit)."""
    return _cached(lambda: encoder.encode_images(paths), len(paths), cache_path)


def cached_video_embeddings(encoder, video, centers, src_fps, cache_path):
    """Video-level encoders: one clip per centre frame. Raw (unnormalized) features. Returns (emb, cache_hit)."""
    return _cached(lambda: encoder.encode_video(video, centers, src_fps), len(centers), cache_path)


def _cached(compute, n, cache_path):
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if len(emb) == n:
            return emb, True
    emb = compute()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, emb)
    return emb, False
