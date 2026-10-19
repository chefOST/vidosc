"""Free-text object-state descriptions from a VLM (Qwen2.5-VL-3B-Instruct), cached per (frame, object).

The prompt gives only the object name (the task input) and generic state aspects, no MOSCATO vocabulary.
Generation runs on Modal (pipeline/discovery/modal_jobs.py); `Descriptions` is the local cache.
"""
import json
import os

from .. import ROOT

VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = ("This is a frame from a first-person cooking video. Describe the current state of the {object} in a short "
          "phrase of at most 8 words: for example its shape or form, whether it is cut, mixed, cooked, full or empty, "
          "open or closed, clean or dirty, and where it is. If the {object} is not visible, describe its most likely "
          "state. Answer with the phrase only.")


def key(vid, t, obj):
    """Cache key of frame t (0-based, 1 fps) of `vid` and object `obj`."""
    return f"data/frames/{vid}_fps1/{t + 1:06d}.jpg|{obj}"


class Generator:
    """Batched greedy generation; runs wherever the weights and frames are."""

    def __init__(self, name=VLM, device=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(name, min_pixels=128 * 28 * 28, max_pixels=320 * 28 * 28)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(self.device).eval()

    def __call__(self, items, batch=32, max_new_tokens=24):
        """items: [(image path, object)] -> [description]."""
        from PIL import Image
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            texts = [self.proc.apply_chat_template([{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT.format(object=o)}]}], add_generation_prompt=True)
                for _, o in chunk]
            ims = [Image.open(p).convert("RGB") for p, _ in chunk]
            inp = self.proc(text=texts, images=ims, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                g = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
            ans = self.proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [a.strip().strip(".").strip() for a in ans]
        return out


class Descriptions:
    """Local cache {key: description}."""

    def __init__(self, name=VLM):
        self.path = f"{ROOT}/data/cache/describe/{name.split('/')[-1]}.json"
        self.cache = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.cache, open(self.path, "w"), indent=0)
