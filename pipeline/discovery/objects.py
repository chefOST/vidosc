"""Object-specific frame features: zero-shot detection of the tracked object, then crop embeddings.

Input is only (video frames, object name). OWLv2 is queried with the object name; per frame we keep the
top box and its score (a visibility signal). Crops (box plus a context margin) are embedded with an
open_clip image encoder, so outputs live in that encoder's text space.

Usage (caches boxes and crop features for every GT video):
    python -m pipeline.discovery.objects detect
    python -m pipeline.discovery.objects crops ViT-SO400M-14-SigLIP-384/webli
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from .. import ROOT, data, encoders

DET = "google/owlv2-base-patch16-ensemble"


def query(obj):
    """Detector phrase for a MOSCATO object name (the task input): 'the dish' -> 'a photo of a dish'."""
    o = obj.removeprefix("the ").replace("/", " or ")
    return f"a photo of {o}"


def frame_paths(vid):
    _, paths = data.sample_frames(data.cmu_video(vid, f"{ROOT}/data/cmu_videos"), 30, 1, f"{ROOT}/data/frames/{vid}_fps1")
    return paths


class Detector:
    def __init__(self, name=DET, device=None):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.device = device or encoders.default_device()
        self.proc = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def top_boxes(self, paths, objs):
        """[T, len(objs), 5]: per frame and object, the highest-scoring box (x0, y0, x1, y1 in pixels) and score."""
        q = [query(o) for o in objs]
        out = np.zeros((len(paths), len(objs), 5), np.float32)
        for t, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            inp = self.proc(text=[q], images=[im], return_tensors="pt").to(self.device)
            r = self.model(**inp)
            logits = r.logits[0].sigmoid().cpu()                     # [patches, queries]
            boxes = r.pred_boxes[0].cpu()                            # cx, cy, w, h on the padded square
            side = max(im.size)
            for j in range(len(objs)):
                k = int(logits[:, j].argmax())
                cx, cy, w, h = (boxes[k] * side).tolist()
                out[t, j] = [max(cx - w / 2, 0), max(cy - h / 2, 0), min(cx + w / 2, im.size[0]),
                             min(cy + h / 2, im.size[1]), float(logits[k, j])]
        return out


def boxes_path(vid):
    return f"{ROOT}/data/cache/det/owlv2/{vid}.npz"


def load_boxes(vid):
    z = np.load(boxes_path(vid), allow_pickle=False)
    return list(z["objects"]), z["boxes"]


def crop(im, box, margin=0.25, min_side=48):
    x0, y0, x1, y1 = box[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0, min_side) * (1 + margin)
    return im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))


def crops_path(enc_name, vid):
    return f"{ROOT}/data/cache/emb/crops-{enc_name}/{vid}_fps1.npy"


@torch.no_grad()
def encode_crops(enc, paths, boxes, batch=32):
    """[T, n_obj, D] unit crop embeddings."""
    T, n = boxes.shape[:2]
    out = []
    items = [(t, j) for t in range(T) for j in range(n)]
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ims = {t: Image.open(paths[t]).convert("RGB") for t in {t for t, _ in chunk}}
        x = torch.stack([enc.preprocess(crop(ims[t], boxes[t, j])) for t, j in chunk]).to(enc.device)
        out.append(torch.nn.functional.normalize(enc.model.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(out).numpy().reshape(T, n, -1).astype(np.float16)


def main():
    ds = data.Moscato(f"{ROOT}/../chefOST-data/MOSCATO", cache_dir=f"{ROOT}/data/cache")
    if sys.argv[1] == "detect":
        det = Detector()
        for vid in ds.videos:
            if os.path.exists(boxes_path(vid)):
                continue
            objs = ds.objects(vid)
            b = det.top_boxes(frame_paths(vid), objs)
            os.makedirs(os.path.dirname(boxes_path(vid)), exist_ok=True)
            np.savez(boxes_path(vid), objects=np.array(objs), boxes=b)
            print(vid, b.shape, f"mean top score {b[..., 4].mean():.3f}", flush=True)
    elif sys.argv[1] == "crops":
        enc = encoders.CLIPEncoder(sys.argv[2])
        for vid in ds.videos:
            p = crops_path(enc.name, vid)
            if os.path.exists(p) or not os.path.exists(boxes_path(vid)):
                continue
            _, b = load_boxes(vid)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.save(p, encode_crops(enc, frame_paths(vid), b))
            print(vid, flush=True)


if __name__ == "__main__":
    main()
