"""OWLv2 object crops of EPIC frames around each action (noun as the detector prompt), embedded with SigLIP.
Gives same-object before/after crop pairs for training (corpus.crop_pairs). Separate Modal app so it can run
next to chefost-vidosc jobs.

    modal run experiments/statespace/modal_crops.py     # -> data/cache/statespace/epic_crops/<vid>.npz
"""
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from pipeline.discovery import modal_jobs  # noqa: E402

app = modal.App("chefost-statespace", image=modal_jobs.image)
OUT = os.path.join(HERE, "..", "..", "data", "cache", "statespace", "epic_crops")
ARCH = "ViT-SO400M-14-SigLIP-384/webli"


@app.function(gpu="A10G", volumes=modal_jobs.MOUNTS, timeout=3600, max_containers=6)
def crop_video(vid, items):
    """items: [(t, noun)] (t = 1 fps frame index) -> npz bytes: t, noun, boxes [N, 5], emb [N, D]."""
    import io
    import numpy as np
    import torch
    from PIL import Image
    from pipeline import encoders
    from pipeline.discovery import objects
    paths = modal_jobs.frames(vid)
    items = [(t, n) for t, n in items if 0 <= t < len(paths)]
    det = objects.Detector(device="cuda")
    enc = encoders.CLIPEncoder(ARCH, device="cuda")
    # no hf.commit(): models are cached, and concurrent commits from many containers failed
    boxes = np.zeros((len(items), 5), np.float32)
    by_t = {}
    for k, (t, n) in enumerate(items):
        by_t.setdefault(t, []).append((k, n))
    for t, ks in by_t.items():
        nouns = sorted({n for _, n in ks})
        b = det.top_boxes([paths[t]], nouns)[0]
        for k, n in ks:
            boxes[k] = b[nouns.index(n)]
    embs = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            x = torch.stack([enc.preprocess(objects.crop(Image.open(paths[t]).convert("RGB"), boxes[k]))
                             for k, (t, _) in enumerate(items[i:i + 64], start=i)]).to("cuda")
            embs.append(torch.nn.functional.normalize(enc.model.encode_image(x).float(), dim=-1).cpu().numpy())
    buf = io.BytesIO()
    np.savez(buf, t=np.array([t for t, _ in items]), noun=np.array([n for _, n in items]), boxes=boxes,
             emb=np.concatenate(embs).astype(np.float16) if embs else np.zeros((0, 1152), np.float16))
    return vid, buf.getvalue()


@app.local_entrypoint()
def main():
    sys.path.insert(0, HERE)
    import corpus
    os.makedirs(OUT, exist_ok=True)
    want = {}
    for p, v, t0, t1, verb, n, b, a in corpus.frames():
        for t in (t0 - 1, t0, t1, t1 + 1):
            want.setdefault(v, set()).add((t, n))
    todo = [(v, sorted(s)) for v, s in sorted(want.items()) if not os.path.exists(f"{OUT}/{v}.npz")]
    print(f"cropping {sum(len(s) for _, s in todo)} (frame, noun) items in {len(todo)} videos")
    for res in crop_video.starmap(todo, return_exceptions=True):
        if isinstance(res, Exception):
            print("FAILED", repr(res)[:500], flush=True)
            continue
        vid, blob = res
        open(f"{OUT}/{vid}.npz", "wb").write(blob)
        print(vid, flush=True)
