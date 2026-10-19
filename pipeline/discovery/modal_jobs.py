"""GPU stages on Modal: object detection, crop embeddings, VLM descriptions.

Frames live on the Modal volume `chefost-vidosc` under /frames/{vid}_fps1/ (uploaded once with
`modal volume put chefost-vidosc data/frames /frames`). Results come back to the local caches.

    modal run -m pipeline.discovery.modal_jobs::detect_all
    modal run -m pipeline.discovery.modal_jobs::crops_all --arch ViT-SO400M-14-SigLIP-384/webli
VLM descriptions are requested by discovery/run.py (describe_items).
"""
import io
import os

import modal

from .. import ROOT

vol = modal.Volume.from_name("chefost-vidosc")
hf = modal.Volume.from_name("chefost-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "torchvision==0.21.0", "transformers==4.57.1", "accelerate", "open_clip_torch==3.3.0",
                      "timm", "pillow", "numpy", "scikit-learn", "qwen-vl-utils")
         .env({"HF_HOME": "/hf"})
         .add_local_python_source("pipeline"))
app = modal.App("chefost-vidosc", image=image)
MOUNTS = {"/vol": vol, "/hf": hf}


def frames(vid):
    d = f"/vol/frames/{vid}_fps1"
    return [f"{d}/{f}" for f in sorted(os.listdir(d)) if f.endswith(".jpg")]


@app.function(gpu="A10G", volumes=MOUNTS, timeout=3600)
def detect_video(vid, objs):
    import numpy as np
    from pipeline.discovery.objects import Detector
    b = Detector(device="cuda").top_boxes(frames(vid), objs)
    hf.commit()
    buf = io.BytesIO()
    np.savez(buf, objects=np.array(objs), boxes=b)
    return vid, buf.getvalue()


@app.function(gpu="A10G", volumes=MOUNTS, timeout=3600)
def crops_video(vid, boxes, arch):
    from pipeline import encoders
    from pipeline.discovery import objects
    enc = encoders.CLIPEncoder(arch, device="cuda")
    hf.commit()
    return vid, objects.encode_crops(enc, frames(vid), boxes, batch=128)


@app.function(gpu="A10G", volumes=MOUNTS, timeout=3600)
def describe_items(items):
    """items: [(cache key 'data/frames/<vid>_fps1/<n>.jpg|<object>')] -> [(key, description)]."""
    from pipeline.discovery.describe import Generator
    gen = Generator(device="cuda")
    hf.commit()
    todo = [("/vol/" + k.split("|")[0].removeprefix("data/"), k.split("|", 1)[1]) for k in items]
    return list(zip(items, gen(todo, batch=32)))


def describe(keys, chunk=400):
    """Fill the local description cache for `keys` (parallel GPU containers)."""
    from .describe import Descriptions
    d = Descriptions()
    todo = sorted({k for k in keys if k not in d.cache})
    if todo:
        print(f"[modal] describing {len(todo)} (frame, object) pairs")
        with modal.enable_output(), app.run():
            for res in describe_items.map([todo[i:i + chunk] for i in range(0, len(todo), chunk)]):
                d.cache.update(dict(res))
                d.save()
    return d.cache


def _ds():
    from pipeline import data
    return data.Moscato(f"{ROOT}/../chefOST-data/MOSCATO", cache_dir=f"{ROOT}/data/cache")


@app.local_entrypoint()
def detect_all():
    from pipeline.discovery import objects
    ds = _ds()
    todo = [(v, ds.objects(v)) for v in ds.videos if not os.path.exists(objects.boxes_path(v))]
    print(f"detecting {len(todo)} videos")
    for vid, blob in detect_video.starmap(todo):
        os.makedirs(os.path.dirname(objects.boxes_path(vid)), exist_ok=True)
        open(objects.boxes_path(vid), "wb").write(blob)
        print(vid, flush=True)


@app.local_entrypoint()
def crops_all(arch="ViT-SO400M-14-SigLIP-384/webli"):
    import numpy as np
    from pipeline.discovery import objects
    name = "clip-" + arch.replace("/", "-")
    ds = _ds()
    todo = [(v, objects.load_boxes(v)[1], arch) for v in ds.videos
            if os.path.exists(objects.boxes_path(v)) and not os.path.exists(objects.crops_path(name, v))]
    print(f"encoding crops for {len(todo)} videos")
    for vid, emb in crops_video.starmap(todo):
        os.makedirs(os.path.dirname(objects.crops_path(name, vid)), exist_ok=True)
        np.save(objects.crops_path(name, vid), emb)
        print(vid, flush=True)
