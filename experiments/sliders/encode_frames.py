"""Frame-level image features (1 fps, raw frames 0,30,60.. = InternVideo fps1 centres) for the GT videos.
Usage: python experiments/sliders/encode_frames.py ViT-L-14/openai"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline import data, encoders
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
arch = sys.argv[1]
ds = data.Moscato(ROOT + '/../chefOST-data/MOSCATO', cache_dir=ROOT + '/data/cache')
enc = encoders.CLIPEncoder(arch)
out = f'{ROOT}/data/cache/emb/{enc.name}'; os.makedirs(out, exist_ok=True)
for vid in ds.videos:
    p = f'{out}/{vid}_fps1.npy'
    if os.path.exists(p): continue
    idx, paths = data.sample_frames(data.cmu_video(vid, ROOT + '/data/cmu_videos'), 30, 1, f'{ROOT}/data/frames/{vid}_fps1')
    np.save(p, enc.encode_images(paths).astype(np.float16)); print(vid, len(paths), flush=True)
