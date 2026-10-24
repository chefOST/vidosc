"""Training data for state-sensitive spaces. Nothing here comes from MOSCATO.

text():   EPIC-KITCHENS (verb, noun) before/after state phrases (pipeline/discovery/epic_states.py, blind LLM).
          The before and after phrase of one action are a natural contradiction pair (same object, other state).
frames(): EPIC 1 fps SigLIP frame embeddings at each action's start and end, with that action's before/after
          phrases: an image-grounded contradiction pair (same object, same scene, seconds apart).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from pipeline import ROOT  # noqa: E402
from pipeline.discovery import epic, run  # noqa: E402

FDIR = "clip-ViT-SO400M-14-SigLIP-384-webli"
SPEC = "clip:ViT-SO400M-14-SigLIP-384/webli"


_BANK = None


def bank():
    """One shared SigLIP TextBank per process (its cache file is large; load it once)."""
    global _BANK
    if _BANK is None:
        _BANK = run.TextBank(SPEC)
    return _BANK


def state_of(noun, phrase):
    """'garlic chopped on chopping board' -> 'chopped on chopping board' (EPIC phrases start with the noun)."""
    p = phrase.strip().rstrip(".")
    return p[len(noun):].lstrip(" ,") if p.lower().startswith(noun.lower()) else p


def prompt(obj, state):
    """The pipeline's basic template (pipeline/scoring.py)."""
    return f"a photo of {obj}, {state}."


def text():
    """[(verb, noun, before state, after state)] with the noun stripped from the phrases."""
    d = json.load(open(f"{epic.DIR}/state_phrases.json"))
    out = []
    for k, (b, a) in sorted(d.items()):
        v, n = k.split("|")
        out.append((v, n, state_of(n, b), state_of(n, a)))
    return out


def frames():
    """[(participant, vid, t_start, t_end, verb, noun, before, after)] for EPIC actions whose video has cached
    SigLIP frame embeddings, with t in seconds (1 fps index)."""
    d = json.load(open(f"{epic.DIR}/state_phrases.json"))
    have = {f[:-9] for f in os.listdir(f"{ROOT}/data/cache/emb/{FDIR}") if f.startswith("P") and "_" in f[:4]}
    sec = lambda ts: int(int(ts.split(":")[0]) * 3600 + int(ts.split(":")[1]) * 60 + float(ts.split(":")[2]))
    out = []
    for r in csv.DictReader(open(f"{epic.DIR}/EPIC_100_train.csv")):
        v = r["video_id"]
        if v not in have:
            continue
        n = epic.noun(r["noun"])
        ph = d.get(f"{r['verb']}|{n}")
        if not ph:
            continue
        out.append((v.split("_")[0], v, sec(r["start_timestamp"]), sec(r["stop_timestamp"]), r["verb"], n,
                    state_of(n, ph[0]), state_of(n, ph[1])))
    return out


def frame_emb(vid):
    return np.load(f"{ROOT}/data/cache/emb/{FDIR}/{vid}_fps1.npy").astype(np.float32)
