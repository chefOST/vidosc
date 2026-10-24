"""E1: match cached Qwen state descriptions to the object's candidate states (MOSCATO GT used only to score).
Same items as experiments/matcha (runs/matcha/items.json), so rows are comparable to MATCHA (test top-1 .23)
and SigLIP text cosine (.392). Also reports bunching: mean pairwise cosine of an object's candidate states."""
import json
from collections import defaultdict

import numpy as np

from corpus import ROOT, bank, prompt

ITEMS = f"{ROOT}/runs/matcha/items.json"


def load():
    items = json.load(open(ITEMS))
    tb = bank()
    for i in items:
        i["d_emb"] = tb.prompts([prompt(i["obj"], i["desc"])])[0]
        i["s_emb"] = tb.prompts([prompt(i["obj"], s) for s in i["states"]])
    return items


def stats(scores, i):
    order = np.argsort(-np.asarray(scores))
    hit = [r for r, k in enumerate(order) if i["states"][k] in i["gt"]]
    return float(hit[0] == 0), 1 / (hit[0] + 1)


def evaluate(items, spaces, base="identity", n_boot=2000):
    """spaces: {name: Space}. Returns {split: {name: {top1, mrr, ci, bunch}}}."""
    out = {}
    for split in ("dev", "test"):
        S = [i for i in items if i["split"] == split]
        pv = defaultdict(lambda: defaultdict(list))
        bunch = {}
        for name, sp in spaces.items():
            D = sp.text(np.stack([i["d_emb"] for i in S]))
            bs = []
            for k, i in enumerate(S):
                B = sp.text(i["s_emb"])
                pv[i["vid"]][name].append(stats(B @ D[k], i))
                C = B @ B.T
                bs.append(C[~np.eye(len(B), dtype=bool)].mean())
            bunch[name] = float(np.mean(bs))
        vids = sorted(pv)
        rng = np.random.default_rng(0)
        boots = [rng.choice(vids, len(vids)) for _ in range(n_boot)]
        res = {}
        for name in spaces:
            a = np.array([s for v in vids for s in pv[v][name]])
            d = [np.mean([s[0] for v in b for s in pv[v][name]]) - np.mean([s[0] for v in b for s in pv[v][base]])
                 for b in boots]
            res[name] = {"top1": a[:, 0].mean(), "mrr": a[:, 1].mean(), "ci": (np.quantile(d, .025), np.quantile(d, .975)),
                         "bunch": bunch[name]}
        out[split] = res
    return out


def show(out):
    for split, res in out.items():
        print(split)
        for n, r in res.items():
            print(f"  {n:28s} top1 {r['top1']:.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  mrr {r['mrr']:.3f}  bunch {r['bunch']:.3f}")
