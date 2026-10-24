"""Scores modal_textmatch.py output like eval_desc.py: top-1 / MRR per split, paired video bootstrap vs centred
SigLIP text cosine.

    python experiments/statespace/score_textmatch.py
"""
import json
from collections import defaultdict

import numpy as np

from eval_desc import load, stats
from modal_textmatch import OUT
from space import Space, text_mean


def main(base="siglip-centred", n_boot=2000):
    r = json.load(open(OUT))
    idx = {tuple(p): k for k, p in enumerate(r["pairs"])}
    items = load()
    sp = Space(text_mean())
    for i in items:
        i["scores"] = {"siglip-centred": sp.text(i["s_emb"]) @ sp.text(i["d_emb"][None])[0]}
        for name, s in r["scores"].items():
            i["scores"][name] = np.array([s[idx[(i["obj"], i["desc"], st)]] for st in i["states"]])
    names = list(items[0]["scores"])
    for split in ("dev", "test"):
        pv = defaultdict(lambda: defaultdict(list))
        for i in items:
            if i["split"] == split:
                for n in names:
                    pv[i["vid"]][n].append(stats(i["scores"][n], i))
        vids = sorted(pv)
        rng = np.random.default_rng(0)
        boots = [rng.choice(vids, len(vids)) for _ in range(n_boot)]
        top1 = lambda n, vs: np.mean([s[0] for v in vs for s in pv[v][n]])
        print(f"{split} ({sum(len(pv[v][base]) for v in vids)} items, {len(vids)} videos)")
        for n in sorted(names, key=lambda n: -top1(n, vids)):
            a = np.array([s for v in vids for s in pv[v][n]])
            d = [top1(n, b) - top1(base, b) for b in boots]
            print(f"  {n:22s} top1 {a[:, 0].mean():.3f}  mrr {a[:, 1].mean():.3f}  "
                  f"vs {base} {a[:, 0].mean() - top1(base, vids):+.3f} [{np.quantile(d, .025):+.3f},{np.quantile(d, .975):+.3f}]")


if __name__ == "__main__":
    main()
