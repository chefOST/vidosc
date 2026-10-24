"""Contradiction-axis spaces (zero-shot, per object; MATCHA's 'distance from a counterfactual' as a basis).

A blind LLM, given only the object name, writes the object's state contrasts as contradictory pairs, each side in
several phrasings. Axis k of object o is the mean text-embedding difference
    a_k = mean e(o, positive phrasing) - mean e(o, negative phrasing)
and the object's space maps any embedding (frame, crop, description, candidate state) to its coordinates
    z = A_o (x - mu_modality),   A_o = rows a_k / ||a_k||
so only variation along a named contrast survives. Ranking and discovery then run on unit(z).

    python experiments/statespace/axes.py      # -> data/cache/statespace/axes.json {object: [[pos phrases], [neg phrases]], ...}
"""
import json
import os
import sys

import numpy as np

import corpus
from corpus import ROOT, bank, prompt
from space import unit

sys.path.insert(0, os.path.join(ROOT, "experiments", "attrs"))
import llm  # noqa: E402

OUT = f"{ROOT}/data/cache/statespace/axes.json"
PROMPT = """Each line below is an object that appears in first-person cooking videos. For each object, list the
ways its VISIBLE state can change while someone cooks with it (for example how it is shaped or cut, mixed,
cooked, how full it is, open or closed, clean or dirty, on or off, covered, wet, where it is), as 4 to 8
contrasts. Each contrast is a pair of states that contradict each other (both cannot be true at once). Write
each side as 3 different short phrasings (at most 4 words each, without the object name).

Reply with JSON: {{"objects": {{"<line id>": [{{"axis": "<name>", "pos": ["..", "..", ".."], "neg": ["..", "..", ".."]}}, ...], ...}}}}
covering every line.

{lines}"""


def generate(objs, size=40, model="claude-opus-5-5"):
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rest = [o for o in objs if o not in done]
    for i in range(0, len(rest), size):
        b = rest[i:i + size]
        r = llm.call(PROMPT.format(lines="\n".join(f"[{k}] {o}" for k, o in enumerate(b))), model=model,
                     tag="stateaxes")["objects"]
        done.update({o: r[str(k)] for k, o in enumerate(b) if isinstance(r.get(str(k)), list)})
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(done, open(OUT, "w"), indent=1)
    return done


class AxisSpace:
    """Per-object contradiction-axis space. mode 'proj': coordinates only; 'mix': SigLIP (centred) plus
    lam * the axis coordinates mapped back, i.e. state directions up-weighted (x + lam * A^T A x)."""

    def __init__(self, mu_txt, mu_img=None, mode="proj", lam=1.0, templates=("a photo of {o}, {s}.",)):
        self.axes = json.load(open(OUT))
        self.mu_txt, self.mu_img, self.mode, self.lam, self.templates = mu_txt, mu_img, mode, lam, templates
        self.tb = bank()
        self.tb.prompts(sorted({t.format(o=o, s=x) for o, axs in self.axes.items() for ax in axs
                                for x in ax.get("pos", []) + ax.get("neg", []) for t in templates}))  # one batch
        self.cache = {}
        self.name = f"axes-{mode}" + (f"{lam:g}" if mode == "mix" else "")

    def A(self, obj):
        if obj not in self.cache:
            rows = []
            for ax in self.axes.get(obj, []):
                emb = lambda ss: unit(np.mean([self.tb.prompts([t.format(o=obj, s=s) for t in self.templates]).mean(0)
                                               for s in ss], 0))
                if ax.get("pos") and ax.get("neg"):
                    rows.append(emb(ax["pos"]) - emb(ax["neg"]))
            A = np.stack(rows) if rows else np.zeros((0, len(self.mu_txt)), np.float32)
            self.cache[obj] = (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)
        return self.cache[obj]

    def _map(self, X, obj):
        A = self.A(obj)
        if self.mode == "proj":
            return unit(X @ A.T)
        Q, _ = np.linalg.qr(A.T)  # orthonormal basis of the axes
        return unit(X + self.lam * (X @ Q) @ Q.T)

    def text(self, E, obj):
        return self._map(np.asarray(E, np.float32) - self.mu_txt, obj)

    def image(self, X, obj, mu_img=None):
        return self._map(np.asarray(X, np.float32) - (self.mu_img if mu_img is None else mu_img), obj)


if __name__ == "__main__":
    from pipeline import data
    objs = sorted({o for d in ("CMU", "EGTEA")
                   for ds in [data.Moscato(f"{ROOT}/../chefOST-data/MOSCATO", d, cache_dir=f"{ROOT}/data/cache")]
                   for v in ds.videos for o in ds.objects(v)})
    r = generate(objs)
    print(len(r), "objects;", np.mean([len(v) for v in r.values()]), "axes each")
    print(json.dumps(r.get(objs[0]), indent=0)[:800])
