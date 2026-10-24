"""Intrinsic state-space test: does a text space match same-state phrases and separate contradicting ones,
independent of any VLM description?

Item = (object, anchor state, 3 paraphrases, 4 counterfactual contradictions), all blind-LLM written
(gen_triplets.PROMPT). Paraphrases change the wording, contradictions keep it, so word overlap points the wrong
way. Sets:
  epic     data/epic/state_triplets.json (5k EPIC states)
  moscato  same prompt on the MOSCATO candidate-state names (36 objects, 357 states; names only, eval only)

Per item: pair accuracy = P[s(a, para) > s(a, contra)] over 3x4 pairs, strict = min para > max contra.
Bootstrap over objects, paired vs SigLIP centred. Scores come from modal_textmatch.sep_scores.

    python experiments/statespace/eval_sep.py gen [moscato|epic-big|moscato-big]   # triplets (claude -p)
    modal run experiments/statespace/modal_textmatch.py::sep
    python experiments/statespace/eval_sep.py [set ...]
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import corpus
import gen_triplets

SETS = {"epic": gen_triplets.OUT, "moscato": f"{corpus.ROOT}/data/cache/statespace/moscato_vocab_triplets.json",
        "epic-big": f"{corpus.ROOT}/data/cache/statespace/epic_big_triplets.json",
        "moscato-big": f"{corpus.ROOT}/data/cache/statespace/moscato_big_triplets.json"}
# -big sets: 8 paraphrases, 10 contradictions (at most 2 per aspect); epic-big is 1000 random EPIC states
BIG = (gen_triplets.PROMPT.replace('"same": 3 paraphrases', '"same": 8 paraphrases')
       .replace('"contra": 4 counterfactuals', '"contra": 10 counterfactuals')
       .replace("Use a different aspect for each where possible", "Use at most 2 per aspect, covering as many aspects as fit"))
SCORES = f"{corpus.ROOT}/data/cache/statespace/sep_scores.json"


def gen(name="moscato", size=100, model="claude-opus-5-5"):
    prompt = BIG if name.endswith("-big") else gen_triplets.PROMPT
    if name.startswith("moscato"):
        items = json.load(open(f"{corpus.ROOT}/runs/matcha/items.json"))
        pairs = sorted({(i["obj"], s) for i in items for s in i["states"]})
    else:
        import random
        pairs = sorted(random.Random(0).sample(gen_triplets.todo(), 1000))
    batches = [pairs[k:k + size] for k in range(0, len(pairs), size)]

    def call(b):
        lines = "\n".join(f"[{k}] {o}: {s}" for k, (o, s) in enumerate(b))
        try:
            r = gen_triplets.llm.call(prompt.format(lines=lines), model=model, tag="statesep")["items"]
        except Exception as e:
            print("failed", e, flush=True)
            return {}
        return {f"{o}|{s}": r[str(k)] for k, (o, s) in enumerate(b)
                if isinstance(r.get(str(k)), dict) and r[str(k)].get("same") and r[str(k)].get("contra")}

    out = {}
    with ThreadPoolExecutor(4) as ex:
        for res in ex.map(call, batches):
            out.update(res)
    json.dump(out, open(SETS[name], "w"), indent=0)
    print(len(out), "of", len(pairs))


ASPECTS = ["shape or form", "mixed or separate", "cooked or raw or heated", "full or empty", "open or closed", "on or off",
           "clean or dirty", "wet or dry", "covered or uncovered", "location"]


def aspect_of(label):
    """LLM labels drift ('full', 'clean/dirty'); map to the prompt's aspect by first word."""
    w = label.lower().replace("/", " ").split()[0] if label.strip() else ""
    return next((a for a in ASPECTS if w in a.split()), "other")


def load(name):
    """[(obj, anchor, [para], [(aspect, contra)])]"""
    d = json.load(open(SETS[name]))
    return [(k.split("|")[0], k.split("|", 1)[1], v["same"], [tuple(c) for c in v["contra"]]) for k, v in d.items()]


def pairs(items):
    """Unique (obj, phrase a, phrase b) for every anchor-para and anchor-contra pair."""
    return sorted({(o, a, p) for o, a, ps, cs in items for p in ps + [c for _, c in cs]})


def jaccard(a, b):
    x, y = set(a.lower().replace(",", " ").split()), set(b.lower().replace(",", " ").split())
    return len(x & y) / max(len(x | y), 1)


def evaluate(name, scores, base="siglip-centred", n_boot=2000):
    items = load(name)
    idx = {tuple(p): k for k, p in enumerate(scores["pairs"][name])}
    meths = dict(scores["scores"][name])
    meths["word-overlap"] = [jaccard(a, b) for _, a, b in scores["pairs"][name]]
    per = defaultdict(lambda: defaultdict(list))  # obj -> meth -> [(acc, strict)]
    aspect = defaultdict(lambda: defaultdict(list))
    for o, a, ps, cs in items:
        for m, s in meths.items():
            sp = np.array([s[idx[(o, a, p)]] for p in ps])
            sc = np.array([s[idx[(o, a, c)]] for _, c in cs])
            per[o][m].append(((sp[:, None] > sc[None]).mean(), float(sp.min() > sc.max())))
            for (asp, _), col in zip(cs, (sp[:, None] > sc[None]).T):
                aspect[m][aspect_of(asp)].append(col.mean())
    objs = sorted(per)
    rng = np.random.default_rng(0)
    boots = [rng.choice(objs, len(objs)) for _ in range(n_boot)]
    acc = lambda m, os_: np.mean([r[0] for o in os_ for r in per[o][m]])
    print(f"{name}: {len(items)} states, {len(objs)} objects")
    for m in sorted(meths, key=lambda m: -acc(m, objs)):
        strict = np.mean([r[1] for o in objs for r in per[o][m]])
        d = [acc(m, b) - acc(base, b) for b in boots]
        print(f"  {m:22s} pair acc {acc(m, objs):.3f}  strict {strict:.3f}  "
              f"vs {base} {acc(m, objs) - acc(base, objs):+.3f} [{np.quantile(d, .025):+.3f},{np.quantile(d, .975):+.3f}]")
    asps = sorted({a for m in aspect for a in aspect[m]}, key=lambda a: -len(aspect[base][a]))
    print("  per aspect (pair acc):", "  ".join(f"{a}" for a in asps))
    for m in sorted(meths, key=lambda m: -acc(m, objs)):
        print(f"  {m:22s}", "  ".join(f"{np.mean(aspect[m][a]):.2f}" for a in asps))


if __name__ == "__main__":
    if sys.argv[1:2] == ["gen"]:
        gen(sys.argv[2] if len(sys.argv) > 2 else "moscato", size=25 if sys.argv[2:3] and "big" in sys.argv[2] else 100)
    else:
        sc = json.load(open(SCORES))
        for n in sys.argv[1:] or SETS:
            evaluate(n, sc)
