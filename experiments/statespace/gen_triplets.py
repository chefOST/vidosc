"""MATCHA-style triplets for EPIC state phrases, written by a blind LLM (no frames, no MOSCATO vocabulary).

For each (object, state): paraphrases that describe the same visible state in other words, and counterfactuals
that keep the object and most of the wording but change the state to one that cannot hold at the same time
(MATCHA's 'adversarially generated counterfactual contradiction'), each along a named axis.

    python experiments/statespace/gen_triplets.py     # -> data/epic/state_triplets.json {"noun|state": {...}}
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import corpus

sys.path.insert(0, os.path.join(corpus.ROOT, "experiments", "attrs"))
import llm  # noqa: E402

OUT = f"{corpus.epic.DIR}/state_triplets.json"
PROMPT = """Each line below is an object from a first-person kitchen video and a short phrase describing its
visible state. For each line write:
- "same": 3 paraphrases describing the SAME visible state in different words (synonyms, other phrasing, more or
  less detail), each at most 8 words, without the object name.
- "contra": 4 counterfactuals that keep the object and as much of the wording as possible but change the state
  so that it CONTRADICTS the original (both cannot be true at once), each at most 8 words, without the object
  name. Use a different aspect for each where possible, chosen from: shape or form (whole, cut, sliced, grated,
  mashed), mixed or separate, cooked or raw or heated, full or empty, open or closed, on or off, clean or dirty,
  wet or dry, covered or uncovered, location.
  Label each with its aspect.

Reply with JSON: {{"items": {{"<line id>": {{"same": ["..."], "contra": [["<aspect>", "..."], ...]}}, ...}}}}
covering every line.

{lines}"""


def todo(n=5000, per_noun=6, seed=0):
    """Up to n distinct (noun, state), at most per_noun per noun."""
    pairs = sorted({(nn, s) for _, nn, b, a in corpus.text() for s in (b, a) if s})
    rng = random.Random(seed)
    rng.shuffle(pairs)
    cnt, out = {}, []
    for nn, s in pairs:
        if cnt.get(nn, 0) < per_noun:
            cnt[nn] = cnt.get(nn, 0) + 1
            out.append((nn, s))
    return sorted(out[:n])


def main(size=100, model="claude-opus-5-5", workers=4):
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rest = [p for p in todo() if f"{p[0]}|{p[1]}" not in done]
    batches = [rest[i:i + size] for i in range(0, len(rest), size)]
    print(f"{len(rest)} phrases in {len(batches)} batches", flush=True)

    def call(b):
        lines = "\n".join(f"[{k}] {nn}: {s}" for k, (nn, s) in enumerate(b))
        try:
            r = llm.call(PROMPT.format(lines=lines), model=model, tag="statetriplets")["items"]
        except Exception as e:
            print("failed", e, flush=True)
            return {}
        return {f"{nn}|{s}": r[str(k)] for k, (nn, s) in enumerate(b)
                if isinstance(r.get(str(k)), dict) and r[str(k)].get("same") and r[str(k)].get("contra")}

    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(call, batches):
            done.update(res)
            json.dump(done, open(OUT, "w"), indent=0)
            print(len(done), flush=True)


if __name__ == "__main__":
    main()
