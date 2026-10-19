"""Eval-map reliability: agreement between two independent LLM mappers over (object, word, attribute) cells."""
import json, sys
import numpy as np
from evalmap import gt_words
S = json.load(open('schemas/schema_s0.json'))
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
W = gt_words(); ya, yb, same = [], [], []
for r in S:
    for o, s in S[r].items():
        for w in W[(r, o)]:
            for at in s['attributes']:
                x = set(A[r].get(o, {}).get(w, {}).get(at['name'], [])); y = set(B[r].get(o, {}).get(w, {}).get(at['name'], []))
                ya.append(bool(x)); yb.append(bool(y))
                if x and y: same.append(x == y)
ya, yb = np.array(ya), np.array(yb); po = (ya == yb).mean(); pe = ya.mean() * yb.mean() + (1 - ya.mean()) * (1 - yb.mean())
print(f'cells={len(ya)} assert rate A={ya.mean():.3f} B={yb.mean():.3f} agreement={po:.3f} kappa={(po - pe) / (1 - pe):.3f} '
      f'exact value-set match when both assert={np.mean(same):.3f} (n={len(same)})')
