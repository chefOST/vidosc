"""EVALUATION ONLY: map MOSCATO GT words to values of the (blindly generated) attributes, via an LLM that sees both.
Never used by the method. Usage: python evalmap.py <schema_s{k}.json> [model]  -> schemas/evalmap_<schema>_<model>.json"""
import json, os, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from common import HERE, ds, recipe
import llm

PROMPT = """EVALUATION MAPPING. Human annotators labeled the state of objects in egocentric cooking videos with short words. \
Separately, a tracking system describes each object with attributes and values (below), defined without seeing the annotations. \
For each object and each annotation word, give the attribute values that the word asserts about that object in this recipe.
Rules:
- Assert a value only if the word clearly implies it for this object; otherwise leave that attribute out.
- A word may assert values of several attributes, or nothing (empty object {{}}).
- If the word is compatible with several values of an attribute but rules out the others, list all compatible values.
- Use attribute and value names exactly as given.

Recipe: {recipe}
Objects (attributes -> values, and the annotation words used for that object):
{objects}

Return JSON only: {{"<object>": {{"<word>": {{"<attribute>": ["<value>", ...]}}, ...}}, ...}}"""


def gt_words():
    w = defaultdict(set)
    for v in ds.videos:
        for o in ds.objects(v):
            for f in ds.gt[v]['state'][o][::30]:
                w[(recipe(v), o)].update(f)
    return w


def build(schema, model):
    W = gt_words()
    def one(r):
        objs = {o: {'attributes': {a['name']: list(dict.fromkeys(a['values'])) for a in s['attributes']},
                    'words': sorted(W[(r, o)])} for o, s in schema[r].items() if W[(r, o)]}
        out = llm.call(PROMPT.format(recipe=r, objects=json.dumps(objs, indent=1)), model=model, tag=f'map_{r}')
        # keep only valid attribute/value names
        clean = {}
        for o, ws in out.items():
            if o not in objs: continue
            A = objs[o]['attributes']
            clean[o] = {w: {a: [x for x in vs if x in A[a]] for a, vs in m.items() if a in A and any(x in A[a] for x in vs)}
                        for w, m in ws.items() if w in objs[o]['words']}
        return clean
    with ThreadPoolExecutor(5) as ex:
        return dict(zip(schema, ex.map(one, schema)))


if __name__ == '__main__':
    sp = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else 'claude-opus-5-5'
    schema = json.load(open(sp))
    m = build(schema, model)
    out = os.path.join(HERE, 'schemas', f"evalmap_{os.path.basename(sp)[:-5]}_{model}.json")
    json.dump(m, open(out, 'w'), indent=1)
    n = sum(len(ws) for r in m.values() for ws in r.values())
    na = sum(bool(x) for r in m.values() for ws in r.values() for x in ws.values())
    print(out, 'words', n, 'asserting', na)
