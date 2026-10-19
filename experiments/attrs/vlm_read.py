"""Stage 2b (method): a local VLM (Qwen3-VL-8B, MLX 4-bit) reads every attribute of every tracked object from keyframes.
Input: frames, recipe name, object names, the LLM schema. No MOSCATO information.
Usage (vidosc/.venv-vlm): python vlm_read.py --schema s0 --every 6  -> vlm_cache/<schema>/<vid>.jsonl"""
import argparse, json, os, re, string, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = 'mlx-community/Qwen3-VL-8B-Instruct-4bit'


def items(schema, recipe, objects):
    """Numbered (object, attribute, values) lines for one video."""
    out = []
    for o in objects:
        s = schema[recipe].get(o)
        if not s: continue
        for a in s['attributes']:
            vals = list(dict.fromkeys(a['values']))
            out.append((o, a['name'], vals))
    return out


def prompt(recipe, obj, its):
    """One object per prompt (multi-object prompts degenerate to 'not visible' for everything)."""
    lines = [f"{i + 2}. {a.replace('_', ' ')}: " + ' '.join(f"({string.ascii_lowercase[j]}) {v.replace('_', ' ')}" for j, v in enumerate(vals))
             for i, (o, a, vals) in enumerate(its)]
    return (f"Head-mounted camera frame of a person making {recipe.lower()}. Object: {obj}.\n"
            f"1. Is the {obj} visible in the frame? (a) yes (b) no\n" + '\n'.join(lines) +
            "\nAnswer each numbered question with its letter, e.g. 1a 2b 3a. No other text.")


def parse(txt, its):
    """-> value per item; None if the object is reported not visible; 'missing' if unparsed."""
    got = {int(n): c for n, c in re.findall(r'(\d+)\s*[\.:)]?\s*\(?([a-z])\)?', txt.lower())}
    if got.get(1) == 'b': return [None] * len(its)
    out = []
    for i, (_, _, vals) in enumerate(its):
        j = string.ascii_lowercase.find(got.get(i + 2, '?'))
        out.append(vals[j] if 0 <= j < len(vals) else 'missing')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--schema', default='s0'); ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--videos', nargs='*'); a = ap.parse_args()
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    gt = json.load(open(ROOT + '/../chefOST-data/MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json'))
    objects = {v: sorted(gt[v]['state']) for v in gt}; del gt  # pipeline input: object names only
    schema = json.load(open(f'{HERE}/schemas/schema_{a.schema}.json'))
    model, proc = load(MODEL)
    od = f'{HERE}/vlm_cache/{a.schema}'; os.makedirs(od, exist_ok=True)
    vids = a.videos or sorted(objects)
    for vi, vid in enumerate(vids):
        r = vid.split('_')[1]; its = items(schema, r, objects[vid])
        groups = {}
        for it in its: groups.setdefault(it[0], []).append(it)
        qs = {o: apply_chat_template(proc, model.config, prompt(r, o, g), num_images=1) for o, g in groups.items()}
        fd = f'{ROOT}/data/frames/{vid}_fps1'; frames = sorted(os.listdir(fd))
        path = f'{od}/{vid}.jsonl'
        done = {json.loads(l)['t'] for l in open(path)} if os.path.exists(path) else set()
        t0 = time.time(); n = 0
        with open(path, 'a') as f:
            for t in range(0, len(frames), a.every):
                if t in done: continue
                raw, ans = {}, []
                for o, g in groups.items():
                    txt = generate(model, proc, qs[o], [f'{fd}/{frames[t]}'], max_tokens=6 * len(g) + 10, temperature=0.0, verbose=False)
                    raw[o] = getattr(txt, 'text', txt); ans += parse(raw[o], g)
                items_ = [list(x[:2]) for g in groups.values() for x in g]
                f.write(json.dumps({'t': t, 'raw': raw, 'items': items_, 'ans': ans}) + '\n'); f.flush(); n += 1
        print(f'[{vi + 1}/{len(vids)}] {vid} items={len(its)} frames={n} {time.time() - t0:.0f}s', flush=True)
