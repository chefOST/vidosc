'Generate attribute state predictions from saved visual evidence. Scoring lives in chefOST-data.'
import argparse, json, os, time
from collections import defaultdict
import numpy as np
from common import HERE, ROOT, ds, recipe
import method as M


def load_tracks(schema, enc, vlm=None, vlm_every=1):
    emb = M.text_emb(M.schema_texts(schema), enc)
    tracks = []
    for vid in ds.videos:
        F = M.feats(vid, enc); T = len(F); idx = list(range(0, 30 * T, 30)); r = recipe(vid)
        VL = M.load_vlm(f'{HERE}/vlm_cache/{vlm}/{vid}.jsonl') if vlm else {}
        if vlm and not VL: continue
        if vlm_every > 1: VL = {k: [(t, v) for t, v in x if t % vlm_every == 0] for k, x in VL.items()}
        for o in ds.objects(vid):
            sch = schema[r].get(o)
            if not sch: continue
            for a in map(M.norm_attr, sch['attributes']):
                V = a['values']
                S, E, pres = M.evidence(F, a, sch['visible'], emb)
                tracks.append(dict(vid=vid, recipe=r, o=o, a=a['name'], V=V, one_way=bool(a.get('one_way')),
                                   script=[V.index(v) for v in a['script']], T=T, S=S, E=E, pres=pres,
                                   vlm=VL.get((o, a['name']), [])))
    return tracks




def methods(rng, tracks, vlm=None):
    pool = defaultdict(list)
    for t in tracks: pool[(t['recipe'], t['o'], t['a'])].append(t)
    def other(t):  # control: calibrated evidence from another video of the same recipe/object/attribute
        c = [u for u in pool[(t['recipe'], t['o'], t['a'])] if u['vid'] != t['vid']]
        if not c: return t['E'][rng.permutation(t['T'])]
        E = c[rng.integers(len(c))]['E']; x = np.linspace(0, 1, len(E))
        return np.stack([np.interp(np.linspace(0, 1, t['T']), x, E[:, j]) for j in range(E.shape[1])], 1)
    def other_vlm(t):  # control: VLM answers from another video of the same recipe/object/attribute, time-rescaled
        c = [u for u in pool[(t['recipe'], t['o'], t['a'])] if u['vid'] != t['vid']]
        if not c: return M.vlm_E([], t['V'], t['T'])
        u = c[rng.integers(len(c))]
        return M.vlm_E([(int(k * t['T'] / u['T']), v) for k, v in u['vlm']], t['V'], t['T'])
    return {
        'prior: script, even split (no video)': lambda t: M.prior_split(t['T'], t['script']),
        'prior: first script value (no video)': lambda t: np.full(t['T'], t['script'][0]),
        'prior: last script value (no video)': lambda t: np.full(t['T'], t['script'][-1]),
        'chance: uniform random value': lambda t: rng.integers(len(t['V']), size=t['T']),
        'zero-shot frame argmax': lambda t: M.frame_argmax(t['S']),
        'calibrated+gated argmax': lambda t: t['E'].argmax(1),
        'calibrated+gated, Viterbi (no script)': lambda t: M.monotone_free(t['E'], t['one_way']),
        'script + video (hard DP)': lambda t: M.script_dp(t['E'], t['script']),
        'script + video (soft)': lambda t: M.script_soft(t['E'], t['script']),
        'control: script + other video (soft)': lambda t: M.script_soft(other(t), t['script']),
        **({} if not vlm else {
            'VLM: nearest keyframe answer': lambda t: M.vlm_hold(t['vlm'], t['V'], t['T'], t['script'][0]),
            'VLM: Viterbi (no script)': lambda t: M.monotone_free(M.vlm_E(t['vlm'], t['V'], t['T']), t['one_way']),
            'script + VLM (soft)': lambda t: M.script_soft(M.vlm_E(t['vlm'], t['V'], t['T']), t['script']),
            'script + VLM (hard DP)': lambda t: M.script_dp(M.vlm_E(t['vlm'], t['V'], t['T']), t['script']),
            'control: script + other-video VLM (soft)': lambda t: M.script_soft(other_vlm(t), t['script']),
            'script + VLM + encoder (soft)': lambda t: M.script_soft(M.vlm_E(t['vlm'], t['V'], t['T']) + .1 * t['E'], t['script']),
        }),
    }










def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schema', default='s0')
    ap.add_argument('--enc', default=M.ENCODERS[0])
    ap.add_argument('--vlm', default=None)
    ap.add_argument('--vlm_every', type=int, default=1)
    ap.add_argument('--name', default='attribute-predictions')
    a = ap.parse_args()
    schema = json.load(open(f'{HERE}/schemas/schema_{a.schema}.json'))
    tracks = load_tracks(schema, a.enc, a.vlm, a.vlm_every)
    meths = methods(np.random.default_rng(0), tracks, a.vlm)
    rows = []
    for t in tracks:
        rows.append({'video': t['vid'], 'object': t['o'], 'attribute': t['a'],
                     'source_frame_indices': list(range(0, 30 * t['T'], 30)),
                     'predictions': {k: [t['V'][int(i)] for i in f(t)] for k, f in meths.items()}})
    out = os.path.join(ROOT, 'runs', a.name)
    os.makedirs(out, exist_ok=True)
    json.dump({'args': vars(a), 'predictions': rows}, open(os.path.join(out, 'predictions.json'), 'w'))
    print(out)

if __name__ == '__main__':
    main()
