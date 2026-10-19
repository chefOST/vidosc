"""Run the attribute pipeline on the 30 MOSCATO GT videos (CMU, 1 fps) and score it.
Scoring (eval only): GT words -> attribute values via evalmap; a frame is scored for an attribute iff its GT words
assert some value; correct iff the predicted value is among the asserted ones.
Usage: python evaluate.py [--schema s0] [--enc internvideo-mm-l14] [--map claude-opus-5-5] [--wandb --name ...]"""
import argparse, json, os, time
from collections import defaultdict
import numpy as np
from common import HERE, ROOT, ds, recipe
import method as M


def load_tracks(schema, emap, enc, vlm=None, vlm_every=1):
    emb = M.text_emb(M.schema_texts(schema), enc)
    tracks = []
    for vid in ds.videos:
        F = M.feats(vid, enc); T = len(F); idx = list(range(0, 30 * T, 30)); r = recipe(vid)
        VL = M.load_vlm(f'{HERE}/vlm_cache/{vlm}/{vid}.jsonl') if vlm else {}
        if vlm and not VL: continue
        if vlm_every > 1: VL = {k: [(t, v) for t, v in x if t % vlm_every == 0] for k, x in VL.items()}
        for o in ds.objects(vid):
            sch = schema[r].get(o); wm = emap[r].get(o, {})
            if not sch: continue
            G = ds.gt_states(vid, o, idx)
            for a in map(M.norm_attr, sch['attributes']):
                V = a['values']
                gt = [set(x for w in g for x in wm.get(w, {}).get(a['name'], [])) for g in G]
                if not any(gt): continue
                S, E, pres = M.evidence(F, a, sch['visible'], emb)
                tracks.append(dict(vid=vid, recipe=r, o=o, a=a['name'], V=V, one_way=bool(a.get('one_way')),
                                   script=[V.index(v) for v in a['script']], T=T, S=S, E=E, pres=pres,
                                   gt=[frozenset(V.index(x) for x in g) for g in gt], vlm=VL.get((o, a['name']), [])))
    return tracks


def coverage(schema, emap):
    """Share of (video, object, frame) with GT words where >=1 word asserts some attribute value."""
    n = c = 0
    for vid in ds.videos:
        r = recipe(vid)
        for o in ds.objects(vid):
            wm = emap[r].get(o, {})
            for g in ds.gt[vid]['state'][o][::30]:
                if g: n += 1; c += any(wm.get(w) for w in g)
    return c / n


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
        'oracle: best constant (uses GT)': lambda t: np.full(t['T'], max(range(len(t['V'])), key=lambda v: sum(v in g for g in t['gt']))),
    }


def dynamic(t):
    g = [x for x in t['gt'] if x]
    return any(not (a & b) for a in set(g) for b in set(g))


def track_scores(t, p):
    m = np.array([bool(g) for g in t['gt']]); hit = np.array([p[i] in t['gt'][i] for i in range(t['T'])])[m]
    cls = [t['gt'][i] for i in np.flatnonzero(m)]
    bal = np.mean([hit[[c == k for c in cls]].mean() for k in set(cls)])
    return hit, bal


def summarize(rows, vids=None):
    """rows: list of (vid, dyn, hit array, bal). Returns micro acc (all), micro acc (dyn), mean bal acc (dyn)."""
    if vids is not None:
        byv = defaultdict(list)
        for r in rows: byv[r[0]].append(r)
        rows = [r for v in vids for r in byv[v]]
    allh = np.concatenate([r[2] for r in rows]); dyn = [r for r in rows if r[1]]
    return dict(acc=allh.mean(), acc_dyn=np.concatenate([r[2] for r in dyn]).mean(), bacc_dyn=np.mean([r[3] for r in dyn]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schema', default='s0'); ap.add_argument('--enc', default=M.ENCODERS[0])
    ap.add_argument('--map', default='claude-opus-5-5'); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--wandb', action='store_true'); ap.add_argument('--name', default=None)
    ap.add_argument('--quiet', action='store_true'); ap.add_argument('--vlm', default=None, help='vlm_cache subdir, e.g. s0-8b')
    ap.add_argument('--vlm_every', type=int, default=1, help='keep VLM keyframes with t %% n == 0 (cache stride is 6 s)')
    a = ap.parse_args(); t0 = time.time()
    if a.vlm and a.vlm_every > 1: a.vlm_every *= 6  # stride in multiples of the 6 s cache stride
    schema = json.load(open(f'{HERE}/schemas/schema_{a.schema}.json'))
    emap = json.load(open(f'{HERE}/schemas/evalmap_schema_{a.schema}_{a.map}.json'))
    tracks = load_tracks(schema, emap, a.enc, a.vlm, a.vlm_every)
    rng = np.random.default_rng(0); meths = methods(rng, tracks, a.vlm)
    res = {k: [] for k in meths}; per_track = []; preds = {}
    for t in tracks:
        d = dynamic(t); row = dict(video=t['vid'], recipe=t['recipe'], object=t['o'], attribute=t['a'], dynamic=d,
                                   values=json.dumps(t['V']), script=json.dumps([t['V'][i] for i in t['script']]))
        for k, f in meths.items():
            p = f(t); preds[(id(t), k)] = p; hit, bal = track_scores(t, p)
            res[k].append((t['vid'], d, hit, bal)); row[k] = float(bal)
        per_track.append(row)
    vids = sorted({t['vid'] for t in tracks})
    summ = {k: summarize(v) for k, v in res.items()}
    base, main_k = 'prior: script, even split (no video)', 'script + video (soft)'
    boots = defaultdict(list)
    for _ in range(a.boot):
        vs = rng.choice(vids, len(vids))
        for k in res:
            s = summarize(res[k], vs); b = summarize(res[base], vs)
            for m in s: boots[(k, m)].append(s[m] - b[m])
    cov = coverage(schema, emap)
    nd = sum(dynamic(t) for t in tracks)
    if not a.quiet:
        print(f'schema={a.schema} enc={a.enc} map={a.map}  tracks={len(tracks)} dynamic={nd} videos={len(vids)} GT coverage={cov:.3f}')
        print(f'{"method":42s} {"acc(all)":>9s} {"acc(dyn)":>9s} {"bacc(dyn)":>10s}   bacc(dyn) gain vs prior, 95% CI')
        for k, s in summ.items():
            lo, hi = np.percentile(boots[(k, 'bacc_dyn')], [2.5, 97.5])
            print(f'{k:42s} {s["acc"]:9.3f} {s["acc_dyn"]:9.3f} {s["bacc_dyn"]:10.3f}   [{lo:+.3f}, {hi:+.3f}]')
    out = dict(args=vars(a), tracks=len(tracks), dynamic_tracks=nd, videos=len(vids), coverage=cov,
               summary={k: {m: float(x) for m, x in s.items()} for k, s in summ.items()},
               ci={f'{k}|{m}': [float(x) for x in np.percentile(v, [2.5, 97.5])] for (k, m), v in boots.items()},
               runtime_s=time.time() - t0)
    name = a.name or f'attrs-{a.schema}-{a.enc.split("-")[0]}-{a.map.split("-")[1]}' + (f'-vlm{a.vlm}' if a.vlm else '') + (f'-every{a.vlm_every}' if a.vlm_every > 1 else '')
    od = f'{ROOT}/runs/{name.replace("/", "__")}'; os.makedirs(od, exist_ok=True)
    json.dump(dict(out, per_track=per_track), open(f'{od}/results.json', 'w'), indent=1)
    if a.wandb:
        import wandb_log; wandb_log.log(name, a, out, per_track, tracks, preds, meths, schema, emap)
    return out, tracks, preds


if __name__ == '__main__':
    main()
