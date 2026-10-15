"""Slider pipeline (LLM script + video timing) on the 30 GT videos, logged to W&B.
ChefOST experiment, zero-shot; MOSCATO used only for scoring (sl.EVAL). See README.md.

Usage (from vidosc/): .venv/bin/python experiments/sliders/run_wandb.py [--name ryan/...] [--no_wandb]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bench as B, script_align as S, sl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SLIDERS = ['fullness', 'dirt', 'mixed', 'done', 'cut', 'heat']
METHODS = {'video': S.M['script + vis-hold readout'],
           'no_video': S.M['script, equal split (no video)'],
           'no_script': S.M['no script: vis-hold d>0']}


def acc(y, p):
    fa = float((p == y).mean())
    ba = float(.5 * (p[y].mean() + (~p[~y]).mean())) if y.any() and (~y).any() else float('nan')
    return ba, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=f"ryan/sliders-script-align-{time.strftime('%m%d-%H%M%S')}")
    ap.add_argument('--anchors', default='both', choices=['generic', 'recipe', 'both'])
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--no_wandb', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    tracks, _ = B.load(None, a.anchors)
    rng = np.random.default_rng(0)
    ev = {id(t): S.ev(t) for t in tracks}

    def other(t):  # control: evidence from another video's track of the same slider
        c = [u for u in tracks if u['s'] == t['s'] and u['vid'] != t['vid']]
        e = ev[id(c[rng.integers(len(c))])]
        return np.interp(np.linspace(0, 1, t['T']), np.linspace(0, 1, len(e)), e)
    methods = {**METHODS, 'control_other_video': lambda t, sc: S.align(sc, other(t))}

    rows, pooled, per_vid, plots = [], {}, {}, {}
    for t in tracks:
        recipe = t['vid'].split('_')[1]
        sc = S.SC.get(recipe, {}).get(t['o'], {}).get(t['s'])
        if sc is None:
            continue
        m = t['y'] >= 0; y = t['y'][m] >= .5
        preds = {k: f(t, sc) for k, f in methods.items()}
        row = {'video': t['vid'], 'recipe': recipe, 'object': t['o'], 'slider': t['s'], 'script': json.dumps(sc),
               'frames': t['T'], 'eval_frames': int(m.sum()), 'gt_high_rate': float(y.mean()),
               'gt_changes': bool(y.any() and (~y).any())}
        bounds = np.flatnonzero(np.diff(preds['video']) != 0) + 1
        row['pred_change_s'] = json.dumps(bounds.tolist())
        gtb = np.flatnonzero(np.diff((t['y'] >= .5).astype(int)[m]) != 0)
        row['gt_change_s'] = json.dumps(np.flatnonzero(m)[gtb + 1].tolist())
        for k, p in preds.items():
            pb = p[m] >= .5
            row[f'{k}/bal_acc'], row[f'{k}/frame_acc'] = acc(y, pb)
            pooled.setdefault((t['s'], k), [[], []]); pooled[(t['s'], k)][0].append(y); pooled[(t['s'], k)][1].append(pb)
            per_vid.setdefault((t['s'], k), {}).setdefault(t['vid'], []).append((y, pb))
        rows.append(row)
        plots.setdefault(t['s'], []).append((t, sc, ev[id(t)], preds['video'], preds['no_video']))

    summary = {}
    for s in SLIDERS:
        for k in methods:
            if (s, k) not in pooled:
                continue
            y = np.concatenate(pooled[(s, k)][0]); p = np.concatenate(pooled[(s, k)][1])
            summary[f'{s}/{k}/bal_acc'], summary[f'{s}/{k}/frame_acc'] = acc(y, p)
        summary[f'{s}/tracks'] = sum(r['slider'] == s for r in rows)
        summary[f'{s}/videos'] = len(per_vid[(s, 'video')])
        summary[f'{s}/eval_frames'] = int(sum(len(x) for x in pooled[(s, 'video')][0]))
        # bootstrap over videos: gain of video timing over the even split (bal acc; frame acc if one GT class)
        vids = list(per_vid[(s, 'video')])
        def score(k, vs):
            yy = np.concatenate([y for v in vs for y, _ in per_vid[(s, k)][v]])
            pp = np.concatenate([p for v in vs for _, p in per_vid[(s, k)][v]])
            ba, fa = acc(yy, pp); return fa if np.isnan(ba) else ba
        d = []
        for _ in range(a.boot):
            vs = rng.choice(vids, len(vids)); d.append(score('video', vs) - score('no_video', vs))
        summary[f'{s}/gain_vs_no_video'] = score('video', vids) - score('no_video', vids)
        summary[f'{s}/gain_ci_lo'], summary[f'{s}/gain_ci_hi'] = map(float, np.percentile(d, [2.5, 97.5]))
    summary['runtime_s'] = time.time() - t0
    summary['tracks'] = len(rows)

    # Timelines: per slider, every track (evidence, script+video, even split, GT).
    figs = {}
    for s, items in plots.items():
        n = len(items); cols = 4; rws = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rws, cols, figsize=(16, 2.1 * rws), squeeze=False)
        for ax, (t, sc, e, pv, pn) in zip(axs.flat, items):
            x = np.arange(t['T']); m = t['y'] >= 0
            ax.plot(x, (e - e.min()) / (np.ptp(e) + 1e-8), color='0.75', lw=.8, label='evidence')
            ax.plot(x, pv, color='C0', lw=1.6, label='script+video')
            ax.plot(x, pn, color='C1', lw=1, ls='--', label='even split')
            ax.scatter(x[m], t['y'][m], s=3, color='k', label='GT', zorder=3)
            ax.set_title(f"{t['vid'].split('_7150991')[0]} | {t['o']} {sc}", fontsize=8); ax.set_ylim(-.1, 1.1)
            ax.tick_params(labelsize=6)
        for ax in list(axs.flat)[n:]:
            ax.axis('off')
        axs.flat[0].legend(fontsize=6, loc='lower right')
        fig.suptitle(f'{s}: x = seconds, y = slider level', fontsize=10); fig.tight_layout()
        figs[s] = fig

    out = os.path.join(ROOT, 'runs', a.name.replace('/', '__'))
    os.makedirs(out, exist_ok=True)
    json.dump({'args': vars(a), 'summary': summary, 'per_track': rows}, open(os.path.join(out, 'results.json'), 'w'), indent=1)
    for s, f in figs.items():
        f.savefig(os.path.join(out, f'timeline_{s}.png'), dpi=100)
    for s in SLIDERS:
        if f'{s}/video/bal_acc' in summary:
            print(f"{s:9s} video {summary[f'{s}/video/bal_acc']:.3f}/{summary[f'{s}/video/frame_acc']:.3f}  "
                  f"no-video {summary[f'{s}/no_video/bal_acc']:.3f}/{summary[f'{s}/no_video/frame_acc']:.3f}  "
                  f"gain CI [{summary[f'{s}/gain_ci_lo']:+.3f}, {summary[f'{s}/gain_ci_hi']:+.3f}]")
    print(f'[saved] {out}')
    if a.no_wandb:
        return

    import wandb
    dest = json.load(open(os.path.join(ROOT, 'weights_and_biases', 'settings.json')))
    load = lambda f: json.load(open(os.path.join(HERE, f)))
    run = wandb.init(entity=dest['entity'], project=dest['projects']['experiments'], name=a.name,
                     job_type='sliders', group='sliders',
                     tags=['sliders', 'zero-shot', 'proposal-metric', 'llm-not-blind'],
                     notes='LLM slider script (recipe+object) + video timing via DP over visibility-gated InternVideo '
                           'readouts. Metric: range hit vs GT mapped to slider levels (eval-side map, unconfirmed). '
                           'Proposal metric, not the README precision. LLM outputs not blind (author saw MOSCATO labels).',
                     config={**vars(a), 'encoder': sl.ENC, 'fps': 1, 'videos': len(sl.ds.videos),
                             'evidence': 'z(hi-lo text sim), visibility-weighted gaussian smoothing (sigma=10s, top 30% presence)',
                             'decoder': 'DP split into script segments, min segment 3 s',
                             'eval': 'per-frame range hit (pred>=.5)==(GT>=.5); bal acc pooled per slider; bootstrap over videos',
                             'eval_map': sl.EVAL, 'slider_prompts': {k: v[:3] for k, v in sl.SL.items()},
                             'scripts': load('scripts.json'), 'object_sliders': load('object_sliders.json')})
    cols = list(rows[0])
    run.log({'per_track': wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows]),
             'summary_table': wandb.Table(columns=['slider', 'method', 'bal_acc', 'frame_acc'],
                                          data=[[s, k, summary.get(f'{s}/{k}/bal_acc'), summary.get(f'{s}/{k}/frame_acc')]
                                                for s in SLIDERS for k in methods if f'{s}/{k}/bal_acc' in summary]),
             **{f'timeline/{s}': wandb.Image(f) for s, f in figs.items()}})
    run.summary.update(summary)
    art = wandb.Artifact('slider-llm-inputs', type='config')
    for f in ['scripts.json', 'object_sliders.json', 'recipe_anchors.json', 'sl.py', 'script_align.py', 'bench.py']:
        art.add_file(os.path.join(HERE, f))
    run.log_artifact(art)
    print(f'W&B: {run.url}')
    run.finish()


if __name__ == '__main__':
    main()
