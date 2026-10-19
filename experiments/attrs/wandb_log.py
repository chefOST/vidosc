"""W&B telemetry for an evaluate.py run (milestone logging). Destination from weights_and_biases/settings.json."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, ROOT

TIMELINE = ['prior: script, even split (no video)', 'script + video (soft)', 'script + VLM (soft)']


def timelines(tracks, preds, meths, n=16):
    dyn = [t for t in tracks if any(not (a & b) for a in set(g for g in t['gt'] if g) for b in set(g for g in t['gt'] if g))][:n]
    ks = [k for k in TIMELINE if k in meths]
    fig, axs = plt.subplots(int(np.ceil(len(dyn) / 4)), 4, figsize=(16, 2.2 * np.ceil(len(dyn) / 4)), squeeze=False)
    for ax, t in zip(axs.flat, dyn):
        x = np.arange(t['T'])
        for i, g in enumerate(t['gt']):
            for v in g: ax.plot(i, v, 's', ms=2, color='k')
        for j, k in enumerate(ks):
            ax.plot(x, preds[(id(t), k)] + .08 * (j + 1), lw=1.2, label=k.split(' (')[0])
        ax.set_yticks(range(len(t['V']))); ax.set_yticklabels([v[:12] for v in t['V']], fontsize=6)
        ax.set_title(f"{t['vid'].split('_7150991')[0]} | {t['o']} / {t['a']}", fontsize=8); ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[len(dyn):]: ax.axis('off')
    axs.flat[0].legend(fontsize=6); fig.suptitle('black = GT-asserted values (eval map); x = seconds', fontsize=9); fig.tight_layout()
    return fig


def log(name, args, out, per_track, tracks, preds, meths, schema, emap, tags=()):
    import wandb
    dest = json.load(open(os.path.join(ROOT, 'weights_and_biases', 'settings.json')))
    run = wandb.init(entity=dest['entity'], project=dest['projects']['experiments'], name=name, group='attrs',
                     job_type='attr-pipeline', tags=['attrs', 'zero-shot', 'blind-llm', 'proposal-metric', *tags],
                     notes='Blind LLM attribute schema (recipe + object names) -> per-frame evidence (encoder or VLM) -> '
                           'script-aligned decoding. MOSCATO GT used only through an LLM eval map (words -> attribute values). '
                           'Proposal metric, not the README precision.',
                     config={**vars(args), 'fps': 1, 'tracks': out['tracks'], 'dynamic_tracks': out['dynamic_tracks'],
                             'videos': out['videos'], 'gt_coverage': out['coverage'],
                             'metric': 'bacc(dyn): per dynamic track, mean over distinct GT value-sets of frame accuracy; '
                                       'averaged over tracks. dynamic = two scored frames with disjoint GT value-sets.'})
    s = {}
    for k, m in out['summary'].items():
        for mm, v in m.items():
            s[f'{k}/{mm}'] = v
            lo, hi = out['ci'][f'{k}|{mm}']; s[f'{k}/{mm}_gain_ci_lo'], s[f'{k}/{mm}_gain_ci_hi'] = lo, hi
    run.summary.update({**s, 'coverage': out['coverage'], 'tracks': out['tracks'], 'dynamic_tracks': out['dynamic_tracks']})
    cols = list(per_track[0])
    run.log({'summary_table': wandb.Table(columns=['method', 'acc_all', 'acc_dyn', 'bacc_dyn', 'bacc_gain_lo', 'bacc_gain_hi'],
                                          data=[[k, m['acc'], m['acc_dyn'], m['bacc_dyn'], *out['ci'][f'{k}|bacc_dyn']]
                                                for k, m in out['summary'].items()]),
             'per_track': wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in per_track]),
             'timelines': wandb.Image(timelines(tracks, preds, meths))})
    art = wandb.Artifact(f'attrs-inputs-{args.schema}', type='config')
    for f in [f'schemas/schema_{args.schema}.json', f'schemas/evalmap_schema_{args.schema}_{args.map}.json',
              'generate.py', 'evalmap.py', 'method.py', 'evaluate.py', 'vlm_read.py', 'llm.py']:
        art.add_file(os.path.join(HERE, f))
    art.add_dir(os.path.join(HERE, 'llm_cache'), name='llm_cache')
    run.log_artifact(art)
    print('W&B:', run.url); run.finish()
