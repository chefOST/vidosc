"""Save slider-script inference outputs; scoring is centralized in chefOST-data."""
import argparse
import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence as B
import script_align as S

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
METHODS = {'video': S.M['script + vis-hold readout'],
           'no_video': S.M['script, equal split (no video)'],
           'no_script': S.M['no script: vis-hold d>0']}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=f"ryan/sliders-{time.strftime('%m%d-%H%M%S')}")
    ap.add_argument('--anchors', default='both', choices=['generic', 'recipe', 'both'])
    ap.add_argument('--no_wandb', action='store_true')
    a = ap.parse_args()
    tracks, _ = B.load(None, a.anchors)
    rows = []
    for t in tracks:
        recipe = t['vid'].split('_')[1]
        script = S.SC.get(recipe, {}).get(t['o'], {}).get(t['s'])
        if script is None:
            continue
        rows.append({'video': t['vid'], 'object': t['o'], 'slider': t['s'], 'script': script,
                     'source_frame_indices': list(range(0, 30*t['T'], 30)),
                     'predictions': {k: f(t, script).tolist() for k, f in METHODS.items()}})
    out = os.path.join(ROOT, 'runs', a.name)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'predictions.json')
    json.dump({'args': vars(a), 'predictions': rows}, open(path, 'w'))
    if not a.no_wandb:
        import wandb
        dest = json.load(open(os.path.join(ROOT, 'weights_and_biases', 'settings.json')))
        run = wandb.init(entity=dest['entity'], project=dest['projects']['experiments'],
                         name=a.name, job_type='predictions', config=vars(a))
        artifact = wandb.Artifact(a.name.replace('/', '-'), type='predictions')
        artifact.add_file(path, name='predictions.json')
        run.log_artifact(artifact)
        run.finish()
    print(path)

if __name__ == '__main__':
    main()
