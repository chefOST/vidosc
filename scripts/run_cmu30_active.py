"""Sequential CMU30 inference and W&B logging with conservative compute reservations.

No metrics or per-object ground-truth labels are passed to inference. Failed videos
are recorded, not automatically retried. This ledger is not a provider billing cap.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.batch_vidosc import run_stage


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--budget-usd', type=float, default=25.0)
    p.add_argument('--wait-s08', action='store_true')
    a = p.parse_args()
    if not 0 < a.budget_usd <= 25:
        p.error('This experiment is authorized for at most $25')
    experiment = 'cmu30-active-object-v1'
    directory = ROOT / 'runs/batches' / experiment
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'batch_status.json'
    manifest = json.loads((ROOT / 'configs/moscato_cmu30_v1.json').read_text())
    rates = json.loads((ROOT / 'runs/batch-preparation/modal_rates.json').read_text())
    hourly = 4 * float(rates['cpu_hour_cost']) + 16 * float(rates['mem_gib_hour_cost'])
    ids = manifest['video_ids']
    if not a.execute:
        print(json.dumps({'videos': ids, 'worker_hourly_usd': hourly,
                          'budget_usd': a.budget_usd, 'metrics': False}, indent=2))
        return
    defer_s08 = False
    if a.wait_s08:
        retry_status = directory / 'S08_Eggs_7150991-727/retry_status.json'
        print('Waiting for the existing S08 retry; no duplicate compute.', flush=True)
        while json.loads(retry_status.read_text())['state'] == 'running':
            time.sleep(30)
        defer_s08 = json.loads(retry_status.read_text())['state'] != 'uploaded'
    lock = directory / 'batch.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        'experiment': experiment, 'budget_usd': a.budget_usd,
        'reserved_compute_usd': 0.0, 'setup_transfer_reserve_usd': 0.75,
        'worker_hourly_usd': hourly, 'videos': {}, 'state': 'running'}
    state['budget_usd'] = a.budget_usd
    state['state'] = 'running'
    state.pop('pause_reason', None)
    inventory = {r['video_id']: r for r in json.loads((ROOT / 'runs/batch-preparation/inventory.json').read_text())['videos']}
    try:
        for vid in ids:
            if defer_s08 and vid == 'S08_Eggs_7150991-727':
                state['videos'][vid] = {'state': 'deferred',
                    'reason': 'Retry did not finish; move to S09 first. Saved remote progress retained.'}
                save(status_path, state)
                continue
            receipt = directory / vid / 'model/wandb_inference_receipt.json'
            if receipt.exists():
                state['videos'][vid] = {'state': 'uploaded', **json.loads(receipt.read_text())}
                save(status_path, state)
                continue
            plan_path = ROOT / 'runs/batch-preparation' / f'{experiment}-{vid}.json'
            subprocess.run([sys.executable, 'scripts/prepare_full_video_plan.py',
                '--video-id', vid, '--active-object', '--experiment', experiment], cwd=ROOT, check=True)
            plan = json.loads(plan_path.read_text())
            model_stage = next(s for s in plan['jobs'][0]['stages'] if s['name'] == 'model')
            model_stage['env']['VIDOSC_TIMEOUT'] = '7200'
            seconds = int(model_stage['env']['VIDOSC_TIMEOUT'])
            reserve = (seconds + 10) / 3600 * hourly
            if state['reserved_compute_usd'] + reserve + state['setup_transfer_reserve_usd'] > a.budget_usd:
                state['state'] = 'budget_stop'
                state['next_video'] = vid
                break
            state['reserved_compute_usd'] += reserve
            state['videos'][vid] = {'state': 'running', 'reserved_compute_usd': reserve,
                'started_utc': datetime.now(timezone.utc).isoformat()}
            save(status_path, state)
            print(f'Starting {vid}; reserved worker compute ${reserve:.3f}', flush=True)
            log_path = directory / f'{vid}.log'
            try:
                # Child processes inherit this per-video log via explicit FD redirection.
                with log_path.open('a', buffering=1) as log:
                    old_out, old_err = os.dup(1), os.dup(2)
                    try:
                        sys.stdout.flush(); sys.stderr.flush()
                        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
                        for stage in plan['jobs'][0]['stages']:
                            if stage['name'] == 'global' and (directory / vid / 'global/feature_manifest.json').exists():
                                from moscato.reuse_global import validate_global
                                row = inventory[vid]
                                validate_global(directory / vid / 'global', row['local_video'],
                                    Path.home() / 'Downloads/InternVideo-MM-L-14.ckpt',
                                    row['metadata_frame_count'], row['fps'])
                                print('Reused verified global features; no re-extraction.', flush=True)
                                continue
                            run_stage(stage, execute=True, allow_modal=True)
                        subprocess.run([sys.executable, 'scripts/log_inference_wandb.py',
                            '--run', str(directory / vid / 'model')], cwd=ROOT, check=True)
                    finally:
                        sys.stdout.flush(); sys.stderr.flush()
                        os.dup2(old_out, 1); os.dup2(old_err, 2)
                        os.close(old_out); os.close(old_err)
                state['videos'][vid].update(state='uploaded', **json.loads(receipt.read_text()))
            except Exception as e:
                state['videos'][vid].update(state='failed', error=str(e))
            save(status_path, state)
            print(vid, state['videos'][vid]['state'], flush=True)
        else:
            state['state'] = 'complete' if all(state['videos'].get(v, {}).get('state') == 'uploaded' for v in ids) else 'finished_with_failures'
        save(status_path, state)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
