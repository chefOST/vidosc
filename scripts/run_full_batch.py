"""Sequential inference over the available R2 release, with explicit compute budget.

The local coordinator must stay running. Failures are recorded per video and paid
jobs are never automatically retried. Reservations are not a provider billing cap.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.batch_vidosc import fingerprint, run_stage
from scripts.log_inference_wandb import receipt_matches
from scripts.prepare_full_video_plan import (build_plan, validate_experiment, validate_timeout,
                                             validate_compute, DEFAULT_EXPERIMENT, DEFAULT_TIMEOUT)
from weights_and_biases.config import WANDB_ENTITY, WANDB_PROJECT

WEIGHT_SHA256 = {
    "checkpoints/changeit/state_best.ckpt": "e6d1fa0d4708c8dffecba2a37f25149cc80c85fed33e6d4537bcaf98b0d94d36",
    "checkpoints/internvideo/InternVideo-MM-L-14.ckpt": "a508e42229072f84a3fcd5fc3acdb7ebc969b27519ee1f56dac9996999cd40f6",
}


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def selected_rows(inventory, requested):
    rows = inventory["videos"]
    ids = [row["video_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Inventory contains duplicate video IDs")
    available = {row["video_id"]: row for row in rows if row.get("available") is True}
    if requested:
        missing = set(requested) - set(available)
        if missing:
            raise ValueError("Requested videos are absent or unavailable: " + ", ".join(sorted(missing)))
        return [available[video_id] for video_id in dict.fromkeys(requested)]
    return list(available.values())


def worker_hourly_rate(rates, accelerator="cpu"):
    validate_compute(accelerator, 1)
    values = [rates[name] for name in ("cpu_hour_cost", "mem_gib_hour_cost")]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Modal CPU and memory rates must be finite positive numbers")
    hourly = 4 * values[0] + 16 * values[1]
    if accelerator != "cpu":
        gpu_rates = rates.get("gpu_hour_cost", {})
        gpu_rate = gpu_rates.get(accelerator) if isinstance(gpu_rates, dict) else None
        if (not isinstance(gpu_rate, (int, float)) or isinstance(gpu_rate, bool)
                or not math.isfinite(gpu_rate) or gpu_rate <= 0):
            raise ValueError(f"Missing finite positive hourly GPU rate for {accelerator}")
        hourly += gpu_rate
    if not math.isfinite(hourly):
        raise ValueError("Combined Modal rate must be finite")
    return hourly


def validate_inventory_source(inventory):
    config = json.loads((ROOT / "configs/r2_release.json").read_text())
    if any(inventory["r2"].get(name) != config[name] for name in
           ("bucket", "endpoint", "prefix", "manifest_key", "release_sha256")):
        raise ValueError("Inventory R2 source differs from the pinned release; regenerate inventory")


def validate_saved_compute(record, accelerator, batch_size):
    saved_accelerator, saved_batch_size = record.get("accelerator", "cpu"), record.get("batch_size", 1)
    validate_compute(saved_accelerator, saved_batch_size)
    if (saved_accelerator, saved_batch_size) != (accelerator, batch_size):
        raise ValueError("Saved compute settings differ; use a fresh experiment directory")


def validate_existing_source(folder, row, accelerator="cpu", batch_size=1):
    manifest = json.loads((folder / "feature_manifest.json").read_text())
    validate_saved_compute(manifest, accelerator, batch_size)
    if (manifest.get("video_id") != row["video_id"]
            or manifest.get("video_sha256") != row["sha256"]
            or manifest.get("dataset", "CMU") != row["dataset"]
            or manifest.get("clip_start_seconds") != 0
            or not math.isclose(float(manifest["clip_end_seconds"]),
                                row["metadata_frame_count"] / row["fps"], rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Saved outputs differ from this R2 video/timeline; use a fresh experiment directory")


def preflight_runtime():
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"Install the repository environment first: {python}")
    subprocess.run([str(python), "-c", "import modal, numpy, wandb, boto3"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([str(python), "scripts/setup_accounts.py", "--check"],
                   cwd=ROOT, check=True, stdin=subprocess.DEVNULL)


def preflight_weights():
    for name, expected in WEIGHT_SHA256.items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weights: {path}")
        if fingerprint(path) != expected:
            raise ValueError(f"Model checkpoint fingerprint mismatch: {path}")


def preflight_plan(plan, dataset):
    produced = set()
    for stage in plan["jobs"][0]["stages"]:
        for name in stage["inputs"]:
            if name not in produced and not Path(name).is_file():
                raise FileNotFoundError(f"Missing stage input: {name}")
        produced.update(stage["outputs"])
    if dataset.upper() == "CMU":
        mapping = json.loads((ROOT / "configs/changeit_moscato_alias_mapping.json").read_text())
        if mapping["checkpoint_sha256"] != WEIGHT_SHA256["checkpoints/changeit/state_best.ckpt"]:
            raise ValueError("CMU mapping refers to different model weights")
        if fingerprint(ROOT / "data/moscato/vocabulary/state_dict.json") != mapping["target_vocabulary_sha256"]:
            raise ValueError("CMU vocabulary differs from the fixed mapping")


def execute_plan(plan, log_path):
    with log_path.open('a', buffering=1) as log:
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            for stage in plan['jobs'][0]['stages']:
                run_stage(stage, execute=True, allow_modal=True)
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(old_out, 1); os.dup2(old_err, 2)
            os.close(old_out); os.close(old_err)


def upload(folder, experiment):
    subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/log_inference_wandb.py",
                    "--run", str(folder), "--group", experiment], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--budget-usd', type=float, help='Explicit approved budget required for execution')
    p.add_argument('--experiment', default=DEFAULT_EXPERIMENT)
    p.add_argument('--video-id', action='append', help='Restrict to this available video; repeat for a subset')
    p.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
    p.add_argument('--accelerator', choices=('cpu', 'L4'), default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    try:
        validate_experiment(a.experiment)
        validate_timeout(a.timeout)
        validate_compute(a.accelerator, a.batch_size)
        if a.budget_usd is not None and (not math.isfinite(a.budget_usd) or a.budget_usd <= 0):
            raise ValueError('Budget must be finite and greater than zero')
        if a.execute and a.budget_usd is None:
            raise ValueError('--execute requires an explicit --budget-usd')
        preparation = ROOT / 'runs/batch-preparation'
        inventory = json.loads((preparation / 'inventory.json').read_text())
        validate_inventory_source(inventory)
        rows = selected_rows(inventory, a.video_id)
        if not rows:
            raise ValueError('Inventory has no available videos')
        hourly = worker_hourly_rate(json.loads((preparation / 'modal_rates.json').read_text()), a.accelerator)
    except (OSError, KeyError, TypeError, ValueError) as error:
        p.error(str(error))
    reserve = (a.timeout + 10) / 3600 * hourly
    if not math.isfinite(reserve):
        p.error('Worker reservation must be finite')
    if not a.execute:
        print(json.dumps({'videos': [row['video_id'] for row in rows],
                          'worker_hourly_usd': hourly, 'worker_timeout_seconds': a.timeout,
                          'reservation_per_video_usd': reserve, 'budget_usd': a.budget_usd,
                          'reservation_selected_batch_usd': len(rows) * reserve + 0.75,
                          'accelerator': a.accelerator, 'batch_size': a.batch_size,
                          'experiment': a.experiment, 'metrics': False}, indent=2))
        return

    directory = ROOT / 'runs/batches' / a.experiment
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'batch.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        status_path = directory / 'batch_status.json'
        state = json.loads(status_path.read_text()) if status_path.exists() else {
            'experiment': a.experiment, 'reserved_compute_usd': 0.0,
            'setup_transfer_reserve_usd': 0.75, 'videos': {}}
        if state['experiment'] != a.experiment or any(
                not isinstance(state[name], (int, float)) or isinstance(state[name], bool)
                or not math.isfinite(state[name]) or state[name] < 0
                for name in ('reserved_compute_usd', 'setup_transfer_reserve_usd')):
            raise ValueError('Invalid saved budget ledger; inspect it before running')
        if status_path.exists():
            validate_saved_compute(state, a.accelerator, a.batch_size)
        state.update(budget_usd=a.budget_usd, worker_hourly_usd=hourly, state='running',
                     accelerator=a.accelerator, batch_size=a.batch_size,
                     selected_video_ids=[row['video_id'] for row in rows])
        state.pop('next_video', None)
        items = []
        # Prepare and check every selected video before any paid stage starts.
        for row in rows:
            video_id = row['video_id']
            receipt = directory / video_id / 'model/wandb_inference_receipt.json'
            try:
                if receipt.is_file():
                    validate_existing_source(receipt.parent, row, a.accelerator, a.batch_size)
                    previous = json.loads(receipt.read_text())
                    validate_saved_compute(previous, a.accelerator, a.batch_size)
                    if receipt_matches(previous, WANDB_ENTITY, WANDB_PROJECT, a.experiment):
                        state['videos'][video_id] = {**state['videos'].get(video_id, {}),
                                                    'state': 'uploaded', **previous}
                        continue
                    items.append((row, receipt, None))
                else:
                    previous_state = state['videos'].get(video_id, {})
                    if previous_state.get('paid_attempted') or previous_state.get('started_utc'):
                        raise ValueError('A paid attempt already started for this video; one-try execution forbids relaunch')
                    plan = build_plan(row, inventory['r2'], a.experiment, a.timeout, root=ROOT,
                                      accelerator=a.accelerator, batch_size=a.batch_size)
                    preflight_plan(plan, row['dataset'])
                    save(preparation / f'{a.experiment}-{video_id}.json', plan)
                    items.append((row, receipt, plan))
            except Exception as error:
                state['videos'][video_id] = {**state['videos'].get(video_id, {}),
                                            'state': 'failed', 'error': str(error)}
        if items:
            try:
                preflight_runtime()
            except Exception as error:
                for row, _, _ in items:
                    state['videos'][row['video_id']] = {'state': 'failed', 'error': str(error)}
                items = []
        if any(plan is not None for _, _, plan in items):
            try:
                preflight_weights()
            except Exception as error:
                for row, _, plan in items:
                    if plan is not None:
                        state['videos'][row['video_id']] = {'state': 'failed', 'error': str(error)}
                items = [item for item in items if item[2] is None]
        paid_items = [item for item in items if item[2] is not None]
        batch_reserve = len(paid_items) * reserve
        required_budget = state['reserved_compute_usd'] + batch_reserve + state['setup_transfer_reserve_usd']
        if not math.isfinite(required_budget):
            raise ValueError('Combined batch reservation must be finite')
        state.update(batch_reservation_usd=batch_reserve, required_budget_usd=required_budget)
        if paid_items and required_budget > a.budget_usd:
            state.update(state='budget_stop', next_video=paid_items[0][0]['video_id'])
            save(status_path, state)
            return
        # Reserve the complete selected paid workload before launching its first video.
        state['reserved_compute_usd'] += batch_reserve
        save(status_path, state)
        for row, receipt, plan in items:
            video_id = row['video_id']
            if plan is not None:
                state['videos'][video_id] = {'state': 'running', 'reserved_compute_usd': reserve,
                    'paid_attempted': True, 'accelerator': a.accelerator, 'batch_size': a.batch_size,
                    'started_utc': datetime.now(timezone.utc).isoformat()}
                save(status_path, state)
            try:
                if plan is not None:
                    print(f'Starting {video_id}; reserved worker compute ${reserve:.3f}', flush=True)
                    execute_plan(plan, directory / f'{video_id}.log')
                upload(receipt.parent, a.experiment)
                state['videos'][video_id] = {**state['videos'].get(video_id, {}),
                                            'state': 'uploaded', **json.loads(receipt.read_text())}
            except Exception as error:
                state['videos'][video_id] = {**state['videos'].get(video_id, {}),
                                            'state': 'failed', 'error': str(error)}
            save(status_path, state)
            print(video_id, state['videos'][video_id]['state'], flush=True)
        else:
            state['state'] = 'complete' if all(state['videos'].get(row['video_id'], {}).get('state') == 'uploaded'
                                             for row in rows) else 'finished_with_failures'
        save(status_path, state)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
