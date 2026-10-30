"""One explicitly authorized S08 retry; other videos remain paused."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.batch_vidosc import run_stage


def main():
    folder = ROOT / 'runs/batches/cmu30-active-object-v1/S08_Eggs_7150991-727'
    status = folder / 'retry_status.json'
    if status.exists():
        raise RuntimeError('Retry already recorded; inspect before launching again')
    def record(state, **extra):
        status.write_text(json.dumps({'state': state, **extra}, indent=2) + '\n')
    record('running', timeout_seconds=3600)
    try:
        plan = json.loads((ROOT / 'runs/batch-preparation/cmu30-active-object-v1-S08_Eggs_7150991-727.json').read_text())
        for stage in plan['jobs'][0]['stages']:
            if stage['name'] == 'global':
                continue  # Already extracted successfully; remote code verifies its hash.
            if stage['name'] == 'model':
                stage['env']['VIDOSC_TIMEOUT'] = '3600'
            run_stage(stage, execute=True, allow_modal=True)
        subprocess.run([sys.executable, 'scripts/log_inference_wandb.py', '--run',
                        str(folder / 'model'), '--project', 'vidosc - full evals'], cwd=ROOT, check=True)
        receipt = json.loads((folder / 'model/wandb_inference_receipt.json').read_text())
        record('uploaded', **receipt)
    except Exception as e:
        record('failed', error=str(e))
        raise


if __name__ == '__main__':
    main()
