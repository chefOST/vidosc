"""Verify run accounts; optionally copy the existing R2 profile into your Modal workspace."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weights_and_biases.config import WANDB_ENTITY, WANDB_PROJECT


def modal_workspace():
    result = subprocess.run([sys.executable, '-m', 'modal', 'token', 'info'],
                            capture_output=True, text=True,
                            env={**os.environ, 'NO_COLOR': '1', 'TERM': 'dumb'})
    match = re.search(r'Workspace:\s+(\S+)', result.stdout)
    if result.returncode or not match:
        raise RuntimeError('Modal login required: .venv/bin/python -m modal token new')
    return match.group(1)


def check_accounts(create_r2_secret=False, expected_workspace=None, profile='chefost-r2'):
    import modal
    from modal.config import config
    workspace = modal_workspace()
    if expected_workspace is not None and workspace != expected_workspace:
        raise RuntimeError(f'Modal workspace is {workspace}, expected {expected_workspace}')
    print(f'Modal workspace: {workspace}', flush=True)
    environment = config.get('environment') or ''
    names = {secret.name for secret in modal.Secret.objects.list(environment_name=environment)}
    if 'chefost-r2' not in names:
        if not create_r2_secret:
            raise RuntimeError('Missing Modal Secret chefost-r2; run setup_accounts.py --r2-secret --workspace YOUR_WORKSPACE')
        if expected_workspace is None:
            raise RuntimeError('--workspace is required before copying R2 credentials to Modal')
        import boto3
        credentials = boto3.Session(profile_name=profile).get_credentials()
        if credentials is None:
            raise RuntimeError(f'Missing AWS profile credentials: {profile}')
        credentials = credentials.get_frozen_credentials()
        fields = {'AWS_ACCESS_KEY_ID': credentials.access_key, 'AWS_SECRET_ACCESS_KEY': credentials.secret_key}
        if credentials.token:
            fields['AWS_SESSION_TOKEN'] = credentials.token
        modal.Secret.objects.create('chefost-r2', fields, environment_name=environment)
        print('Created chefost-r2 from the local AWS profile; credential values are not displayed.', flush=True)
    print(f'Modal R2 secret present in environment {environment or "workspace default"}.', flush=True)
    import wandb
    try:
        api = wandb.Api(timeout=30)
        viewer = api.viewer
        if viewer is None or api.project(WANDB_PROJECT, entity=WANDB_ENTITY) is None:
            raise RuntimeError('W&B account or project unavailable')
    except Exception:
        raise RuntimeError(f'W&B login/project access required: .venv/bin/python -m wandb login; destination {WANDB_ENTITY}/{WANDB_PROJECT}') from None
    print(f'W&B account authenticated; project readable: {WANDB_ENTITY}/{WANDB_PROJECT}.', flush=True)
    print('No inference, model deployment, W&B run or artifact upload was started.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only account checks (the default)')
    parser.add_argument('--r2-secret', action='store_true', help='Create a missing chefost-r2 secret from the local AWS profile')
    parser.add_argument('--workspace', help='Expected Modal workspace; required to create the secret')
    parser.add_argument('--profile', default='chefost-r2', help='Local AWS profile containing read-only R2 credentials')
    args = parser.parse_args()
    try:
        check_accounts(args.r2_secret, args.workspace, args.profile)
    except Exception as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
