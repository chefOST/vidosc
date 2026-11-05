"""Log one active-object inference output bundle, without metrics or GT."""
import argparse
import json
from pathlib import Path
import hashlib
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weights_and_biases.config import WANDB_ENTITY, WANDB_PROJECT


def validate(folder):
    files = ['feature_manifest.json', 'combined_features.pth.tar',
             'object_features.pth.tar', 'raw_logits.npz', 'raw_probabilities.npz',
             'predictions/manifest.json', 'predictions/probabilities.npz',
             'predictions/predictions_by_second.csv']
    for name in files:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    fm = json.loads((folder / 'feature_manifest.json').read_text())
    if fm.get('ground_truth_used') is not False or fm.get('object') != 'active-object':
        raise ValueError('Expected an explicitly unnamed active-object inference bundle')
    timings = fm.get('stage_elapsed_seconds', {})
    if not isinstance(timings, dict) or any(
        not isinstance(stage, str) or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool) or not np.isfinite(seconds) or seconds < 0
        for stage, seconds in timings.items()
    ):
        raise ValueError('Invalid stage timing metadata')
    global_features = folder / 'global_features.pth.tar'
    if global_features.exists():
        with global_features.open('rb') as stream:
            actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_hash != fm.get('global_feature_sha256'):
            raise ValueError('Global feature fingerprint mismatch')
        files.append(global_features.name)
    n = len(fm['interval_starts_seconds'])
    with np.load(folder / 'raw_logits.npz') as logits, np.load(folder / 'raw_probabilities.npz') as probabilities:
        for name, width in [('state', 52), ('action', 28)]:
            raw, prob = logits[name + '_logits'], probabilities[name]
            if raw.shape != (n, width) or prob.shape != raw.shape or not np.isfinite(raw).all():
                raise ValueError('Invalid output shape or logits')
            exp = np.exp(raw - raw.max(1, keepdims=True))
            if not np.allclose(exp / exp.sum(1, keepdims=True), prob, atol=1e-5):
                raise ValueError('Logits do not reproduce saved probabilities')
    return fm, files


def receipt_matches(receipt, entity, project):
    """Accept saved destinations, including older receipts containing only a run URL."""
    from urllib.parse import urlparse, unquote
    parts = unquote(urlparse(receipt.get('url', '')).path).strip('/').split('/')
    try:
        run_index = parts.index('runs')
    except ValueError:
        run_index = 0
    url_entity, url_project = parts[run_index - 2:run_index] if run_index >= 2 else (None, None)
    return (receipt.get('entity', url_entity) == entity
            and receipt.get('project', url_project) == project)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--entity', default=WANDB_ENTITY)
    p.add_argument('--project', default=WANDB_PROJECT)
    p.add_argument('--group', default='cmu30-active-object-v1')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    folder = a.run.resolve()
    fm, files = validate(folder)
    if a.dry_run:
        print(f"Validated {fm['video_id']}: {len(fm['interval_starts_seconds'])} intervals; no metrics")
        return
    receipt = folder / 'wandb_inference_receipt.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if receipt_matches(previous, a.entity, a.project):
            print(receipt.read_text())
            return
        backup = folder / ('wandb_inference_receipt.' + hashlib.sha256(previous['url'].encode()).hexdigest()[:12] + '.json')
        backup.write_text(json.dumps(previous, indent=2) + '\n')
    config = {'video_id': fm['video_id'], 'scope': 'one unnamed active-object stream per video',
              'checkpoint_sha256': fm['vidosc_checkpoint_sha256'],
              'feature_sha256': fm['feature_sha256'], 'crop_policy': fm['crop_policy'],
              'metrics_computed': False, 'temporal_postprocessing': 'none',
              'input_timesteps': len(fm['interval_starts_seconds']),
              'duration_seconds': fm['clip_end_seconds'] - fm['clip_start_seconds'],
              'provenance': fm.get('reuse_provenance', 'fresh extraction')}
    for key in ('feature_extraction_device', 'global_intervals_reused'):
        if key in fm:
            config[key] = fm[key]
    summary = {'coverage/object_windows': fm['object_windows'],
               'coverage/missing_object_windows': fm['missing_object_windows'],
               'completion': 'outputs_saved'}
    for stage, seconds in fm.get('stage_elapsed_seconds', {}).items():
        summary[f'timing/{stage}_seconds'] = seconds
    if 'global_intervals_reused' in fm:
        summary['reuse/global_intervals'] = fm['global_intervals_reused']
    import wandb
    with wandb.init(entity=a.entity, project=a.project, group=a.group,
        id=hashlib.sha256((a.group + fm['video_id'] + fm['feature_sha256']).encode()).hexdigest()[:16],
        resume='allow', name=f"{fm['video_id']}-active-object", job_type='inference',
        tags=['vidosc', 'cmu30', 'active-object', 'outputs-only'],
        config=config) as run:
        run.summary.update(summary)
        artifact = wandb.Artifact(f"{fm['video_id']}-active-object-outputs", type='inference-outputs')
        for name in files:
            artifact.add_file(str(folder / name), name=name)
        run.log_artifact(artifact).wait()
        result = {'url': run.url, 'run_id': run.id, 'group': a.group,
                  'entity': a.entity, 'project': a.project}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(result['url'])


if __name__ == '__main__':
    main()
