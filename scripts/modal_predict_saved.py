"""Run original ChangeIt temporal padding on already extracted clip features."""
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_detector import base_image

app = modal.App('vidosc-saved-feature-prediction')
image = (base_image
         .run_commands('git clone https://github.com/facebookresearch/VidOSC.git /opt/vidosc',
                       'git -C /opt/vidosc checkout 59575773c97878ea30a0228f28ab00a3ee2f1ea2')
         .add_local_dir(ROOT/'scripts', '/root/scripts', ignore=['__pycache__/**', '*.local.sh'])
         .add_local_dir(ROOT/'moscato', '/root/moscato', ignore=['__pycache__/**'])
         .add_local_file(ROOT/'checkpoints/changeit/state_best.ckpt', '/inputs/vidosc.ckpt'))


@app.function(image=image, cpu=2, memory=4096, timeout=180, retries=0,
              max_containers=1, scaledown_window=2)
def predict_saved(features: bytes):
    import io
    import numpy as np
    import torch
    from moscato.checkpoint_adapter import predict
    torch.set_num_threads(2)
    Path('/tmp/features.pth.tar').write_bytes(features)
    state, action, info = predict('/inputs/vidosc.ckpt', '/tmp/features.pth.tar', '/opt/vidosc', pad_to=900)
    stream = io.BytesIO()
    np.savez_compressed(stream, state=state, action=action)
    return stream.getvalue(), info


@app.local_entrypoint()
def main(run: str):
    import json
    import shutil
    folder = Path(run)
    fm = json.loads((folder/'feature_manifest.json').read_text())
    if fm.get('model_info', {}).get('transformer_timesteps') == 900:
        raise ValueError('This run already uses 900-step temporal padding')
    backup = folder/'raw_probabilities_unpadded.npz'
    if backup.exists():
        raise FileExistsError(backup)
    probabilities, info = predict_saved.remote((folder/'combined_features.pth.tar').read_bytes())
    shutil.copy2(folder/'raw_probabilities.npz', backup)
    shutil.copy2(folder/'feature_manifest.json', folder/'feature_manifest_unpadded.json')
    (folder/'raw_probabilities.npz').write_bytes(probabilities)
    fm['model_info'] = info
    fm['temporal_padding'] = 'zero-pad input to 900 timesteps, no attention mask, then retain first 107 outputs; upstream ChangeIt loader'
    (folder/'feature_manifest.json').write_text(json.dumps(fm, indent=2)+'\n')
    print(f'Saved padded inference; preserved original unpadded probabilities: {folder}')
