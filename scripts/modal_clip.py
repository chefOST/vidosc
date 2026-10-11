"""Run a documented exploratory 107-second VidOSC inference on Modal CPU.

No MOSCATO ground truth or target-label mapping is uploaded. Raw source-index
probabilities are saved even when the training vocabulary is unavailable.
"""
import os
import sys
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_detector import base_image

app = modal.App('vidosc-s29-exploratory')
image = (
    base_image
    .apt_install('ffmpeg')
    .pip_install('timm==1.0.15', 'scikit-image==0.25.2', 'scikit-learn==1.6.1')
    .run_commands(
        'git clone --filter=blob:none --no-checkout https://github.com/OpenGVLab/InternVideo.git /opt/internvideo',
        'git -C /opt/internvideo sparse-checkout set InternVideo1/Pretrain/Multi-Modalities-Pretraining',
        'git -C /opt/internvideo checkout 3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3',
        'git clone https://github.com/facebookresearch/VidOSC.git /opt/vidosc',
        'git -C /opt/vidosc checkout 59575773c97878ea30a0228f28ab00a3ee2f1ea2',
    )
    .add_local_dir(ROOT / 'scripts', '/root/scripts', ignore=['__pycache__/**', '*.local.sh'])
    .add_local_dir(ROOT / 'moscato', '/root/moscato', ignore=['__pycache__/**', '*.local.sh'])
    .add_local_file(ROOT / 'data/moscato/videos/S29_Salad_7150991-1.avi', '/inputs/video.avi')
    .add_local_file(ROOT / 'data/moscato/features/s29-global/global_features.pth.tar', '/inputs/global.pth.tar')
    .add_local_file(ROOT / 'data/moscato/features/s29-global/feature_manifest.json', '/inputs/global.json')
    .add_local_file(ROOT / 'checkpoints/changeit/state_best.ckpt', '/inputs/vidosc.ckpt')
    .add_local_file(Path.home() / 'Downloads/InternVideo-MM-L-14.ckpt', '/inputs/internvideo.ckpt')
)


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0,
              max_containers=1, scaledown_window=2)
def run_clip():
    import gc
    import io
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from moscato.baseline import digest
    from moscato.hand_object import load_detector, detect_frame
    from moscato.object_crops import choose_crop, POLICY
    from moscato.extract_internvideo import sampled_clips, load_encoder
    from moscato.checkpoint_adapter import predict

    started = time.monotonic()
    torch.set_num_threads(4)
    manifest = json.loads(Path('/inputs/global.json').read_text())
    for path, expected in [('/inputs/video.avi', manifest['video_sha256']),
                           ('/inputs/global.pth.tar', manifest['feature_sha256']),
                           ('/inputs/internvideo.ckpt', manifest['checkpoint_sha256'])]:
        if digest(path) != expected:
            raise ValueError(f'Input fingerprint mismatch: {path}')
    windows = manifest['sampled_global_frame_indices']
    if len(windows) != 107:
        raise ValueError('This pilot is limited to the prepared 107 windows')
    detector, cfg = load_detector('/opt/hand-object', '/models/hand_object_100k.pth', 'cpu')
    crops = []
    # Choose a crop using one of the exact frames used by the global encoder.
    for index, frames in enumerate(sampled_clips('/inputs/video.avi', windows)):
        frame = frames[4]
        detections = detect_frame(detector, cfg, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 'cpu')
        box = choose_crop(detections, frame.shape[1], frame.shape[0])
        crops.append({'window': index, 'frame_index': windows[index][4],
                      'crop_xyxy_exclusive': box, 'detections': detections})
        if index % 10 == 0 or index == 106:
            print(f'Detection {index+1}/107; elapsed {time.monotonic()-started:.0f}s', flush=True)
    del detector
    gc.collect()
    model, preprocess = load_encoder('/inputs/internvideo.ckpt', '/opt/internvideo', 'cpu')
    object_features = torch.zeros((107, 768))
    with torch.inference_mode():
        for index, frames in enumerate(sampled_clips('/inputs/video.avi', windows)):
            box = crops[index]['crop_xyxy_exclusive']
            if box:
                x1, y1, x2, y2 = box
                clips = [f[y1:y2, x1:x2] for f in frames]
                feat = torch.nn.functional.normalize(model.encode_video(preprocess(clips).unsqueeze(0)), dim=-1)
                if feat.shape != (1, 768) or not torch.isfinite(feat).all():
                    raise ValueError('Invalid object encoder output')
                object_features[index] = feat[0].cpu()
            if index % 10 == 0 or index == 106:
                print(f'Object features {index+1}/107; elapsed {time.monotonic()-started:.0f}s', flush=True)
    del model
    gc.collect()
    global_features = torch.load('/inputs/global.pth.tar', weights_only=True, map_location='cpu')
    if global_features.shape != object_features.shape:
        raise ValueError('Global/object dimensions differ')
    features = torch.cat((global_features, object_features), dim=-1)
    torch.save(features, '/tmp/combined.pth.tar')
    torch.save(object_features, '/tmp/object.pth.tar')
    state, action, info = predict('/inputs/vidosc.ckpt', '/tmp/combined.pth.tar', '/opt/vidosc', pad_to=900)
    if state.shape != (107, 52) or action.shape != (107, 28):
        raise ValueError(f'Unexpected prediction shapes: {state.shape}, {action.shape}')
    output = io.BytesIO()
    np.savez_compressed(output, state=state, action=action)
    result_manifest = dict(manifest)
    result_manifest.update(
        feature_layout='global_768_then_object_768', object_features_present=True,
        feature_sha256=digest('/tmp/combined.pth.tar'),
        object_feature_sha256=digest('/tmp/object.pth.tar'),
        global_feature_sha256=manifest['feature_sha256'],
        crop_policy=POLICY, crops=crops,
        object_windows=sum(c['crop_xyxy_exclusive'] is not None for c in crops),
        missing_object_windows=sum(c['crop_xyxy_exclusive'] is None for c in crops),
        detector_checkpoint_sha256=digest('/models/hand_object_100k.pth'),
        vidosc_checkpoint_sha256=digest('/inputs/vidosc.ckpt'), model_info=info,
        elapsed_seconds=time.monotonic()-started,
        interpretation='exploratory transfer; raw source indices; not an official benchmark',
        ground_truth_used=False, prediction_device='Modal CPU',
    )
    return {'feature_manifest.json': json.dumps(result_manifest, indent=2).encode(),
            'combined_features.pth.tar': Path('/tmp/combined.pth.tar').read_bytes(),
            'object_features.pth.tar': Path('/tmp/object.pth.tar').read_bytes(),
            'raw_probabilities.npz': output.getvalue()}


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    output = ROOT / 'runs/modal-clip' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = run_clip.remote()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in result.items():
        (output / name).write_bytes(data)
    print(f'Saved real-video VidOSC predictions to {output}')
