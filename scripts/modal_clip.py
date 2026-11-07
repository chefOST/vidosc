"""Extract features on Modal CPU/L4 and run exploratory VidOSC inference on CPU.

No MOSCATO ground truth or target-label mapping is uploaded. Raw source-index
probabilities are saved even when the training vocabulary is unavailable.
"""
import os
import sys
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moscato.paths import checkpoint_path
from scripts.modal_detector import base_image

R2_BUCKET = os.environ.get('VIDOSC_R2_BUCKET', 'chefost-moscato-gt-videos')
R2_ENDPOINT = os.environ.get('VIDOSC_R2_ENDPOINT', 'https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com')
R2_PREFIX = os.environ.get('VIDOSC_R2_PREFIX', 'moscato/releases/564f350e536734db/videos/')
RUN_TIMEOUT = int(os.environ.get('VIDOSC_TIMEOUT', '21600'))
if not 60 <= RUN_TIMEOUT <= 86400:
    raise ValueError('VIDOSC_TIMEOUT must be between 60 and 86400 seconds')
GPU = os.environ.get('VIDOSC_GPU', '')
if GPU not in ('', 'L4'):
    raise ValueError('VIDOSC_GPU must be empty or L4')
BATCH_SIZE = int(os.environ.get('VIDOSC_BATCH_SIZE', '1'))
if not 1 <= BATCH_SIZE <= 16:
    raise ValueError('VIDOSC_BATCH_SIZE must be between 1 and 16')
app = modal.App('vidosc-gpu-batching' if GPU else 'vidosc-exploratory')
progress_volume = modal.Volume.from_name('vidosc-active-object-progress', create_if_missing=True)
videos = modal.CloudBucketMount(bucket_name=R2_BUCKET, bucket_endpoint_url=R2_ENDPOINT,
                                key_prefix=R2_PREFIX, secret=modal.Secret.from_name('chefost-r2'), read_only=True)
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
    .add_local_file(checkpoint_path('changeit/state_best.ckpt', ROOT), '/inputs/vidosc.ckpt')
    .add_local_file(checkpoint_path('internvideo/InternVideo-MM-L-14.ckpt', ROOT), '/inputs/internvideo.ckpt')
)


@app.function(image=image, gpu=GPU or None, cpu=(4, 4), memory=(16384, 16384), timeout=RUN_TIMEOUT, retries=0,
              max_containers=1, scaledown_window=2, volumes={'/progress': progress_volume, '/r2-videos': videos})
def run_clip(video_id, frame_count, fps, r2_key, expected_sha256, expected_bytes, dataset, r2_prefix,
             accelerator='cpu', batch_size=1):
    import gc
    import io
    import json
    import time
    import hashlib
    import cv2
    import numpy as np
    import torch
    from moscato.baseline import digest
    from moscato.hand_object import load_detector, detect_frame
    from moscato.object_crops import choose_crop, POLICY
    from moscato.extract_internvideo import sampled_clips, load_encoder, prepare_manifest, validate_compute, extract_feature_batches
    from moscato.checkpoint_adapter import predict, read_weights, inspect_weights, inference_timesteps
    from moscato.progress_cache import save_features, load_features
    from moscato.r2_inputs import relative_key, stage_video

    started = time.monotonic()
    stage_elapsed = {}
    torch.set_num_threads(4)
    validate_compute(accelerator, batch_size)
    device = 'cuda' if accelerator == 'L4' else 'cpu'
    compute = {'accelerator': accelerator, 'batch_size': batch_size, 'dtype': 'float32', 'tf32': False}
    gpu_info = None
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('L4 requested but CUDA is unavailable; refusing CPU fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats()
        gpu_info = {'name': torch.cuda.get_device_name(), 'tf32_enabled': False}
    if dataset not in {'CMU', 'EGTEA'}:
        raise ValueError('Unsupported dataset')
    video = str(stage_video(Path('/r2-videos') / relative_key(r2_key, r2_prefix),
                            Path('/tmp/video.mp4'), expected_sha256, expected_bytes))
    stage_elapsed['input_transfer'] = time.monotonic() - started
    manifest = prepare_manifest(video, video_id, 'active-object',
                                '/inputs/internvideo.ckpt', '/opt/internvideo',
                                0, frame_count, fps, device)
    windows = manifest['sampled_global_frame_indices']
    count = len(windows)
    capacity = inspect_weights(read_weights('/inputs/vidosc.ckpt'))['max_timesteps']
    pad_to = inference_timesteps(count, capacity)
    worker_sha = digest('/root/scripts/modal_clip.py')
    # Global features can be reused independently of the hand detector/crop policy.
    global_identity = {'video': manifest['video_sha256'], 'windows': windows,
                       'encoder': manifest['checkpoint_sha256'], 'upstream': manifest['upstream_commit'],
                       'worker': worker_sha, 'compute': compute,
                       'code': {name: digest('/root/moscato/' + name + '.py') for name in
                                ['extract_internvideo', 'progress_cache']}}
    global_key = hashlib.sha256(json.dumps(global_identity, sort_keys=True).encode()).hexdigest()
    global_cache = Path('/progress/global') / global_key
    global_cache.mkdir(parents=True, exist_ok=True)
    global_features, global_completed = load_features(global_cache / 'features.pt', count)
    identity = {'video': manifest['video_sha256'], 'windows': windows,
                'encoder': manifest['checkpoint_sha256'], 'policy': POLICY,
                'upstream': manifest['upstream_commit'],
                'worker': worker_sha, 'compute': compute,
                'detector': digest('/models/hand_object_100k.pth'),
                'code': {name: digest('/root/moscato/' + name + '.py') for name in
                         ['hand_object', 'object_crops', 'extract_internvideo', 'progress_cache']}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path('/progress') / key
    cache.mkdir(parents=True, exist_ok=True)
    crop_path = cache / 'crops.json'
    crops = json.loads(crop_path.read_text()) if crop_path.exists() else []
    if len(crops) > count or any(c['window'] != i or c['frame_index'] != windows[i][4] for i,c in enumerate(crops)):
        raise ValueError('Invalid saved crop timeline')
    object_features, completed = load_features(cache / 'features.pt', count)
    if completed > len(crops):
        raise ValueError('Feature checkpoint extends beyond saved detections')
    validation_path = cache / 'numerical_validation.json'
    numerical_validation = json.loads(validation_path.read_text()) if validation_path.exists() else {
        'status': 'not_requested_cpu_singleton' if device == 'cpu' and batch_size == 1 else 'no_new_encoding',
        'checks': []}
    detector_validation_path = cache / 'detector_validation.json'
    detector_validation = json.loads(detector_validation_path.read_text()) if detector_validation_path.exists() else {
        'status': 'not_requested_cpu' if device == 'cpu' else 'no_new_detection'}
    print(f'Resumable cache: {global_completed}/{count} global, {len(crops)}/{count} detections, '
          f'{completed}/{count} object intervals', flush=True)
    stage_elapsed['setup'] = time.monotonic() - started - stage_elapsed['input_transfer']
    stage_started = time.monotonic()
    if len(crops) < count:
        detector, cfg = load_detector('/opt/hand-object', '/models/hand_object_100k.pth', device)
        # Choose a crop using one of the exact frames used by the global encoder.
        for index, frames in enumerate(sampled_clips(video, windows)):
            if index < len(crops):
                continue
            frame = frames[4]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            detections = detect_frame(detector, cfg, bgr, device)
            box = choose_crop(detections, frame.shape[1], frame.shape[0])
            if device == 'cuda' and detector_validation.get('status') != 'passed':
                detector.to('cpu')
                try:
                    cpu_detections = detect_frame(detector, cfg, bgr, 'cpu')
                finally:
                    detector.to(device)
                cpu_box = choose_crop(cpu_detections, frame.shape[1], frame.shape[0])
                difference = None if box is None or cpu_box is None else max(abs(a-b) for a,b in zip(box, cpu_box))
                passed = (box is None and cpu_box is None) or (difference is not None and difference <= 2)
                detector_validation = {'status': 'passed' if passed else 'failed',
                    'scope': 'first uncached detection frame only; not full detector parity',
                    'frame_index': windows[index][4], 'gpu_crop': box, 'cpu_crop': cpu_box,
                    'max_coordinate_error_pixels': difference, 'tolerance_pixels': 2}
                if not passed:
                    raise ValueError(f'GPU/CPU detector selected materially different crops: {box} versus {cpu_box}')
                detector_validation_path.write_text(json.dumps(detector_validation))
            crops.append({'window': index, 'frame_index': windows[index][4],
                          'crop_xyxy_exclusive': box, 'detections': detections})
            if (index + 1) % 10 == 0 or index == count - 1:
                temp = crop_path.with_suffix('.tmp')
                temp.write_text(json.dumps(crops))
                temp.replace(crop_path)
                progress_volume.commit()
            if index % 10 == 0 or index == count - 1:
                print(f'Detection {index+1}/{count}; elapsed {time.monotonic()-started:.0f}s', flush=True)
        del detector
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    stage_elapsed['detection'] = time.monotonic() - stage_started
    stage_elapsed['encoder_load'] = stage_elapsed['global_features'] = stage_elapsed['object_features'] = 0.0
    if global_completed < count or completed < count:
        stage_started = time.monotonic()
        model, preprocess = load_encoder('/inputs/internvideo.ckpt', '/opt/internvideo', device)
        stage_elapsed['encoder_load'] = time.monotonic() - stage_started
        if device == 'cpu' and batch_size == 1:
            # Preserve the original CPU singleton operation order and arithmetic.
            with torch.inference_mode():
                stage_started = time.monotonic()
                if global_completed < count:
                    for index, frames in enumerate(sampled_clips(video, windows)):
                        if index < global_completed:
                            continue
                        tensor = preprocess(frames).unsqueeze(0).to('cpu')
                        feat = model.encode_video(tensor)
                        feat = torch.nn.functional.normalize(feat, dim=-1).cpu().float()
                        if feat.shape != (1, 768) or not torch.isfinite(feat).all():
                            raise ValueError('Unexpected global encoder output')
                        global_features[index] = feat[0]
                        if (index + 1) % 10 == 0 or index == count - 1:
                            save_features(global_cache / 'features.pt', global_features, index + 1)
                            progress_volume.commit()
                        if index % 10 == 0 or index == count - 1:
                            print(f'Global features {index+1}/{count}; elapsed {time.monotonic()-started:.0f}s', flush=True)
                stage_elapsed['global_features'] = time.monotonic() - stage_started
                stage_started = time.monotonic()
                if completed < count:
                    for index, frames in enumerate(sampled_clips(video, windows)):
                        if index < completed:
                            continue
                        box = crops[index]['crop_xyxy_exclusive']
                        if box:
                            x1, y1, x2, y2 = box
                            clips = [f[y1:y2, x1:x2] for f in frames]
                            feat = torch.nn.functional.normalize(model.encode_video(preprocess(clips).unsqueeze(0)), dim=-1)
                            if feat.shape != (1, 768) or not torch.isfinite(feat).all():
                                raise ValueError('Invalid object encoder output')
                            object_features[index] = feat[0].cpu()
                        if (index + 1) % 10 == 0 or index == count - 1:
                            save_features(cache / 'features.pt', object_features, index + 1)
                            progress_volume.commit()
                        if index % 10 == 0 or index == count - 1:
                            print(f'Object features {index+1}/{count}; elapsed {time.monotonic()-started:.0f}s', flush=True)
                stage_elapsed['object_features'] = time.monotonic() - stage_started
        else:
            def checkpoint_batch(name, features, position):
                target = global_cache if name == 'global' else cache
                save_features(target / 'features.pt', features, position)
                progress_volume.commit()
                print(f'{name.capitalize()} features {position}/{count}; '
                      f'elapsed {time.monotonic()-started:.0f}s', flush=True)

            stage_started = time.monotonic()
            numerical_validation, feature_seconds = extract_feature_batches(
                video, windows, crops, model, preprocess, device, batch_size,
                global_features, object_features, global_completed, completed, checkpoint_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            stage_elapsed['feature_extraction_wall'] = time.monotonic() - stage_started
            stage_elapsed['global_features'] = feature_seconds['global']
            stage_elapsed['object_features'] = feature_seconds['object']
            numerical_validation['scope'] = 'first new batch of each stream; not full-run CPU parity'
            validation_path.write_text(json.dumps(numerical_validation))
            progress_volume.commit()
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    if global_features.shape != object_features.shape:
        raise ValueError('Global/object dimensions differ')
    features = torch.cat((global_features, object_features), dim=-1)
    torch.save(features, '/tmp/combined.pth.tar')
    torch.save(object_features, '/tmp/object.pth.tar')
    torch.save(global_features, '/tmp/global_features.pth.tar')
    stage_started = time.monotonic()
    state, action, info = predict('/inputs/vidosc.ckpt', '/tmp/combined.pth.tar', '/opt/vidosc',
                                  pad_to=pad_to, raw_output='/tmp/raw_logits.npz')
    stage_elapsed['inference'] = time.monotonic() - stage_started
    if state.shape != (count, 52) or action.shape != (count, 28):
        raise ValueError(f'Unexpected prediction shapes: {state.shape}, {action.shape}')
    output = io.BytesIO()
    np.savez_compressed(output, state=state, action=action)
    if device == 'cuda':
        torch.cuda.synchronize()
        gpu_info.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        gpu_info.update(peak_allocated_gib=gpu_info['peak_allocated_bytes'] / 1024**3,
                        peak_reserved_gib=gpu_info['peak_reserved_bytes'] / 1024**3)
    result_manifest = dict(manifest)
    result_manifest.update(
        dataset=dataset, source_r2_key=r2_key,
        feature_layout='global_768_then_object_768', object_features_present=True,
        feature_sha256=digest('/tmp/combined.pth.tar'),
        object_feature_sha256=digest('/tmp/object.pth.tar'),
        global_feature_sha256=digest('/tmp/global_features.pth.tar'),
        crop_policy=POLICY, crops=crops,
        object_windows=sum(c['crop_xyxy_exclusive'] is not None for c in crops),
        missing_object_windows=sum(c['crop_xyxy_exclusive'] is None for c in crops),
        detector_checkpoint_sha256=digest('/models/hand_object_100k.pth'),
        vidosc_checkpoint_sha256=digest('/inputs/vidosc.ckpt'), model_info=info,
        elapsed_seconds=time.monotonic()-started, stage_elapsed_seconds=stage_elapsed,
        interpretation='exploratory transfer; raw source indices; not an official benchmark',
        ground_truth_used=False, prediction_device='Modal CPU',
        feature_extraction_device='Modal L4' if device == 'cuda' else 'Modal CPU',
        accelerator=accelerator, batch_size=batch_size, feature_dtype='float32',
        tf32_enabled=False, gpu=gpu_info, numerical_validation=numerical_validation,
        detector_numerical_validation=detector_validation,
        global_resumable_cache_key=global_key,
        global_intervals_reused=global_completed,
        resumable_cache_key=key, object_intervals_reused=completed,
        output_stages={'raw_logits': 'raw_logits.npz',
                       'softmax_probabilities': 'raw_probabilities.npz',
                       'temporal_postprocessing': 'none; not applied'},
    )
    return {'feature_manifest.json': json.dumps(result_manifest, indent=2).encode(),
            'combined_features.pth.tar': Path('/tmp/combined.pth.tar').read_bytes(),
            'object_features.pth.tar': Path('/tmp/object.pth.tar').read_bytes(),
            'raw_logits.npz': Path('/tmp/raw_logits.npz').read_bytes(),
            'raw_probabilities.npz': output.getvalue()}


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    output = Path(os.environ.get('VIDOSC_OUTPUT', ROOT / 'runs/modal-clip' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    if output.exists():
        raise FileExistsError(output)
    # The planner supplies metadata; no feature extraction runs in this process.
    result = run_clip.remote(os.environ['VIDOSC_VIDEO_ID'],
                             int(os.environ['VIDOSC_FRAME_COUNT']), float(os.environ['VIDOSC_FPS']),
                             os.environ['VIDOSC_R2_KEY'], os.environ['VIDOSC_VIDEO_SHA256'],
                             int(os.environ['VIDOSC_VIDEO_BYTES']), os.environ['VIDOSC_DATASET'], R2_PREFIX,
                             GPU or 'cpu', BATCH_SIZE)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in result.items():
        (output / name).write_bytes(data)
    print(f'Saved real-video VidOSC predictions to {output}')
