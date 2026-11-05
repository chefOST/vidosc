"""InternVideo-MM-L14 encoder and sampling helpers for the Modal worker.

Uses the official visual encoder and its inference preprocessing. One-second
windows and eight evenly sampled frames are explicit choices, recorded in the
manifest; exact parity with VidOSC's original extraction pipeline is unverified.
Does not synthesize the object-centric half required by det=1 checkpoints.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import numpy as np

from .baseline import digest

FEATURE_ATOL = 1e-5
FEATURE_RTOL = 1e-3
FEATURE_COSINE_MIN = 0.99999


def validate_compute(accelerator, batch_size):
    if accelerator not in ('cpu', 'L4'):
        raise ValueError('Accelerator must be cpu or L4')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 16:
        raise ValueError('Batch size must be between 1 and 16')


def compare_features(actual, reference, label):
    """Compare normalized FP32 features before accepting any new batch checkpoint."""
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError(f'{label}: invalid numerical comparison')
    difference = np.abs(actual - reference)
    norms = np.linalg.norm(actual, axis=-1) * np.linalg.norm(reference, axis=-1)
    if (norms == 0).any():
        raise ValueError(f'{label}: zero encoder features')
    cosine = float(np.min(np.sum(actual * reference, axis=-1) / norms))
    passed = bool(np.allclose(actual, reference, atol=FEATURE_ATOL, rtol=FEATURE_RTOL)
                  and cosine >= FEATURE_COSINE_MIN)
    result = {'passed': passed, 'max_abs_error': float(difference.max()),
              'mean_abs_error': float(difference.mean()),
              'min_cosine_similarity': cosine}
    if not passed:
        raise ValueError(f'{label}: encoder features diverged (max abs error '
                         f'{result["max_abs_error"]:.6g}, cosine={cosine:.7g}; '
                         f'atol={FEATURE_ATOL}, rtol={FEATURE_RTOL}, cosine>={FEATURE_COSINE_MIN})')
    return result


def extract_feature_batches(video, windows, crops, model, preprocess, device, batch_size,
                            global_features, object_features, global_completed, object_completed,
                            checkpoint):
    """Decode once for both streams; checkpoint only completely encoded prefixes.

    `checkpoint(stream, features, completed)` receives CPU tensors. Missing object
    crops retain zero rows and still advance the completed prefix. The first batch
    of each stream checks every entry against singleton inference, plus
    one CPU reference when using CUDA. No automatic OOM fallback or paid retry.
    """
    import torch
    validate_compute('L4' if device == 'cuda' else 'cpu', batch_size)
    count = len(windows)
    if len(crops) != count or not 0 <= global_completed <= count or not 0 <= object_completed <= count:
        raise ValueError('Feature batches require matching crop/window counts and valid resume positions')
    streams = {
        'global': {'features': global_features, 'completed': global_completed},
        'object': {'features': object_features, 'completed': object_completed},
    }
    for stream in streams.values():
        stream.update(indices=[], tensors=[], seen=stream['completed'], saved=stream['completed'], checked=False)
    report = {'status': 'passed', 'dtype': 'float32', 'atol': FEATURE_ATOL, 'rtol': FEATURE_RTOL,
              'cosine_minimum': FEATURE_COSINE_MIN,
              'cpu_reference_count': 0, 'checks': [], 'seconds': 0.0}
    elapsed = {'global': 0.0, 'object': 0.0}

    def encode(prepared, target):
        value = prepared[0].unsqueeze(0) if len(prepared) == 1 else torch.stack(prepared, dim=0)
        try:
            features = model.encode_video(value.float().to(target))
            features = torch.nn.functional.normalize(features, dim=-1).cpu().float()
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError(f'Encoder batch size {len(prepared)} exceeded {target} memory; '
                                   'no automatic retry was attempted') from error
            raise
        if features.shape != (len(prepared), 768) or not torch.isfinite(features).all():
            raise ValueError('Invalid batched encoder output')
        return features

    def check_batch(name, stream, features):
        validation_started = time.monotonic()
        positions = list(range(len(stream['tensors'])))
        singleton = [encode([stream['tensors'][position]], device).numpy()[0] for position in positions]
        entry = {'stream': name, 'batch_size': len(stream['tensors']),
                 'window_indices': [stream['indices'][position] for position in positions],
                 'batch_vs_singleton': compare_features(features.numpy()[positions], singleton,
                                                       f'{name} batch versus singleton')}
        if device == 'cuda' and report['cpu_reference_count'] == 0:
            # Reuse the same parameters; avoid keeping a second L14 encoder in RAM.
            model.to('cpu')
            try:
                reference = encode([stream['tensors'][0]], 'cpu')
            finally:
                model.to(device)
            entry['gpu_vs_cpu'] = compare_features(features.numpy()[:1], reference.numpy(),
                                                  f'{name} GPU versus CPU')
            report['cpu_reference_count'] += 1
        report['checks'].append(entry)
        report['seconds'] += time.monotonic() - validation_started
        stream['checked'] = True

    def flush(name, final=False):
        stream = streams[name]
        started = time.monotonic()
        if stream['tensors']:
            features = encode(stream['tensors'], device)
            if not stream['checked']:
                check_batch(name, stream, features)
            for row, index in enumerate(stream['indices']):
                stream['features'][index] = features[row]
            stream['indices'].clear()
            stream['tensors'].clear()
        stream['completed'] = stream['seen']
        if stream['completed'] > stream['saved'] and (final or stream['completed'] - stream['saved'] >= 10):
            checkpoint(name, stream['features'], stream['completed'])
            stream['saved'] = stream['completed']
        elapsed[name] += time.monotonic() - started

    with torch.inference_mode():
        for index, frames in enumerate(sampled_clips(video, windows)):
            for name, resume in (('global', global_completed), ('object', object_completed)):
                if index < resume:
                    continue
                stream = streams[name]
                started = time.monotonic()
                box = crops[index]['crop_xyxy_exclusive'] if name == 'object' else None
                if name == 'global' or box is not None:
                    if name == 'object':
                        x1, y1, x2, y2 = box
                        clip = [frame[y1:y2, x1:x2] for frame in frames]
                    else:
                        clip = frames
                    stream['indices'].append(index)
                    stream['tensors'].append(preprocess(clip))
                else:
                    stream['features'][index] = 0
                stream['seen'] = index + 1
                elapsed[name] += time.monotonic() - started
                if len(stream['tensors']) >= batch_size or (not stream['tensors'] and stream['seen'] - stream['saved'] >= 10):
                    flush(name)
        for name in streams:
            flush(name, final=True)
    if not report['checks']:
        report['status'] = 'no_new_nonzero_features'
    return report, elapsed


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_encoder(checkpoint, source, device):
    import torch
    source = Path(source).resolve()
    package = source / 'InternVideo1/Pretrain/Multi-Modalities-Pretraining/InternVideo'
    if not (package / 'clip_utils/model.py').is_file():
        raise FileNotFoundError(f'Official InternVideo inference source not found: {package}')
    # Import only the encoder, without __init__ side effects from the upstream
    # text tokenizer / decord demo loader (decord has no macOS wheel).
    for name, directory in [('_iv_encoder', package), ('_iv_encoder.clip_utils', package / 'clip_utils')]:
        namespace = types.ModuleType(name)
        namespace.__path__ = [str(directory)]
        sys.modules[name] = namespace
    module = load_module('_iv_encoder.clip_utils.model', package / 'clip_utils/model.py')
    transform = load_module('_iv_video_transform', package / 'video_transform.py')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['state_dict']
    state = {k.removeprefix('clip.'): v for k, v in state.items() if k.startswith('clip.')}
    if tuple(state['visual.conv1.weight'].shape) != (1024, 3, 1, 14, 14):
        raise ValueError('Expected InternVideo-MM-L-14 weights')
    model = module.build_model(state_dict=state)
    # Upstream builder uses strict=False; verify that every visual encoder
    # parameter was supplied before accepting the model.
    missing = set(model.state_dict()) - set(state)
    visual_missing = [k for k in missing if k.startswith(('visual.', 'visual_ln_post.', 'visual_proj'))]
    if visual_missing:
        raise ValueError(f'Missing visual encoder weights: {visual_missing}')
    # Text branches are not used by encode_video. Release them before transfer.
    del model.transformer, model.token_embedding
    del state
    model = model.float().eval().to(device)
    preprocessing = transform.Compose([
        transform.Resize(256), transform.CenterCrop(224),
        transform.ClipToTensor(channel_nb=3),
        transform.Normalize(mean=[.48145466,.4578275,.40821073], std=[.26862954,.26130258,.27577711]),
    ])
    return model, preprocessing


def window_indices(start, end, fps):
    if start < 0 or end <= start or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid frame range or FPS')
    duration = (end-start)/fps
    windows = []
    for t in range(math.ceil(duration)):
        first = start + int(round(t*fps))
        last = min(end, start + int(round((t+1)*fps))) - 1
        if last < first:
            raise ValueError('Empty temporal sampling window')
        windows.append(np.linspace(first,last,8).astype(int).tolist())
    return windows


def sampled_clips(video, windows):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode {video}')
    # Sequential decoding avoids unreliable AVI seeking and keeps frame IDs exact.
    current = 0
    try:
        for indices in windows:
            wanted = set(indices)
            frames = {}
            while current <= indices[-1]:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Video ended at frame {current}, before requested {indices[-1]}')
                if current in wanted:
                    frames[current] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current += 1
            yield [frames[i] for i in indices]
    finally:
        cap.release()


def prepare_manifest(video, video_id, object_name, checkpoint, source, start, end, fps, device):
    """Describe exact encoder inputs before extraction, locally or on Modal."""
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate,duration','-of','json',str(video)]))['streams'][0]
    numerator, denominator = map(float, probe['avg_frame_rate'].split('/'))
    if abs(numerator/denominator-fps) > 1e-6:
        raise ValueError('Requested FPS differs from video metadata')
    windows = window_indices(start, end, fps)
    source_commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    return {
        'video_id':video_id, 'object':object_name, 'video_sha256':digest(video),
        'clip_start_seconds':start/fps, 'clip_end_seconds':end/fps,
        'interval_starts_seconds':list(range(len(windows))), 'sampled_global_frame_indices':windows,
        'feature_backbone':'InternVideo-MM-L-14', 'feature_layout':'global_only_768',
        'feature_normalization':'L2', 'checkpoint_sha256':digest(checkpoint),
        'upstream_commit':source_commit, 'device':device,
        'preprocessing':'official demo Resize(256), CenterCrop(224), RGB normalization',
        'sampling':'8 uniform frames within each one-second interval; final interval shortened',
        'original_vidosc_extraction_parity_verified':False,
        'object_features_present':False,
    }
