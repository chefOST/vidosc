"""Extract genuine InternVideo-MM-L14 global features from a frame-bounded clip.

Uses the official visual encoder and its inference preprocessing. One-second
windows and eight evenly sampled frames are explicit choices, recorded in the
manifest; exact parity with VidOSC's original extraction pipeline is unverified.
Does not synthesize the object-centric half required by det=1 checkpoints.
"""
import argparse
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
import torch

from .baseline import digest, write


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_encoder(checkpoint, source, device):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['video','video-id','object','checkpoint','source','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--start-frame',type=int,required=True)
    p.add_argument('--end-frame',type=int,required=True)
    p.add_argument('--fps',type=float,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(f'Use a fresh output directory: {out}')
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate,duration','-of','json',args.video]))['streams'][0]
    numerator,denominator = map(float,probe['avg_frame_rate'].split('/'))
    if abs(numerator/denominator-args.fps)>1e-6:
        raise ValueError('Requested FPS differs from video metadata')
    windows=window_indices(args.start_frame,args.end_frame,args.fps)
    print(f'Loading InternVideo on {args.device}; {len(windows)} windows',flush=True)
    model,preprocess=load_encoder(args.checkpoint,args.source,args.device)
    features=[]
    started=time.monotonic()
    with torch.inference_mode():
        for i,frames in enumerate(sampled_clips(args.video,windows)):
            tensor=preprocess(frames).unsqueeze(0).to(args.device)
            feat=model.encode_video(tensor)
            feat=torch.nn.functional.normalize(feat,dim=-1).cpu().float()
            if feat.shape!=(1,768) or not torch.isfinite(feat).all():
                raise ValueError('Unexpected encoder output')
            features.append(feat)
            if i==0 or (i+1)%10==0 or i==len(windows)-1:
                print(f'{i+1}/{len(windows)} windows, elapsed {time.monotonic()-started:.1f}s',flush=True)
    out.mkdir(parents=True)
    path=out/'global_features.pth.tar'
    torch.save(torch.cat(features),path)
    source_commit=subprocess.check_output(['git','-C',args.source,'rev-parse','HEAD'],text=True).strip()
    write(out/'feature_manifest.json',{
        'video_id':args.video_id,'object':args.object,'video_sha256':digest(args.video),
        'clip_start_seconds':args.start_frame/args.fps,'clip_end_seconds':args.end_frame/args.fps,
        'interval_starts_seconds':list(range(len(windows))), 'sampled_global_frame_indices':windows,
        'feature_backbone':'InternVideo-MM-L-14','feature_layout':'global_only_768',
        'feature_normalization':'L2','checkpoint_sha256':digest(args.checkpoint),
        'feature_sha256':digest(path),'upstream_commit':source_commit,'device':args.device,
        'preprocessing':'official demo Resize(256), CenterCrop(224), RGB normalization',
        'sampling':'8 uniform frames within each one-second interval; final interval shortened',
        'original_vidosc_extraction_parity_verified':False,
        'object_features_present':False,
    })
    print(f'Saved {path}',flush=True)


if __name__=='__main__':
    main()
