"""Prepare independent visual features and disjoint label files for Shray's pilot."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import os

import numpy as np
import torch
import open_clip
from PIL import Image

HERE = Path(__file__).resolve().parent
def resolve_workspace():
    configured = os.environ.get('SHRAY_WORKSPACE')
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in HERE.parents:
        if (parent / 'work/VidOSC/model.py').is_file():
            return parent
    raise RuntimeError(
        'Set SHRAY_WORKSPACE to the dependency workspace containing work/VidOSC, '
        'work/chefost-data, and work/model-cache. See this experiment README.'
    )


WORKSPACE = resolve_workspace()
DATA = WORKSPACE / 'work/shray-multistate'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def label_path(config, name):
    prefix = config.get('data_label_prefix','')
    if prefix and (Path(prefix).name != prefix or prefix in {'.','..'}):
        raise ValueError('Label prefix must be one filename component')
    return DATA / f'{prefix+"_" if prefix else ""}{name}.json'


def load_upstream_encoder():
    import importlib.util
    from argparse import Namespace
    path = WORKSPACE / 'work/VidOSC/model.py'
    spec = importlib.util.spec_from_file_location('vidosc_readonly', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = torch.load(WORKSPACE / 'work/VidOSC/checkpoints/chopping.ckpt', map_location='cpu', weights_only=False)['state_dict']
    args = Namespace(input_dim=1536, vocab_size=4, transformer_dim=512,
                     transformer_heads=4, transformer_layers=3, transformer_dropout=0.1)
    model = module.FeatTimeTransformer(args)
    model.load_state_dict({key[6:]: value for key, value in raw.items() if key.startswith('model.')}, strict=True)
    model.head = torch.nn.Identity()
    model.eval()
    model.requires_grad_(False)
    return model


def video_path(video_id):
    path = DATA / f'{video_id}.avi'
    if path.exists():
        return path
    return WORKSPACE / 'work/chefost-vidosc/data/moscato/videos' / f'{video_id}.avi'


def probe(path):
    result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames,avg_frame_rate,width,height', '-of', 'json', str(path)])
    stream = next(item for item in json.loads(result)['streams'] if 'nb_read_frames' in item)
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    fps = numerator / denominator
    if fps <= 0:
        raise ValueError(f'Invalid frame rate {fps}')
    return int(stream['nb_read_frames']), fps


def extract(config, batch_size, custom_video_id=None, custom_source=None):
    DATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get('feature_threads',4))
    model, _, preprocess = open_clip.create_model_and_transforms(config['clip_model'], pretrained=config['clip_pretrained'])
    model.eval()
    encoder = load_upstream_encoder()
    vocab = read(WORKSPACE / 'work/chefost-vidosc/data/moscato/vocabulary/state_dict.json')
    labels = sorted(label.strip().lower() for label in vocab['s2i'])
    prompts = [f"A photo of {config['object']} in the {label} state." for label in labels]
    text_path = DATA / 'text_embeddings.pt'
    if custom_video_id and text_path.exists():
        text_cache = torch.load(text_path,map_location='cpu',weights_only=True)
        if text_cache['states'] != labels or text_cache['prompts'] != prompts:
            raise ValueError('Cached candidate embeddings use different labels or prompts')
    else:
        with torch.inference_mode():
            texts = torch.nn.functional.normalize(model.encode_text(open_clip.get_tokenizer(config['clip_model'])(prompts)).float(), dim=-1)
        torch.save({'states': labels, 'prompts': prompts, 'embeddings': texts}, text_path)
    manifest_path = DATA / (f'feature_manifest-{custom_video_id}.json' if custom_video_id else 'feature_manifest.json')
    manifest = read(manifest_path) if manifest_path.exists() else {}
    custom_split = next((split for split in ('train','validation','test') if custom_video_id in config[f'{split}_videos']), 'inference')
    splits = {custom_split:[custom_video_id]} if custom_video_id else {
        split:config[f'{split}_videos'] for split in ('train','validation','test')}
    for split, video_ids in splits.items():
        for video_id in video_ids:
            output = DATA / f'{video_id}.features.pt'
            if output.exists():
                saved = torch.load(output, map_location='cpu', weights_only=True)
                manifest[video_id] = saved['metadata']
                print('Using cached features:', video_id, flush=True)
                continue
            source = Path(custom_source).resolve() if custom_source else video_path(video_id)
            count, fps = probe(source)
            stride = int(fps * config['sample_seconds'])
            anchors = list(range(0, count, stride))
            raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(source),
                '-vf', f'select=not(mod(n\\,{stride})),scale=224:224', '-fps_mode', 'passthrough',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
            frames = np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3)
            if len(frames) != len(anchors):
                raise ValueError(f'Frame sampling mismatch: {len(frames)} vs {len(anchors)}')
            features = []
            print(f'Encoding {video_id}: {len(frames)} frames, split={split}', flush=True)
            with torch.inference_mode():
                for start in range(0, len(frames), batch_size):
                    batch = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames[start:start+batch_size]])
                    features.append(torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1))
                    print(f'  {min(start+batch_size, len(frames))}/{len(frames)}', flush=True)
                visual = torch.cat(features)
                # Initialization from the original checkpoint, with its original input
                # layout. The representation mismatch is recorded, not hidden.
                temporal = encoder(torch.cat([visual, torch.zeros_like(visual)], dim=-1).unsqueeze(0), middle=True).squeeze(0).clone()
            metadata = {'video_id': video_id, 'split': split, 'fps': fps, 'raw_frame_count': count,
                'sample_seconds': config['sample_seconds'], 'sample_anchors': anchors,
                'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'feature_model': config['clip_model'], 'temporal_encoder': 'frozen official VidOSC chopping encoder',
                'feature_warning': 'CLIP differs from the pretrained InternVideo input; object features zero-padded.',
                'labels_used_for_feature_extraction': False}
            torch.save({'visual': visual.clone(), 'temporal': temporal.clone(), 'metadata': metadata}, output)
            manifest[video_id] = metadata
            write(manifest_path, manifest)
            print('Saved', output, flush=True)
    write(manifest_path, manifest)


def labels(config):
    annotations = WORKSPACE / 'work/chefost-data/MOSCATO/annotations'
    pseudo = read(annotations / 'pseudo_labels/CMU/pseudo_annotations_poc_new_cmu.json')
    gt = read(annotations / 'ground_truth/CMU/gt_annotations_cmu.json')
    splits = [set(config[f'{split}_videos']) for split in ('train','validation','test')]
    if any(splits[a] & splits[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Videos must be disjoint between train/validation/test')
    manual_train = set(config.get('manual_train_videos',[]))
    if not manual_train.issubset(splits[0]):
        raise ValueError('Manual training labels must belong to declared training videos')
    if manual_train and config.get('protocol') != 'custom-supervised':
        raise ValueError('Manual GT training requires an explicitly custom supervised protocol')
    if (splits[0] & set(gt)) - manual_train:
        raise ValueError('Pilot training must exclude all manually annotated benchmark test videos')
    for split in ('train','validation','test'):
        selected = {}
        for video_id in config[f'{split}_videos']:
            use_manual = split != 'train' or video_id in manual_train
            source = gt if use_manual else pseudo
            fps = config['evaluation_label_fps'] if use_manual else config['training_label_fps']
            selected[video_id] = {'states':source[video_id]['state'][config['object']],'fps':fps}
        write(label_path(config,f'{split}_labels'), selected)
    write(label_path(config,'split_manifest'), {'train':config['train_videos'], 'validation':config['validation_videos'],
        'test':config['test_videos'], 'training_source':('manual GT, custom supervised protocol' if manual_train else 'MOSCATO pseudo-labels only'),
        'training_excludes_all_gt_videos':not bool(manual_train), 'manual_training_videos':sorted(manual_train),
        'validation_source':'manual GT; model/threshold selection',
        'test_source':'manual GT; scoring saved predictions after training'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features','labels'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--video-id', help='Prepare only this new video; no annotations required')
    parser.add_argument('--video-source', type=Path, help='Path to a video for custom feature extraction')
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    args = parser.parse_args()
    config = read(args.config)
    if args.video_source and not args.video_id:
        parser.error('--video-source needs --video-id')
    if args.video_id and (Path(args.video_id).name != args.video_id or args.video_id in {'.','..'}):
        parser.error('Video ID must be one filename stem')
    if args.stage == 'features':
        extract(config, args.batch_size, args.video_id, args.video_source)
    else:
        labels(config)
