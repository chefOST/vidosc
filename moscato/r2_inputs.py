"""Validate the shared video release and copy verified inputs from its read-only mount."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


AVAILABLE_STATUSES = {'downloaded_verified_identity', 'downloaded_user_selected_identity'}


def relative_key(key, prefix):
    if not key.startswith(prefix):
        raise ValueError('Video key is outside the selected release')
    relative = key[len(prefix):]
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts or str(path) != relative:
        raise ValueError('Invalid video key')
    return relative


def inventory_from_manifest(payload, config, objects):
    """Select every available recording, rejecting missing or unlisted video objects."""
    if hashlib.sha256(payload).hexdigest() != config['release_sha256']:
        raise ValueError('R2 manifest checksum differs from the pinned release')
    manifest = json.loads(payload)
    rows, unavailable, seen = [], [], set()
    for entry in manifest['entries']:
        video_id = entry['ground_truth_video_id']
        if entry['download_status'] not in AVAILABLE_STATUSES:
            unavailable.append({'video_id': video_id, 'dataset': entry['dataset'],
                                'reason': entry['download_status']})
            continue
        if (not video_id or not all(c.isalnum() or c in '-_' for c in video_id)
                or video_id in seen):
            raise ValueError(f'Invalid or duplicate video ID: {video_id}')
        seen.add(video_id)
        if entry['dataset'] not in {'CMU', 'EGTEA'}:
            raise ValueError(f"Unsupported available dataset: {entry['dataset']}")
        key = config['prefix'] + entry['local_path']
        relative_key(key, config['prefix'])
        if not key.endswith('.mp4') or objects.get(key) != entry['bytes']:
            raise ValueError(f'Missing video or size mismatch on R2: {video_id}')
        sha = entry['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError(f'Invalid video checksum: {video_id}')
        streams = [s for s in entry['media']['streams'] if s.get('codec_type', 'video') == 'video']
        if len(streams) != 1:
            raise ValueError(f'Expected one video stream: {video_id}')
        stream = streams[0]
        fps, frames = float(Fraction(stream['avg_frame_rate'])), int(stream['nb_frames'])
        if not math.isfinite(fps) or fps <= 0 or frames <= 0 or entry['bytes'] <= 0:
            raise ValueError(f'Invalid video metadata: {video_id}')
        rows.append({'video_id': video_id, 'dataset': entry['dataset'], 'available': True,
                     'metadata_frame_count': frames, 'fps': fps, 'duration_seconds': frames / fps,
                     'r2_key': key, 'sha256': sha, 'bytes': entry['bytes'],
                     'identity_status': entry['download_status'],
                     'temporal_alignment_status': entry.get('temporal_alignment_status', 'unverified')})
    known = {row['r2_key'] for row in rows}
    extra = sorted(key for key in objects if key.lower().endswith(('.mp4', '.avi', '.mov')) and key not in known)
    if extra:
        raise ValueError(f'Unlisted videos in the pinned R2 release: {extra}')
    if not rows:
        raise ValueError('No available videos in the R2 release')
    return {'r2': dict(config), 'videos': sorted(rows, key=lambda row: (row['dataset'], row['video_id'])),
            'video_count': len(rows), 'unavailable': unavailable,
            'note': 'All available videos in the verified R2 release; no annotation labels used for inference.'}


def stage_video(source, destination, expected_sha256, expected_bytes):
    """Copy from the mount once, verifying bytes before any feature extraction."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    checksum, count = hashlib.sha256(), 0
    try:
        with source.open('rb') as src, temporary.open('wb') as dst:
            while chunk := src.read(8 * 1024 * 1024):
                checksum.update(chunk)
                count += len(chunk)
                dst.write(chunk)
        if count != expected_bytes or checksum.hexdigest() != expected_sha256:
            raise ValueError('R2 video bytes do not match the release manifest')
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
