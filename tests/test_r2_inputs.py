import hashlib
import json

import pytest

from moscato.r2_inputs import inventory_from_manifest, relative_key, stage_video


def release():
    entries = []
    for video_id, dataset, suffix, fps, frames in [
        ('S08_Eggs', 'CMU', '_copies/MP4/CMU/Eggs/S08_Eggs.mp4', '30/1', 6243),
        ('P02-Pasta', 'EGTEA', 'EGTEA/Pasta/P02-Pasta.mp4', '24/1', 43094),
    ]:
        entries.append({'ground_truth_video_id': video_id, 'dataset': dataset,
                        'local_path': suffix, 'download_status': 'downloaded_verified_identity',
                        'bytes': 10, 'sha256': 'a' * 64,
                        'media': {'streams': [{'codec_type': 'video', 'avg_frame_rate': fps,
                                               'nb_frames': str(frames)}]}})
    entries.append({'ground_truth_video_id': 'missing', 'dataset': 'EgoPER', 'download_status': 'blocked_access'})
    payload = json.dumps({'entries': entries}).encode()
    config = {'prefix': 'release/videos/', 'release_sha256': hashlib.sha256(payload).hexdigest()}
    objects = {config['prefix'] + e['local_path']: e['bytes'] for e in entries[:2]}
    return payload, config, objects


def test_every_available_video_uses_full_manifest_path_and_own_fps():
    inventory = inventory_from_manifest(*release())
    assert inventory['video_count'] == 2
    cmu, egtea = inventory['videos']
    assert cmu['r2_key'] == 'release/videos/_copies/MP4/CMU/Eggs/S08_Eggs.mp4'
    assert cmu['duration_seconds'] == 208.1
    assert egtea['fps'] == 24 and egtea['duration_seconds'] == 43094 / 24
    assert inventory['unavailable'] == [{'video_id': 'missing', 'dataset': 'EgoPER', 'reason': 'blocked_access'}]


@pytest.mark.parametrize('fault', ['manifest_hash', 'missing_video', 'size', 'unlisted_video'])
def test_release_discrepancies_fail_before_planning(fault):
    payload, config, objects = release()
    if fault == 'manifest_hash':
        payload += b'\n'
    elif fault == 'missing_video':
        objects.pop(next(iter(objects)))
    elif fault == 'size':
        objects[next(iter(objects))] += 1
    else:
        objects['release/videos/unknown.mp4'] = 10
    with pytest.raises(ValueError):
        inventory_from_manifest(payload, config, objects)


@pytest.mark.parametrize('key', ['elsewhere/video.mp4', 'release/../secret', 'release//absolute', 'release/'])
def test_mount_paths_cannot_escape_release(key):
    with pytest.raises(ValueError):
        relative_key(key, 'release/')


def test_staged_video_requires_exact_bytes_before_replacing_destination(tmp_path):
    source, dest = tmp_path / 'r2.mp4', tmp_path / 'local.mp4'
    source.write_bytes(b'verified video')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert stage_video(source, dest, expected, source.stat().st_size) == dest
    assert dest.read_bytes() == source.read_bytes()
    source.write_bytes(b'changed video!')
    with pytest.raises(ValueError, match='do not match'):
        stage_video(source, dest, expected, source.stat().st_size)
    assert dest.read_bytes() == b'verified video'
    assert not dest.with_suffix('.partial').exists()
