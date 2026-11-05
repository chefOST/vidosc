"""Verify exact frame sampling without downloading a model during tests."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moscato.extract_internvideo import window_indices, sampled_clips
from moscato import extract_internvideo


def test_one_second_windows_include_clip_offset_and_partial_tail():
    windows=window_indices(1469,1504,30)
    assert windows[0]==[1469,1473,1477,1481,1485,1489,1493,1498]
    assert windows[1][0]==1499
    assert windows[1][-1]==1503
    assert len(windows[1])==8


@pytest.mark.parametrize('start,end,fps',[(0,0,30),(-1,10,30),(0,10,0),(0,10,float('nan'))])
def test_invalid_ranges_fail(start,end,fps):
    with pytest.raises(ValueError):
        window_indices(start,end,fps)


def test_sequential_decoder_matches_frame_numbers_and_duplicate_tail_samples(tmp_path):
    path=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((32,32,3),i*10,dtype=np.uint8))
    writer.release()
    windows=window_indices(5,18,10)
    decoded=list(sampled_clips(path,windows))
    for expected,images in zip(windows,decoded):
        assert len(images)==8
        for index,image in zip(expected,images):
            assert abs(float(image.mean())-index*10)<3


def test_decoder_rejects_truncated_video(tmp_path):
    path=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    writer.write(np.zeros((32,32,3),dtype=np.uint8));writer.release()
    with pytest.raises(ValueError,match='ended'):
        list(sampled_clips(path,[[0,1]]))


def test_manifest_preserves_sampling_and_provenance(tmp_path, monkeypatch):
    video, checkpoint = tmp_path / 'video', tmp_path / 'checkpoint'
    video.write_bytes(b'video')
    checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(extract_internvideo.subprocess, 'check_output',
                        lambda command, **kwargs: 'pinned-source\n' if command[0] == 'git'
                        else json.dumps({'streams': [{'avg_frame_rate': '30/1'}]}).encode())
    manifest = extract_internvideo.prepare_manifest(video, 'test', 'active-object', checkpoint,
                                                  tmp_path, 0, 27001, 30, 'cpu')
    assert len(manifest['sampled_global_frame_indices']) == 901
    assert manifest['sampled_global_frame_indices'][-1] == [27000] * 8
    assert manifest['interval_starts_seconds'][-1] == 900
    assert manifest['clip_end_seconds'] == 27001 / 30
    assert manifest['feature_backbone'] == 'InternVideo-MM-L-14'
    assert manifest['feature_normalization'] == 'L2'
    assert manifest['upstream_commit'] == 'pinned-source'
    assert manifest['video_sha256'] == extract_internvideo.digest(video)
    assert manifest['checkpoint_sha256'] == extract_internvideo.digest(checkpoint)
    with pytest.raises(ValueError, match='FPS differs'):
        extract_internvideo.prepare_manifest(video, 'test', 'active-object', checkpoint,
                                            tmp_path, 0, 30, 25, 'cpu')
