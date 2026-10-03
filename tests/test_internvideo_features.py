"""Verify exact frame sampling without downloading a model during tests."""
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moscato.extract_internvideo import window_indices, sampled_clips


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
