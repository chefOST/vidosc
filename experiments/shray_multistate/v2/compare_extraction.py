"""Summarize existing extraction timers; this does not rerun either backbone."""
import hashlib
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'broader'))
from protocol import DATA, WORKSPACE, read, write


def main():
    rows = []
    for slow_path in sorted((DATA / 'features/slowfast').glob('*.json')):
        clip_path = DATA / 'features/clip' / slow_path.name
        if '.smoke.' in slow_path.name or not clip_path.exists():
            continue
        slow, clip = read(slow_path), read(clip_path)
        assert not slow.get('smoke_only') and not clip.get('smoke_only')
        assert slow['source_sha256'] == clip['source_sha256']
        assert slow['sample_anchors'] == clip['sample_anchors']
        rows.append({'video': slow['video_id'], 'samples': len(slow['sample_anchors']),
                     'clip_seconds': clip['extraction_seconds'],
                     'slowfast_seconds': slow['extraction_seconds'],
                     'slowfast_over_clip_ratio': slow['extraction_seconds']/clip['extraction_seconds'],
                     'clip_backend': clip['inference_backend'],
                     'slowfast_backend': slow['inference_backend'],
                     'clip_timer_source_sha256': hashlib.sha256(clip_path.read_bytes()).hexdigest(),
                     'slowfast_timer_source_sha256': hashlib.sha256(slow_path.read_bytes()).hexdigest()})
    if not rows:
        raise ValueError('No paired cached extraction timings')
    result = {'shared_videos': len(rows),
              'median_slowfast_over_clip_ratio': statistics.median(r['slowfast_over_clip_ratio'] for r in rows),
              'slowfast_slower_videos': sum(r['slowfast_seconds'] > r['clip_seconds'] for r in rows),
              'source': 'Existing full-video feature metadata extraction_seconds; same source video and sample anchors per pair.',
              'scope': 'Earlier pilot extraction runs, not the v2 test and not a controlled hardware benchmark.',
              'settings': {'clip': 'ViT-L/14 quickgelu, one 224px image per sample, batch 8, float32 DirectML',
                           'slowfast': 'R50 K400, 32 fast/8 slow frames at 256px, one window per call, float16 DirectML'},
              'caveats': ['Wall times include preprocessing/decoding and ordinary system contention; execution times were not isolated.',
                          'Some per-video timers contain very large delays; all paired timers are retained, and a median ratio is reported.',
                          'CLIP timing also includes cached VidOSC context computation; SlowFast timing covers motion extraction.',
                          'Different precision and batching; results do not establish intrinsic backbone speed or a 32x speedup.'],
              'rows': rows}
    write(HERE / 'results/extraction-timing.json', result)
    write(WORKSPACE / 'outputs/shray-accuracy-v2/extraction-timing.json', result)
    print({k: v for k, v in result.items() if k != 'rows'})


if __name__ == '__main__':
    main()
