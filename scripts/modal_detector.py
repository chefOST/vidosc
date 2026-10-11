"""Five-frame hand/object detection smoke test on Modal; no ground truth used.

Run: .venv/bin/python -m modal run scripts/modal_detector.py
Outputs go to a new, timestamped directory under runs/modal-detector/.
"""
from pathlib import Path
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e6eec712a498ec7844b97893c8d012cea1a71e09'
MODEL_ID = '166IM6CXA32f9L6V7-EMd9m8gin6TFpim'
WEIGHTS_URL = ('https://raw.githubusercontent.com/ImperialCollegeLondon/'
               'MessagePassingFramework/825297bb9c8a20426ade07d3ab5b3310ed0a6210/'
               'tas_perception/hand_object/models/faster_rcnn_1_8_89999.pth')
VIDEO = ROOT / 'data/moscato/videos/S29_Salad_7150991-1.avi'
app = modal.App('vidosc-detector-test')
GPU = os.environ.get('VIDOSC_DETECTOR_GPU') or None
if GPU not in (None, 'T4'):
    raise ValueError('VIDOSC_DETECTOR_GPU must be unset or T4')


def download_weights():
    import urllib.request
    import hashlib
    import torch
    Path('/models').mkdir(exist_ok=True)
    path = '/models/hand_object_100k.pth'
    urllib.request.urlretrieve(WEIGHTS_URL, path)
    # GitHub raw URLs return an LFS pointer for this large checkpoint.
    with open(path, 'rb') as handle:
        header = handle.read(512)
    if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
        fields = dict(line.split(' ', 1) for line in header.decode().strip().splitlines())
        expected_hash = fields['oid'].removeprefix('sha256:')
        media_url = WEIGHTS_URL.replace('raw.githubusercontent.com/',
                                       'media.githubusercontent.com/media/')
        urllib.request.urlretrieve(media_url, path)
        payload = Path(path).read_bytes()
        if len(payload) != int(fields['size']) or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError('Checkpoint does not match the pinned Git LFS fingerprint')
    # Never fall back to unrestricted pickle loading.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' not in checkpoint:
        raise ValueError('Unexpected checkpoint format')


image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git')
    .pip_install('torch==2.6.0', 'torchvision==0.21.0', 'numpy==1.26.4',
                 'opencv-python-headless==4.11.0.86', 'easydict==1.13',
                 'PyYAML==6.0.2', 'Pillow==11.1.0', 'scipy==1.15.2',
                 'gdown==5.2.0', 'tqdm==4.67.1')
    .run_commands('git clone https://github.com/ddshan/hand_object_detector.git /opt/hand-object',
                  f'git -C /opt/hand-object checkout {COMMIT}')
    .run_function(download_weights)
    .add_local_file(ROOT / 'moscato/hand_object.py', '/root/hand_object.py')
)


def detect_frames(frames: list[tuple[int, bytes]], video_hash: str, device: str):
    import hashlib
    import io
    import json
    import time
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from hand_object import load_detector, detect_frame

    if len(frames) > 5:
        raise ValueError('Smoke test is limited to five frames')
    started = time.time()
    torch.set_num_threads(2)
    model, cfg = load_detector('/opt/hand-object', '/models/hand_object_100k.pth', device)
    results, pictures = [], {}
    for frame_id, data in frames:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'Cannot decode frame {frame_id}')
        detections = detect_frame(model, cfg, bgr, device)
        result = {'frame_index': frame_id, 'seconds': frame_id / 30,
                  'frame_png_sha256': hashlib.sha256(data).hexdigest(),
                  'detections': detections}
        results.append(result)
        picture = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(picture)
        for d in detections:
            color = 'lime' if d['class'] == 'targetobject' else 'cyan'
            box = d['box_xyxy']
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], max(0, box[1] - 14)),
                      f"{d['class']} {d['score']:.2f}", fill=color)
        draw.rectangle((0, 0, 650, 25), fill='black')
        draw.text((6, 5), f'Frame {frame_id} | green: objects, cyan: hands | threshold 0.5', fill='white')
        buf = io.BytesIO()
        picture.save(buf, format='JPEG', quality=92)
        pictures[f'frame_{frame_id:05d}.jpg'] = buf.getvalue()
        print(f"Frame {frame_id}: {len(detections)} detections", flush=True)
    manifest = {
        'purpose': 'detector smoke test; not VidOSC predictions or evaluation',
        'video_sha256': video_hash, 'source_commit': COMMIT,
        'checkpoint_google_drive_id': MODEL_ID,
        'checkpoint_download_url': WEIGHTS_URL,
        'checkpoint_sha256': hashlib.sha256(Path('/models/hand_object_100k.pth').read_bytes()).hexdigest(),
        'device': device,
        'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'torch': torch.__version__,
        'threshold': 0.5, 'nms_threshold': float(cfg.TEST.NMS),
        'compatibility': 'torchvision ROIAlign aligned=False; inclusive-coordinate NMS',
        'original_extension_numerical_parity_verified': False,
        'vidosc_crop_selection_parity_verified': False,
        'elapsed_seconds': time.time() - started, 'frames': results,
    }
    return manifest, pictures


@app.function(image=image, gpu=GPU, timeout=300, retries=0, max_containers=1,
              scaledown_window=2, memory=8192, cpu=2)
def detect(frames: list[tuple[int, bytes]], video_hash: str):
    import torch
    return detect_frames(frames, video_hash, 'cuda' if torch.cuda.is_available() else 'cpu')


@app.local_entrypoint()
def main():
    import hashlib
    import json
    from datetime import datetime, timezone
    import cv2

    # Decode sequentially: exact zero-based frame indices, no AVI random seeking.
    indices = [1469, 2800, 3000, 3700, 4600]
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)
    frames = []
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        for index in range(max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Video ended at frame {index}')
            if index in indices:
                ok, encoded = cv2.imencode('.png', frame)
                if not ok:
                    raise ValueError('Frame encoding failed')
                frames.append((index, encoded.tobytes()))
    finally:
        cap.release()
    manifest, pictures = detect.remote(frames, hashlib.sha256(VIDEO.read_bytes()).hexdigest())
    folder = ROOT / 'runs/modal-detector' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'detections.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name, content in pictures.items():
        (folder / name).write_bytes(content)
    from PIL import Image
    sheet = Image.new('RGB', (1024, 1230), '#222222')
    for index, name in enumerate(sorted(pictures)):
        with Image.open(folder / name) as frame:
            frame.thumbnail((512, 384))
            sheet.paste(frame, ((index % 2) * 512, (index // 2) * 410))
    sheet.save(folder / 'contact_sheet.jpg')
    print(f'Detector test complete. Outputs: {folder}')
