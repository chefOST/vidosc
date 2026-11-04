"""Pinned hand-object detector image used by the full-video Modal worker."""
from pathlib import Path
import modal

COMMIT = 'e6eec712a498ec7844b97893c8d012cea1a71e09'
WEIGHTS_URL = ('https://raw.githubusercontent.com/ImperialCollegeLondon/'
               'MessagePassingFramework/825297bb9c8a20426ade07d3ab5b3310ed0a6210/'
               'tas_perception/hand_object/models/faster_rcnn_1_8_89999.pth')


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


base_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git')
    .pip_install('torch==2.6.0', 'torchvision==0.21.0', 'numpy==1.26.4',
                 'opencv-python-headless==4.11.0.86', 'easydict==1.13',
                 'PyYAML==6.0.2', 'Pillow==11.1.0', 'scipy==1.15.2',
                 'gdown==5.2.0', 'tqdm==4.67.1')
    .run_commands('git clone https://github.com/ddshan/hand_object_detector.git /opt/hand-object',
                  f'git -C /opt/hand-object checkout {COMMIT}')
    .run_function(download_weights)
)
