"""Download only the named CMU videos from the frozen 30-video subset."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = json.loads((ROOT / 'runs/batch-preparation/download_inventory.json').read_text())
    expected = set(json.loads((ROOT / 'configs/moscato_cmu30_v1.json').read_text())['video_ids'])
    if {r['video_id'] for r in records} != expected:
        raise ValueError('Download inventory does not match the frozen subset')
    size = sum(r.get('archive_bytes', 0) for r in records)
    if any(r.get('status') != 200 or not r.get('archive_bytes') for r in records):
        raise ValueError('Resolve unavailable archives before download')
    if shutil.disk_usage(ROOT).free < size * 3:
        raise ValueError('Insufficient free disk space for archives and extracted videos')
    dest = ROOT / 'data/moscato/videos'
    dest.mkdir(parents=True, exist_ok=True)
    for r in records:
        video = dest / (r['video_id'] + '.avi')
        if video.exists():
            print('Present:', video.name, flush=True)
            continue
        archive = dest / Path(r['url']).name
        if not archive.exists():
            temp = archive.with_suffix('.download')
            urllib.request.urlretrieve(r['url'], temp)
            if temp.stat().st_size != r['archive_bytes']:
                raise ValueError('Incomplete archive: ' + str(temp))
            temp.replace(archive)
        with zipfile.ZipFile(archive) as z:
            matches = [n for n in z.namelist() if Path(n).name == video.name]
            if len(matches) != 1:
                raise ValueError('Expected exactly one matching video: ' + video.name)
            temp = video.with_suffix('.partial')
            with z.open(matches[0]) as source, temp.open('wb') as output:
                shutil.copyfileobj(source, output)
            temp.replace(video)
        sha = hashlib.sha256()
        with video.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                sha.update(block)
        (dest / (video.stem + '.source.json')).write_text(json.dumps({
            'source_url': r['url'], 'archive_bytes': r['archive_bytes'],
            'video_sha256': sha.hexdigest(), 'archive_crc_verified_on_read': True,
            'author_checksum_available': False}, indent=2))
        print('Downloaded:', video.name, flush=True)


if __name__ == '__main__':
    main()
