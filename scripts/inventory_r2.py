"""Read the shared R2 release and inventory all available videos; never run models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moscato.r2_inputs import inventory_from_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='chefost-r2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/r2_release.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/batch-preparation/inventory.json')
    args = parser.parse_args()
    import boto3
    config = json.loads(args.config.read_text())
    client = boto3.Session(profile_name=args.profile, region_name='auto').client('s3', endpoint_url=config['endpoint'])
    response = client.get_object(Bucket=config['bucket'], Key=config['manifest_key'])
    try:
        payload = response['Body'].read()
    finally:
        response['Body'].close()
    objects = {}
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=config['bucket'], Prefix=config['prefix']):
        objects.update({row['Key']: row['Size'] for row in page.get('Contents', [])})
    inventory = inventory_from_manifest(payload, config, objects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(inventory, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'videos': inventory['video_count'],
                      'datasets': dict(Counter(row['dataset'] for row in inventory['videos'])),
                      'unavailable': len(inventory['unavailable']), 'inventory': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
