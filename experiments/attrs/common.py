"""Shared data access for the attribute pipeline (ChefOST experiment)."""
import os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from pipeline import data  # noqa: E402

ds = data.Moscato(ROOT + '/../chefOST-data/MOSCATO', cache_dir=ROOT + '/data/cache')
recipe = lambda vid: vid.split('_')[1]


def recipe_objects():
    """Pipeline input: recipe name (from the video id) -> object names to track."""
    r = defaultdict(set)
    for v in ds.videos:
        r[recipe(v)].update(ds.objects(v))
    return {k: sorted(v) for k, v in sorted(r.items())}
