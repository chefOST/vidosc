"""Stage 1 (method, blind): LLM proposes attributes per object from recipe name + object names only.
Usage: python generate.py [n_samples]   -> schemas/schema_s{k}.json"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from common import HERE, recipe_objects
import llm

PROMPT = """We track objects in egocentric (head-mounted camera) videos of a person preparing a recipe in a home kitchen. \
The system receives the video, the recipe name and the name of one object, and must report the object's state over time. \
Your job is to define, for each object, the visual attributes that describe its state, so that an image-text model \
(CLIP-style) can read each attribute's value from video frames.

Recipe: {recipe}
Objects: {objects}

For each object give 1-4 attributes that are likely to change, or be informative, while this recipe is prepared. For each attribute:
- "name": short snake_case name.
- "values": 2-4 mutually exclusive values, in natural order when one exists.
- "prompts": for each value, 3 short image captions of the object in that value, with concrete visual cues and the object named.
- "script": the expected sequence of values over a typical video of this recipe, start to end, with consecutive entries different \
(e.g. ["empty", "full", "empty"]); a single value if it does not change.
- "one_way": true if the attribute never reverts to an earlier value.
- "cues": one short caption per transition in "script" (length len(script)-1) of the action that causes it.
Also give, per object, "visible": 2 short captions of the object clearly in view or being handled.

Return JSON only: {{"<object name exactly as given>": {{"visible": [...], "attributes": [...]}}, ...}}"""


def gen(recipe, objects, sample):
    return llm.call(PROMPT.format(recipe=recipe, objects=json.dumps(objects)), tag=f'gen_{recipe}', sample=sample)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ro = recipe_objects()
    os.makedirs(f'{HERE}/schemas', exist_ok=True)
    for s in range(n):
        with ThreadPoolExecutor(5) as ex:
            res = dict(zip(ro, ex.map(lambda r: gen(r, ro[r], s), ro)))
        for r, objs in ro.items():
            miss = [o for o in objs if o not in res[r]]
            assert not miss, (r, miss)
        json.dump(res, open(f'{HERE}/schemas/schema_s{s}.json', 'w'), indent=1)
        print('sample', s, {r: sum(len(v['attributes']) for v in res[r].values()) for r in res})
